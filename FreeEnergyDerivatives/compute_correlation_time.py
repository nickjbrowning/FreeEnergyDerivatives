from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile

from openmmtools import alchemy
import numpy as np

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator

from lib import solvation_potentials as sp
from lib import thermodynamic_integration as TI

import argparse

# from astropy.stats import jackknife_stats
# from astropy.stats import jackknife_resampling

from pymbar import timeseries

parser = argparse.ArgumentParser()
parser.add_argument('-sdf', type=str)
parser.add_argument('-pdb', type=str)
parser.add_argument('-freeze_atoms', type=bool, default=False)
parser.add_argument('-compute_solvation_response', type=bool, default=False)
parser.add_argument('-solute_indexes', type=int, nargs='+', default=None)
parser.add_argument('-lmbda', type=float, default=0.5)

args = parser.parse_args()


def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


platform = openmm.Platform.getPlatformByName('OpenCL')
platform.setPropertyDefaultValue('Precision', 'Mixed')

'''
---SYSTEM PREPARATION---
    setup AM1-BCC charges for the solute, add solvent, set non-bonded method etc
'''
ligand_mol = Molecule.from_file('ethanol.sdf', file_format='sdf')

forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass': 4 * unit.amu }

system_generator = SystemGenerator(
    forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'],
    small_molecule_forcefield='gaff-2.11',
    molecules=[ligand_mol],
    forcefield_kwargs=forcefield_kwargs)

ligand_pdb = PDBFile('ethanol.pdb')

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=10.0 * unit.angstroms)

system = system_generator.forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=9.0 * unit.angstroms, constraints=HBonds, switch_distance=7.5 * unit.angstroms)
    
'''
---FINISHED SYSTEM PREPARATION---
'''

'''
---ALCHEMICAL CONFIGURATION---
    define solute indexes, set up the alchemical region + factory, and specify steric/electrostatic lambda coupling for solvation
'''
# determines solute indexes

if (args.solute_indexes == None):
    solute_indexes = utils.collect_solute_indexes(modeller.topology)
    
alchemical_system = sp.create_alchemical_system(system, solute_indexes, softcore_beta=0.0, softcore_m=1.0)

if (args.freeze_atoms):
    # freeze solute
    for idx in solute_indexes:
        alchemical_system.setParticleMass(idx, 0.0)
    
'''
---FINISHED ALCHEMICAL CONFIGURATION---
'''

# Add a simple barostat for pressure control
alchemical_system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
# Use a simple thermostat for T control
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
integrator.setConstraintTolerance(1.0E-08)
# integrator.setIntegrationForceGroups(set([0]))  # only want to integrate forces coming from force group 0 - force group 1 are the d^2V/dldR forces

simulation = app.Simulation(modeller.topology, alchemical_system, integrator, platform)

simulation.context.setPositions(modeller.positions)

# below corresponds to fully interacting state

print ("lambda", args.lmbda)

simulation.context.setParameter('lambda_electrostatics', args.lmbda)
simulation.context.setParameter('lambda_sterics', 1.0)

# fix any bad contacts etc
simulation.minimizeEnergy()

# lets equilibrate the system for 500 ps first
simulation.step(250000)

# write out the current state for visual inspection
state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

# data collection

N = 2500
n = 100

electrostatics_grid = [args.lmbda]
dV, sample_forces = TI.collect_dvdl_values(simulation, electrostatics_grid, N, n, solute_indexes, lambda_var='lambda_electrostatics')

dV = dV[0, :]

# estimate, bias, stderr, conf_interval = jackknife_stats(dV, np.mean)
# print ("Jackknife stats:")
# print (estimate, bias, stderr, conf_interval)

print ("electrostatics equilibration time")
[t0, g, Neff_max] = timeseries.detectEquilibration(dV)  # compute indices of uncorrelated timeseries
dV_equil = dV[t0:]

print (t0, g, Neff_max)

indices = timeseries.subsampleCorrelatedData(dV_equil, g=g)

dV = dV_equil[indices]

print (np.average(dV), np.std(dV))

simulation.context.setParameter('lambda_electrostatics', 0.0)
simulation.context.setParameter('lambda_sterics', args.lmbda)

simulation.step(100000)

sterics_grid = [args.lmbda]
dV, sample_forces = TI.collect_dvdl_values(simulation, sterics_grid, N, n, solute_indexes, lambda_var='lambda_sterics')
dV = dV[0, :]

print ("sterics equilibration time")

# estimate, bias, stderr, conf_interval = jackknife_stats(dV, np.mean)
# print (estimate, bias, stderr, conf_interval)

[t0, g, Neff_max] = timeseries.detectEquilibration(dV)  # compute indices of uncorrelated timeseries
dV_equil = dV[t0:]

print (t0, g, Neff_max)
indices = timeseries.subsampleCorrelatedData(dV_equil, g=g)
dV = dV_equil[indices]

print (np.average(dV), np.std(dV))
