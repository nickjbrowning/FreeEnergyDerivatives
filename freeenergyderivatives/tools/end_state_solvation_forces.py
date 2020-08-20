from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
import numpy as np
from time import time
from parmed.openmm.reporters import NetCDFReporter

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator

from freeenergyderivatives.lib import utils
from freeenergyderivatives.lib import solvation_potentials as sp

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-pdb', type=str)
parser.add_argument('-nsamples', type=int, default=3000)
parser.add_argument('-nsample_steps', type=int, default=500)

args = parser.parse_args()

platform = openmm.Platform.getPlatformByName('OpenCL')
platform.setPropertyDefaultValue('Precision', 'mixed')

ligand_pdb = PDBFile(args.pdb)

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')

modeller.addSolvent(forcefield, model='tip3p', padding=12.0 * unit.angstroms)

system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds, switch_distance=9.0 * unit.angstroms)

solute_indexes = utils.collect_solute_indexes(modeller.topology)

system, groups = sp.create_end_state_system(system, solute_indexes)

'''
---FINISHED SYSTEM PREPARATION---
'''

for idx in solute_indexes:
    system.setParticleMass(idx, 0.0)

system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))  
  
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
integrator.setIntegrationForceGroups(groups['integration'])

simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# fix any bad contacts etc
simulation.minimizeEnergy()

print ("Equilibrating system for 0.5ns")
simulation.step(250000)
print ("Finished equilibrating system")

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

simulation.reporters.append(StateDataReporter('data.txt', args.nsample_steps, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', args.nsample_steps))

avg_forces = np.zeros((len(solute_indexes), 3))
avg_energy = 0.0

for i in range(args.nsamples):
    
    simulation.step(args.nsample_steps)
    
    state = simulation.context.getState(getForces=True, getEnergy=True, groups=groups['interaction'])

    forces = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]
    energy = state.getPotentialEnergy().value_in_unit_system(unit.md_unit_system)
    
    avg_forces += (1 / args.nsamples) * forces
    avg_energy += (1 / args.nsamples) * energy

print ("-- CALCULATION DONE --")
print ("-Hydration Enthapy (kj/mol):", avg_energy)
print ("-Hydration Forces-")
print (avg_forces)

avg_forces.tofile('avg_forces.npy')

