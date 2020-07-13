from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from mdtraj.reporters import NetCDFReporter  

from openmmtools import alchemy
import numpy as np

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator

from lib import solvation_potentials as sp
from lib import thermodynamic_integration as TI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sdf', type=str)
parser.add_argument('-pdb', type=str)
parser.add_argument('-freeze_atoms', type=bool, default=False)
parser.add_argument('-compute_forces', type=bool, default=False)
parser.add_argument('-solute_indexes', type=int, nargs='+', default=None)
parser.add_argument('-nelectrostatic_points', type=int, default=10)
parser.add_argument('-nsteric_points', type=int, default=20)
parser.add_argument('-nsamples', type=int, default=2500)  # 1ns 
parser.add_argument('-nsample_steps', type=int, default=100)  # 0.5ps using 2fs timestep
args = parser.parse_args()

    
def collect_solute_indexes(topology):
    soluteIndices = []
    for res in topology.residues():
        resname = res.name.upper()
        if (resname != 'HOH' and resname != 'WAT'and resname != 'CL'and resname != 'NA'):
            for atom in res.atoms():
                soluteIndices.append(atom.index)
    return soluteIndices


platform = openmm.Platform.getPlatformByName('OpenCL')
platform.setPropertyDefaultValue('Precision', 'mixed')

'''
---SYSTEM PREPARATION---
    setup AM1-BCC charges for the solute, add solvent, set non-bonded method etc
'''
ligand_mol = Molecule.from_file(args.sdf, file_format='sdf')

forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass': 4 * unit.amu }

system_generator = SystemGenerator(
    forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'],
    small_molecule_forcefield='gaff-2.11',
    molecules=[ligand_mol],
    forcefield_kwargs=forcefield_kwargs)

ligand_pdb = PDBFile(args.pdb)

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
    solute_indexes = collect_solute_indexes(modeller.topology)
else:
    solute_indexes = np.array(args.solute_indexes)

alchemical_system = sp.create_alchemical_system(system, solute_indexes, softcore_beta=0.0, softcore_m=1.0)

# freeze solute
if (args.freeze_atoms):
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

simulation = app.Simulation(modeller.topology, alchemical_system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# below corresponds to fully interacting state
simulation.context.setParameter('lambda_electrostatics', 1.0)
simulation.context.setParameter('lambda_sterics', 1.0)

# fix any bad contacts etc
simulation.minimizeEnergy()

# lets equilibrate the system for 200 ps first
simulation.step(100000)

# simulation.reporters.append(StateDataReporter('data.txt', args.nsample_steps, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
# simulation.reporters.append(NetCDFReporter('output.nc', args.nsample_steps))

# write out the current state for visual inspection
state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

'''
---THERMODYNAMIC INTEGRATION---
    sample dV/dL using two paths:
        1) slowly dcouple electrostatics between solute and solvent
        2) then slowly decouple steric interactions
        3) final dG estimate is then the dG of 1) + 2)
'''
electrostatics_grid = np.linspace(1.0, 0.0, args.nelectrostatic_points)

dV_electrostatics, dVe_forces = TI.collect_dvdl_values(simulation, electrostatics_grid, args.nsamples, args.nsample_steps, solute_indexes, lambda_var='lambda_electrostatics', compute_forces=args.compute_forces)

dG_electrostatics = np.trapz(np.mean(dV_electrostatics, axis=1), x=electrostatics_grid[::-1])

print ("dG electrostatics,", dG_electrostatics)

if (args.compute_forces):
    dG_electrostatics_forces = np.trapz(np.mean(dVe_forces, axis=1), x=electrostatics_grid[::-1], axis=0)
    print ("dG electrostatics forces", dG_electrostatics_forces)

sterics_grid = np.linspace(1.0, 0.0, args.nsteric_points)

print (simulation.context.getParameter('lambda_electrostatics'), simulation.context.getParameter('lambda_sterics'))

dV_sterics, dVs_forces = collect_dvdl_values(simulation, sterics_grid, args.nsamples, args.nsample_steps, solute_indexes, lambda_var='lambda_sterics', compute_forces=args.compute_forces)

dG_sterics = np.trapz(np.mean(dV_sterics, axis=1), x=sterics_grid[::-1])
print ("dG sterics,", dG_sterics)

if (args.compute_forces):
    dG_sterics_forces = np.trapz(np.mean(dVs_forces, axis=1), x=sterics_grid[::-1], axis=0)
    print ("dG sterics forces", dG_sterics_forces)

print ("dG", dG_electrostatics + dG_sterics)

if (args.compute_forces):
    print ("dG forces", dG_electrostatics_forces + dG_sterics_forces)

# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
#  
# plt.plot(electrostatics_grid, dVe, label='electrostatics')
# plt.plot(sterics_grid, dVs, label='sterics')
# plt.legend()
#  
# plt.xlabel(r'$\lambda$')
# plt.ylabel(r'$\left <\frac{\partial U}{\partial \lambda}\right >$')
#  
# plt.savefig('dvdl.png')
