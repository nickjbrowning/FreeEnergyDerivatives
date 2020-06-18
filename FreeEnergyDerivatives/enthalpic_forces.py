from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile

from openmmtools import alchemy
import numpy as np

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator
import solvation_potentials as sp


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
ligand_mol = Molecule.from_file('ethanol.sdf', file_format='sdf')

forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass': 4 * unit.amu }

system_generator = SystemGenerator(
    forcefields=['amber/protein.ff14SB.xml', 'amber/tip4pew_standard.xml', 'amber/tip4pew_HFE_multivalent.xml'],
    small_molecule_forcefield='gaff-2.11',
    molecules=[ligand_mol],
    forcefield_kwargs=forcefield_kwargs)

ligand_pdb = PDBFile('ethanol.pdb')

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

modeller.addSolvent(system_generator.forcefield, model='tip4pew', padding=12.0 * unit.angstroms)

alchemical_system = system_generator.forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=9.0 * unit.angstroms, constraints=HBonds)

'''
---FINISHED SYSTEM PREPARATION---
'''

'''
---ALCHEMICAL CONFIGURATION---
    define solute indexes, set up the alchemical region + factory, and specify steric/electrostatic lambda coupling for solvation
'''
# determines solute indexes
solute_indexes = collect_solute_indexes(modeller.topology)
sp.create_alchemical_system_rxnfield(alchemical_system, solute_indexes, cutoff=9.0 * unit.angstroms)

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

context = Context(alchemical_system, integrator, platform)

context.setPositions(modeller.positions)

# below corresponds to fully interacting state

context.setParameter('lambda_electrostatics', 1.0)
context.setParameter('lambda_sterics', 1.0)

# fix any bad contacts etc
LocalEnergyMinimizer.minimize(context, maxIterations=1000)

# lets equilibrate the system for 100 ps first
integrator.step(50000)

sample_forces = []

for iteration in range(1000):
    integrator.step(250)  
    
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]

    sample_forces.append(forces)

avg_forces = np.average(sample_forces, axis=0)

print ('avg_forces, ', avg_forces)
solute_system = system_generator.forcefield.createSystem(ligand_pdb.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=9.0 * unit.angstroms, constraints=HBonds)

integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
integrator.setConstraintTolerance(1.0E-08)

context = Context(solute_system, integrator, platform)
context.setPositions(ligand_pdb.positions)

state = context.getState(getForces=True)
forces = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)

print ('solute_forces', forces)

print ('---')

print (avg_forces - forces)
