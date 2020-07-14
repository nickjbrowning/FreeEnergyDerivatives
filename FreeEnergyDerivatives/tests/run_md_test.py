from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from mdtraj.reporters import NetCDFReporter   
from openmmtools import alchemy
import numpy as np
from time import time

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator
from openmmtools.forces import find_forces
from openmmtools.constants import ONE_4PI_EPS0
import copy
from openmmtools import forcefactories
from openmmtools import forces

from openmmtools.alchemy import  *

from lib import solvation_potentials as sp
from pathlib import Path


def minimize(system, initial_pos, integrator):
    alchemical_system = sp.create_alchemical_system(system, solute_indexes, softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=False)
    context = Context(system, integrator) 
    context.setPositions(initial_pos)
    
    minimizer = LocalEnergyMinimizer.minimize(context)
    
    return context.getState(getPositions=True).getPositions()
    

def collect_solute_indexes(topology):
    soluteIndices = []
    for res in topology.residues():
        resname = res.name.upper()
        if (resname != 'HOH' and resname != 'WAT'and resname != 'CL'and resname != 'NA'):
            for atom in res.atoms():
                soluteIndices.append(atom.index)
    return soluteIndices


platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

base_path = Path(__file__).parent

'''
---SYSTEM PREPARATION---
    setup AM1-BCC charges for the solute, add solvent, set non-bonded method etc
'''

file_path = (base_path / 'ethanol.sdf').resolve()
print (file_path)

ligand_mol = Molecule.from_file('ethanol.sdf', file_format='sdf')

forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass': 4 * unit.amu }

system_generator = SystemGenerator(
    forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'],
    small_molecule_forcefield='gaff-2.11',
    molecules=[ligand_mol],
    forcefield_kwargs=forcefield_kwargs)

file_path = (base_path / 'ethanol.pdb').resolve()

ligand_pdb = PDBFile('ethanol.pdb')

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=12.0 * unit.angstroms)

system = system_generator.forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=9.0 * unit.angstroms, constraints=HBonds, switchDistance=7.5 * unit.angstroms)

solute_indexes = collect_solute_indexes(modeller.topology)

alchemical_system = sp.create_alchemical_system(system, solute_indexes, softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=True)

'''
---FINISHED SYSTEM PREPARATION---
'''
    
alchemical_system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
integrator.setIntegrationForceGroups(set([0]))

positions = minimize(alchemical_system, modeller.positions, integrator)

simulation = app.Simulation(modeller.topology, alchemical_system, integrator, platform)
simulation.context.setPositions(positions)

# simulation.minimizeEnergy()

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

simulation.reporters.append(StateDataReporter('data.txt', 100, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', 100))

start = time()
simulation.step(100000)
end = time()
print (end - start)

