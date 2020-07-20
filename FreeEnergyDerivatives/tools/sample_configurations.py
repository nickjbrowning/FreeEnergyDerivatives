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
parser.add_argument('-solvate', type=bool, default=True)
parser.add_argument('-nsamples', type=int, default=250)  
parser.add_argument('-nsample_steps', type=int, default=10000)  # 20ps using 2fs timestep
parser.add_argument('-solute_indexes', type=int, nargs='+', default=None)
args = parser.parse_args()

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')


def collect_solute_indexes(topology):
    soluteIndices = []
    for res in topology.residues():
        resname = res.name.upper()
        if (resname != 'HOH' and resname != 'WAT'and resname != 'CL'and resname != 'NA'):
            for atom in res.atoms():
                soluteIndices.append(atom.index)
    return soluteIndices


if (args.solute_indexes == None):
    solute_indexes = collect_solute_indexes(modeller.topology)
else:
    solute_indexes = np.array(args.solute_indexes)
    
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

if (args.solvate):
    modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=12.0 * unit.angstroms)

system = system_generator.forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds, switch_distance=9.0 * unit.angstroms)
    
'''
---FINISHED SYSTEM PREPARATION---
'''

# Add a simple barostat for pressure control
system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
# Use a simple thermostat for T control
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
integrator.setConstraintTolerance(1.0E-08)

simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# fix any bad contacts etc
simulation.minimizeEnergy()

# lets equilibrate the system for 200 ps first
simulation.step(100000)

for iteration in range(args.nsamples):
    
    state = simulation.context.getState(getPositions=True)
    PDBFile.writeFile(ligand_pdb.topology, state.getPositions()[solute_indexes, :], file=open("sample_" + str(iteration) + ".pdb", "w"))
    
    simulation.step(args.nsample_steps)

