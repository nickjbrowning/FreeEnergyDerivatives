from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from parmed.openmm.reporters import  NetCDFReporter
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

from lib import utils

from openmmplumed  import *

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

parser = argparse.ArgumentParser()

parser.add_argument('-pdb', type=str, help='PDB Structure File')
parser.add_argument('-script', type=str, help='Plumed Input File')
parser.add_argument('-solvate', type=bool, default=True, help='solvate system yes/no')

args = parser.parse_args()

script = open(args.script, 'r').read()

print ("--Plumed Script--")
print (script)

'''
---SYSTEM PREPARATION---
    setup AM1-BCC charges for the solute, add solvent, set non-bonded method etc
'''

ligand_pdb = PDBFile(args.pdb)

forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

if (args.solvate):
    modeller.addSolvent(forcefield, model='tip3p', padding=12.0 * unit.angstroms)

system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds, switchDistance=9.0 * unit.angstroms)

solute_indexes = utils.collect_solute_indexes(modeller.topology)

'''
---FINISHED SYSTEM PREPARATION---
'''
    
system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

system.addForce(PlumedForce(script))

simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

simulation.minimizeEnergy()

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

simulation.reporters.append(StateDataReporter('data.txt', 1500, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', 1500))

start = time()
simulation.step(15000000)
end = time()
print (end - start)

