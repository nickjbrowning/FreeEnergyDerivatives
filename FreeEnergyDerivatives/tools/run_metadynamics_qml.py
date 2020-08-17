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
from openmmqml import QMLForce

import argparse

platform = openmm.Platform.getPlatformByName('Reference')
# platform.setPropertyDefaultValue('Precision', 'mixed')

parser = argparse.ArgumentParser()

parser.add_argument('-pdb', type=str, help='PDB Structure File')
parser.add_argument('-script', type=str, help='Plumed Input File')

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

system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds)

solute_indexes = utils.collect_solute_indexes(modeller.topology)

'''
---FINISHED SYSTEM PREPARATION---
'''
    
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

system.addForce(PlumedForce(script))

training_atoms = np.fromfile('atoms.bin', dtype=np.int32).tolist()
training_charges = np.fromfile('charges.bin', dtype=np.int32).tolist()
training_coefficients = np.fromfile('coefficients.bin', dtype=np.double).tolist()
training_reps = np.fromfile('training_reps.bin', dtype=np.double).tolist()

elements2charge = {'H': 1, 'C': 6, 'N': 7, 'O':8}
charges = []
for atom in modeller.topology.atoms():
    charges.append(elements2charge[atom.element.symbol])

print (charges, solute_indexes)

qmlforce = QMLForce(charges, solute_indexes)

qmlforce.setSigma(21.2)

qmlforce.setTrainingReps(training_reps)
qmlforce.setCoefficients(training_coefficients)
qmlforce.setTrainingNumAtoms(training_atoms)
qmlforce.setTrainingCharges(training_charges)

qmlforce.setElements([1, 6, 7, 8])
qmlforce.setNRs2(24)
qmlforce.setNRs3(15)
qmlforce.setNFourrier(1)
qmlforce.setEta2(0.32)
qmlforce.setEta3(2.7)
qmlforce.setZeta(np.pi)
qmlforce.setRcut(8.0)
qmlforce.setAcut(8.0)
qmlforce.setTwoBodyDecay(1.8)
qmlforce.setThreeBodyDecay(0.57)
qmlforce.setThreeBodyWeight(13.4)
qmlforce.setSigma(21.2)

system.addForce(qmlforce)

simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

simulation.minimizeEnergy()

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

simulation.reporters.append(StateDataReporter('data.txt', 1500, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', 1500))

start = time()
simulation.step(7500000)
end = time()
print (end - start)

