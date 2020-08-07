from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from parmed.openmm.reporters import NetCDFReporter  
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
from lib import utils

from openmmqml import QMLForce

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

ligand_pdb = PDBFile('alanine_dipeptide.pdb')
forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds, switchDistance=9.0 * unit.angstroms)

solute_indexes = utils.collect_solute_indexes(modeller.topology)

training_atoms = np.fromfile('atoms.bin')
training_charges = np.fromfile('charges.bin')
training_coefficients = np.fromfile('coefficients.bin')
training_reps = np.fromfile('training_reps.bin')

elements2charge = {'H': 1, 'C': 6, 'N': 7, 'O':8}
charges = []
for atom in modeller.topology.atoms():
    charges.append(elements2charge[atom.element.symbol])

print (charges, solute_indexes)

qmlforce = QMLForce(charges, solute_indexes)

qmlforce.setSigma(21.2)

qmlforce.setTrainingReps(training_reps)
qmlforce.setCoefficients(coefficients)
qmlforce.setTrainingNumAtoms(training_atoms)
qmlforce.setTrainingCharges(training_charges)

qmlforce.setElements([[1, 6, 7, 8]])
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
qmlforce.setSigma(sigma)

system.addForce(qmlforce)

system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

simulation.minimizeEnergy()

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

simulation.reporters.append(StateDataReporter('data.txt', 500, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', 500))

start = time()
simulation.step(10000)
end = time()
print (end - start)

