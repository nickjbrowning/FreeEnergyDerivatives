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
import solvation_potentials as sp

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

timestep_ps = 0.001
total_simulation_time = 1000  # 1 ns in units of 0.001 ps


def collect_solute_indexes(topology):
    soluteIndices = []
    for res in topology.residues():
        resname = res.name.upper()
        if (resname != 'HOH' and resname != 'WAT'and resname != 'CL'and resname != 'NA'):
            for atom in res.atoms():
                soluteIndices.append(atom.index)
    return soluteIndices

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
        nonbondedCutoff=9.0 * unit.angstroms, constraints=HBonds, switchDistance=7.5 * unit.angstroms)

solute_indexes = collect_solute_indexes(modeller.topology)

system = sp.create_alchemical_system(system, solute_indexes)

for idx in solute_indexes:
   system.setParticleMass(idx, 0.0)

'''
---FINISHED SYSTEM PREPARATION---
'''
    
# system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
# integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, timestep_ps * unit.picoseconds)
# integrator.setConstraintTolerance(1.0E-08)

integrator = VerletIntegrator(timestep_ps * unit.picoseconds)

simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.context.setVelocitiesToTemperature(298.15 * unit.kelvin)

simulation.minimizeEnergy()

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

energies = []
temperature = []
start = time()

for iteration in range(np.int(total_simulation_time / (timestep_ps * 500))):
    integrator.step(500)  
    
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() + state.getKineticEnergy()
    
    energies.append(energy._value)

simulation.reporters.append(StateDataReporter('data.txt', 500, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', 500))

energies = np.array(energies)

epsl = (1.0 / len(energies)) * np.sum(np.abs((energies - np.mean(energies)) / np.mean(energies)))

print ("timestep: ", timestep_ps, "epsilon:", epsl, "log(eps)", np.log(epsl))

end = time()
print (end - start)

import matplotlib.pyplot as plt

plt.plot(np.arange(len(energies)), energies - energies[0])

plt.show()

