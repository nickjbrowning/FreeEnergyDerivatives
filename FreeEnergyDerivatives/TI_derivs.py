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

modeller.addSolvent(system_generator.forcefield, model='tip4pew', padding=10.0 * unit.angstroms)

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
integrator.setIntegrationForceGroups(set([0]))  # only want to integrate forces coming from force group 0 - force group 1 are the d^2V/dldR forces

context = Context(alchemical_system, integrator, platform)

context.setPositions(modeller.positions)

# below corresponds to fully interacting state

context.setParameter('lambda_electrostatics', 1.0)
context.setParameter('lambda_sterics', 1.0)

# fix any bad contacts etc
LocalEnergyMinimizer.minimize(context, maxIterations=1000)

# lets equilibrate the system for 200 ps first
integrator.step(100000)

# write out the current state for visual inspection
state = context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

'''
---THERMODYNAMIC INTEGRATION---
    sample dV/dL using two paths:
        1) slowly dcouple electrostatics between solute and solvent
        2) then slowly decouple steric interactions
        3) final dG estimate should then be the dG of 1) + 2)
'''

# electrostatics "should" be smoother than the LJ part, so "should" be able to get away with fewer lambda points
electrostatics_grid = np.linspace(1.0, 0.0, 10)
print ("ELECTROSTATICS")
print ("lambda", "<dV/dl>")

dVe = []
avg_forces = []

for l in electrostatics_grid:
    
    context.setParameter('lambda_electrostatics', l)
    
    # equilibrate for 100ps before collecting data for 1ns, taking dV/dl every 1ps
    integrator.step(50000)

    Es = []
    dV = []
    
    sample_forces = []
    
    # data collection loop
    for iteration in range(1000):
        integrator.step(500)  
        
        state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))
        energy = state.getPotentialEnergy()
        energy_derivs = state.getEnergyParameterDerivatives()
        # test = state.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]
        
        energy_deriv = energy_derivs['lambda_electrostatics']
        Es.append(energy._value)
        dV.append(energy_deriv)
        
        state_deriv = context.getState(getForces=True, groups=set([1]))
        forces = state_deriv.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]

        sample_forces.append(forces)

    avg_forces.append(np.average(sample_forces, axis=0))
    dVe.append(np.average(dV))
    
    print ("%5.2f %5.2f" % (l, np.average(dV)))

# use trapezoidal to integrate
dG_electrostatics = np.trapz(dVe, x=electrostatics_grid)
dG_electrostatics_forces = np.trapz(avg_forces, x=electrostatics_grid, axis=0)

print ("dG electrostatics,", dG_electrostatics)
print ("dG electrostatics forces", dG_electrostatics_forces)

dVs = []
avg_forces = []

context.setParameter('lambda_electrostatics', 0.0)
    
sterics_grid = np.linspace(1.0, 0.0, 20)

print ("STERICS")
print ("lambda", "<dV/dl>")
for l in sterics_grid:
    
    context.setParameter('lambda_sterics', l)
    
    integrator.step(50000)

    Es = []
    dV = []
    
    sample_forces = []
    
    for iteration in range(1000):
        integrator.step(500) 
        
        state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))
        energy = state.getPotentialEnergy()
        energy_derivs = state.getEnergyParameterDerivatives()
        
        energy_deriv = energy_derivs['lambda_sterics']
        Es.append(energy._value)
        dV.append(energy_deriv)
        
        state_deriv = context.getState(getForces=True, groups=set([1]))
        forces = state_deriv.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]
        sample_forces.append(forces)

    avg_forces.append(np.average(sample_forces, axis=0))
    dVs.append(np.average(dV))
    
    print ("%5.2f %5.2f" % (l, np.average(dV)))
 
dG_sterics = np.trapz(dVs, x=sterics_grid)
dG_sterics_forces = np.trapz(avg_forces, x=sterics_grid, axis=0)

print ("dG sterics,", dG_sterics)
print ("dG sterics forces", dG_sterics_forces)

print ("dG", -dG_electrostatics - dG_sterics)
print ("dG forces", -dG_electrostatics_forces - dG_sterics_forces)

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
 
plt.plot(electrostatics_grid, dVe, label='electrostatics')
plt.plot(sterics_grid, dVs, label='sterics')
plt.legend()
 
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\left <\frac{\partial U}{\partial \lambda}\right >$')
 
plt.savefig('dvdl.png')
