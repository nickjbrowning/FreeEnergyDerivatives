from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from simtk.openmm import *

from openmmtools import alchemy
import numpy as np

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator


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
    forcefields=['amber/ff14SB.xml', 'amber/tip4pew_standard.xml'],
    small_molecule_forcefield='gaff-2.11',
    molecules=[ligand_mol],
    forcefield_kwargs=forcefield_kwargs)

ligand_pdb = PDBFile('ethanol.pdb')

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

modeller.addSolvent(system_generator.forcefield, model='tip4pew', padding=12.0 * unit.angstroms)

system = system_generator.forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
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

factory = alchemy.AbsoluteAlchemicalFactory(alchemical_pme_treatment='direct-space')

# Want to retain alchemical-alchemical nonbonded interactions as these are included in the decoupled endpoint solute + solvent system
alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=solute_indexes, annihilate_electrostatics=False, annihilate_sterics=False)
alchemical_system = factory.create_alchemical_system(system, alchemical_region)

# specify that we want to take energy derivatives on-the-fly with respect to both electrostatic and steric lambdas
for force in alchemical_system.getForces():
    if (force.__class__ == openmm.CustomNonbondedForce or force.__class__ == openmm.CustomBondForce):
        for i in range(0, force.getNumGlobalParameters()):
            if (force.getGlobalParameterName(i) == "lambda_electrostatics"):
                force.addEnergyParameterDerivative('lambda_electrostatics')
            elif (force.getGlobalParameterName(i) == "lambda_sterics"):
                force.addEnergyParameterDerivative('lambda_sterics')

alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)

'''
---FINISHED ALCHEMICAL CONFIGURATION---
'''

# Add a simple barostat for pressure control
alchemical_system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
# Use a simple thermostat for T control
integrator = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

context = Context(alchemical_system, integrator, platform)

context.setPositions(modeller.positions)

# below corresponds to fully interacting state
alchemical_state.lambda_electrostatics = 1.0
alchemical_state.lambda_sterics = 1.0

alchemical_state.apply_to_context(context)
    
dVe = []

# fix any bad contacts etc
LocalEnergyMinimizer.minimize(context, maxIterations=1000)

# lets equilibrate the system for 50 ps first
integrator.step(25000)

# write out the current state for visual inspection
state = context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

'''
---THERMODYNAMIC INTEGRATION---
    sample dV/dL using two paths:
        1) slowly turn electrostatics off
        2) then turn off sterics
        3) final dG estimate should then be the dG of 1) + 2)
'''

# electrostatics "should" be smoother than the LJ part, so "should" be able to get away with fewer lambda points
electrostatics_grid = np.linspace(1.0, 0.0, 10)
electrostatics_grid = [0.5]
print ("ELECTROSTATICS")
print ("lambda", "<dV/dl>")

for l in electrostatics_grid:
    
    alchemical_state.lambda_electrostatics = l
    alchemical_state.apply_to_context(context)
    
    # equilibrate for 100ps before collecting data for 500ps, taking dV/dl every 0.5ps
    integrator.step(50000)

    Es = []
    dV = []
    
    # data collection loop
    for iteration in range(1000):
        integrator.step(250)  
        
        state = context.getState(getEnergy=True, getParameterDerivatives=True)
        energy = state.getPotentialEnergy()
        energy_derivs = state.getEnergyParameterDerivatives()
        
        energy_deriv = energy_derivs['lambda_electrostatics']
        Es.append(energy._value)
        dV.append(energy_deriv)
    
    dVe.append(np.average(dV))
    
    print ("%5.2f %5.2f" % (l, np.average(dV)))

# use trapezoidal to integrate
dG_electrostatics = np.trapz(dVe, x=electrostatics_grid)
print ("dG electrostatics,", dG_electrostatics)
dVs = []

alchemical_state.lambda_electrostatics = 0.0
alchemical_state.apply_to_context(context)
    
sterics_grid = np.linspace(1.0, 0.0, 20)

print ("STERICS")
print ("lambda", "<dV/dl>")
for l in sterics_grid:
    
    alchemical_state.lambda_sterics = l
    alchemical_state.apply_to_context(context)
    
    integrator.step(50000)

    Es = []
    dV = []
    for iteration in range(1000):
        integrator.step(250) 
        
        state = context.getState(getEnergy=True, getParameterDerivatives=True)
        energy = state.getPotentialEnergy()
        energy_derivs = state.getEnergyParameterDerivatives()
        
        energy_deriv = energy_derivs['lambda_sterics']
        Es.append(energy._value)
        dV.append(energy_deriv)
    
    dVs.append(np.average(dV))
    
    print ("%5.2f %5.2f" % (l, np.average(dV)))
 
dG_sterics = np.trapz(dVs, x=sterics_grid)

print ("dG sterics,", dG_sterics)
print ("dG", dG_electrostatics + dG_sterics)
 
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
 
plt.plot(electrostatics_grid, dVe, label='electrostatics')
plt.plot(sterics_grid, dVs, label='sterics')
plt.legend()
 
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\left <\frac{\partial U}{\partial \lambda}\right >$')
 
plt.savefig('dvdl.png')
