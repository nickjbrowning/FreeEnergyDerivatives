from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from parmed.openmm.reporters import NetCDFReporter
import netCDF4 as nc

from openmmtools import alchemy
import numpy as np

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator

from lib import solvation_potentials as sp
from lib import thermodynamic_integration as TI
from lib import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sdf', type=str)
parser.add_argument('-pdb', type=str)
parser.add_argument('-solvate', type=int, default=1, choices=[0, 1])
parser.add_argument('-fit_forcefield', type=int, default=1, choices=[0, 1])
parser.add_argument('-nsamples', type=int, default=250)  
parser.add_argument('-nsample_steps', type=int, default=10000)  # 20ps using 2fs timestep
parser.add_argument('-solute_indexes', type=int, nargs='+', default=None)

parser.add_argument('-torsion_restraint_idx', type=int, nargs='+', default=None, help='(N,4) array of torsional restraint atom indexes')
parser.add_argument('-torsion_restraint_k', type=float, nargs='+', default=None, help='(N) array of torsional restraint k values (kJ/mol)')
parser.add_argument('-torsion_restraint_theta0', type=float, nargs='+', default=None, help='(N) array of torsional restraint theta0 values (rad)')

args = parser.parse_args()

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

'''
---SYSTEM PREPARATION---
    setup AM1-BCC charges for the solute, add solvent, set non-bonded method etc
'''

ligand_pdb = PDBFile(args.pdb)

modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

if (args.fit_forcefield):
    ligand_mol = Molecule.from_file(args.sdf, file_format='sdf')
    
    forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass': 4 * unit.amu }
    
    system_generator = SystemGenerator(
       forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'],
       small_molecule_forcefield='gaff-2.11',
       molecules=[ligand_mol],
       forcefield_kwargs=forcefield_kwargs)
    
    forcefield = system_generator.forcefield
else:
    forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml')

if (args.solute_indexes == None):
    solute_indexes = utils.collect_solute_indexes(modeller.topology)
else:
    solute_indexes = np.array(args.solute_indexes)

print ("Solute Indexes:", solute_indexes)

if (args.solvate):
    modeller.addSolvent(forcefield, model='tip3p', padding=12.0 * unit.angstroms)
    
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
        nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds)

if (args.torsion_restraint_idx is not None):
    # add torsional restraints
    print ("Applying Torsional Restraints")
    torsion_restraint_idx = np.array(args.torsion_restraint_idx).reshape((np.int(len(args.torsion_restraint_idx) / 4), 4))
    
    for i in range(torsion_restraint_idx.shape[0]):
        
        iw, ix, iy, iz = torsion_restraint_idx[i]
        
        k = args.torsion_restraint_k[i] * unit.kilojoule_per_mole
        theta0 = args.torsion_restraint_theta0[i] * unit.radian
        
        print ("Torsion ", i, iw, ix, iy, iz, "k", k, "theta0", theta0)
        
        force = openmm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
        force.addPerTorsionParameter("k");
        force.addPerTorsionParameter("theta0");
        
        force.addTorsion(int(iw), int(ix), int(iy), int(iz), [k, theta0])
        
        force.setForceGroup(0)
        
        system.addForce(force)
    
# system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
#         nonbondedCutoff=10.0 * unit.angstroms, constraints=HBonds, switch_distance=9.0 * unit.angstroms)
    
'''
---FINISHED SYSTEM PREPARATION---
'''

# Add a simple barostat for pressure control in periodic systems
if (args.solvate):
    system.addForce(MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin))
    
# Use a simple thermostat for T control
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
integrator.setConstraintTolerance(1.0E-08)

simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# fix any bad contacts etc
simulation.minimizeEnergy()

# lets equilibrate the system for 1ns first
print ("Equilibrating system for 1ns")
simulation.step(500000)
print ("Finished equilibrating system")

state = simulation.context.getState(getPositions=True)
PDBFile.writeFile(modeller.topology, state.getPositions(), file=open("equil.pdb", "w"))

simulation.reporters.append(StateDataReporter('data.txt', args.nsample_steps, step=True, potentialEnergy=True, temperature=True, density=True , volume=True))
simulation.reporters.append(NetCDFReporter('output.nc', args.nsample_steps))

simulation.step(args.nsamples * args.nsample_steps)

if (args.solvate):
    # lets strip out the solvent moecules in a new netcdf file
    utils.slice_netcdf("output.nc", "samples.nc", solute_indexes)
