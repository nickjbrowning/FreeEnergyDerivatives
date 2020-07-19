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
from openmmtools.testsystems import TestSystem, WaterBox

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')


def test_diatomic_system():
    
    class CustomSystem(TestSystem):
    
        def __init__(self, mass=39.9 * unit.amu, sigma=3.350 * unit.angstrom, epsilon=10.0 * unit.kilocalories_per_mole, **kwargs):
    
            TestSystem.__init__(self, **kwargs)
    
            # Store parameters
            self.mass = mass
            self.sigma = sigma
            self.epsilon = epsilon
    
            charge = 0.3 * unit.elementary_charge
    
            system = openmm.System()
            
            force = openmm.NonbondedForce()
            
            force.setForceGroup(0)
            force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            
            # Create positions.
            positions = unit.Quantity(np.zeros([3, 3], np.float32), unit.angstrom)
            
            positions[1, 0] = 4.5 * unit.angstrom
            positions[2, 0] = 2 * 2.2 ** (1.0 / 6.0) * sigma
            
            system.addParticle(mass)
            force.addParticle(charge, sigma, epsilon)
            
            system.addParticle(mass)
            force.addParticle(charge, sigma, epsilon)
           
            system.addParticle(mass)
            force.addParticle(charge, sigma, epsilon)

            system.addForce(force)
    
            self.system, self.positions = system, positions

            topology = app.Topology()
            element = app.Element.getBySymbol('Ar')
            chain = topology.addChain()
            
            for i in range(len(positions)):
                residue = topology.addResidue('Ar', chain)
                topology.addAtom('Ar', element, residue)
          
            self.topology = topology
            
    test = CustomSystem()

    system, positions, topology = test.system, test.positions, test.topology
    
    new_system = sp.create_alchemical_system(system, [0], compute_solvation_response=True, disable_alchemical_dispersion_correction=False)
    
    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    
    context = Context(new_system, integrator, platform)
    
    context.setParameter('lambda_sterics', 1.0)
    context.setParameter('lambda_electrostatics', 1.0)
    
    print (positions)
    print (np.linalg.norm(positions, axis=1))
    context.setPositions(positions)
    
    sp.decompose_energy(context, new_system)


def test_waterbox():
    waterbox = WaterBox()
    
    system, positions, topology = waterbox.system, waterbox.positions, waterbox.topology
    
    system = sp.create_alchemical_system(system, [0, 1, 2], compute_solvation_response=True, disable_alchemical_dispersion_correction=False)
    
    print (system.getNumForces())
    
    for force in system.getForces():
        print (force.__class__.__name__, force.getForceGroup())
        
    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    integrator.setIntegrationForceGroups({0, 1, 2, 3, 4, 5, 6, 7, 8})
    
    context = Context(system, integrator, platform)
    context.setPositions(positions)
    
    context.setParameter('lambda_electrostatics', 0.8)
    context.setParameter('lambda_sterics', 0.8)
    
    sp.decompose_energy(context, system)

    
def test_solvated_ethanol():
    
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

    ligand_mol = Molecule.from_file('ethanol.sdf', file_format='sdf')
    
    forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass': 4 * unit.amu }
    
    system_generator = SystemGenerator(
        forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=[ligand_mol],
        forcefield_kwargs=forcefield_kwargs)

    ligand_pdb = PDBFile('ethanol.pdb')
    
    modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)
    
    modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=12.0 * unit.angstroms)
    
    system = system_generator.forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffPeriodic,
            nonbondedCutoff=9.0 * unit.angstroms, constraints=HBonds, switchDistance=7.5 * unit.angstroms)
    
    solute_indexes = collect_solute_indexes(modeller.topology)

    system = sp.create_alchemical_system(system, solute_indexes, compute_solvation_response=True, disable_alchemical_dispersion_correction=False)
    
    for force in system.getForces():
        print (force.__class__.__name__, force.getForceGroup())
        
    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    integrator.setIntegrationForceGroups({0, 1, 2, 3, 4, 5, 6, 7, 8})
    
    context = Context(system, integrator, platform)
    context.setPositions(positions)
    
    context.setParameter('lambda_electrostatics', 0.8)
    context.setParameter('lambda_sterics', 0.8)
    
    sp.decompose_energy(context, system)


if __name__ == "__main__":
    print ("Diatomic System")
    test_diatomic_system()
    print ("Waterbox")
    test_waterbox()
    print ("Solvated Ethanol Test")
    test_solvated_ethanol()
    # print ("finite diff test")
    # finite_diff_test()
