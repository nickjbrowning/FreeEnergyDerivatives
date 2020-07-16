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
            
            print (reference_force.getUseDispersionCorrection())
            
            # Create positions.
            positions = unit.Quantity(np.zeros([3, 3], np.float32), unit.angstrom)
            # Move the second particle along the x axis to be at the potential minimum.
            positions[1, 0] = 3.0 ** (1.0 / 6.0) * sigma
            positions[2, 0] = 6.0 ** (1.0 / 6.0) * sigma
            
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
    
    system = sp.create_alchemical_system(system, [1], softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=True)
    
    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    
    context = Context(system, integrator, platform)
    context.setParameter('lambda_sterics', 0.0)
    context.setParameter('lambda_electrostatics', 0.0)
    
    for distance in np.linspace(3.5, 5.0, 10):
        positions[1, 1] = distance * unit.angstroms
        
        context.setPositions(positions)
        
        state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))
    
        energy_derivs = state.getEnergyParameterDerivatives()
        
        print ("P.E :", state.getPotentialEnergy())
        
        state = context.getState(getEnergy=True, groups=set([1]))
        print ("electrostatic dVdl", energy_derivs['lambda_electrostatics'], state.getPotentialEnergy(), "Diff: ", energy_derivs['lambda_electrostatics'] - state.getPotentialEnergy()._value)
        
        state = context.getState(getEnergy=True, groups=set([2]))
        print ("steric dV/dl :", energy_derivs['lambda_sterics'], state.getPotentialEnergy(), "Diff: ", energy_derivs['lambda_sterics'] - state.getPotentialEnergy()._value)
    

def test_waterbox():
    waterbox = WaterBox()
    
    system, positions, topology = waterbox.system, waterbox.positions, waterbox.topology
    
    system = sp.create_alchemical_system2(system, [0, 1, 2], softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=True)
    
    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    integrator.setIntegrationForceGroups(set([0]))
    
    context = Context(system, integrator, platform)
    
    print ("ELECTROSTATICS")
    for l in np.linspace(1.0, 0.0, 10):
        context.setParameter('lambda_electrostatics', l)
        
        context.setPositions(positions)
    
        state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))
    
        energy_derivs = state.getEnergyParameterDerivatives()
        dvdle = energy_derivs['lambda_electrostatics']
        
        deriv_state = context.getState(getEnergy=True, groups=set([1]))
        deriv_electrostatic = deriv_state.getPotentialEnergy()._value
        
        print ("lambda: ", context.getParameter('lambda_electrostatics'))
        print ("electrostatic dV/dl :", dvdle, deriv_electrostatic, "Diff: ", dvdle - deriv_electrostatic)
        
    print ("STERICS")
    for l in np.linspace(1.0, 0.0, 10):
        
        context.setParameter('lambda_sterics', l)
        
        context.setPositions(positions)
    
        state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))
    
        energy_derivs = state.getEnergyParameterDerivatives()
        dvdls = energy_derivs['lambda_sterics']
        
        deriv_state = context.getState(getEnergy=True, groups=set([2]))
        deriv_steric = deriv_state.getPotentialEnergy()._value
        
        print ("lambda: ", context.getParameter('lambda_sterics'))
        print("steric dV/dl :", dvdls, deriv_steric, "Diff: ", dvdls - deriv_steric)


def finite_diff_test():
    
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
            # Move the second particle along the x axis to be at the potential minimum.
            positions[1, 0] = 3.0 ** (1.0 / 6.0) * sigma
            positions[2, 0] = 6.0 ** (1.0 / 6.0) * sigma
            
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
    
    system = sp.create_alchemical_system(system, [1], softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=True)
    
    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    
    context = Context(system, integrator, platform)
    
    context.setParameter('lambda_sterics', 0.5)
    context.setParameter('lambda_electrostatics', 0.5)
 
    context.setPositions(positions)
    
    state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))

    energy_derivs = state.getEnergyParameterDerivatives()
    
    print ("P.E :", state.getPotentialEnergy())
    
    state = context.getState(getEnergy=True, groups=2 ** 1)
    
    print ("electrostatic dVdl", energy_derivs['lambda_electrostatics'])
    
    state = context.getState(getEnergy=True, groups=2 ** 2)
    
    print ("steric dV/dl :", energy_derivs['lambda_sterics'])
    
    context.setParameter('lambda_sterics', 0.5 + 0.5 * 0.005)
    
    state1 = context.getState(getEnergy=True, groups=[2 ** 0])
    
    context.setParameter('lambda_sterics', 0.5 - 0.5 * 0.005)
    
    state2 = context.getState(getEnergy=True, groups=[2 ** 0])
    
    print (state1.getPotentialEnergy(), state2.getPotentialEnergy())
    print ((state1.getPotentialEnergy() - state2.getPotentialEnergy()))
      

if __name__ == "__main__":
    print ("Diatomic System")
    test_diatomic_system()
    # print ("Waterbox")
    # test_waterbox()
    # print ("finite diff test")
    # finite_diff_test()
