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
from openmmtools.testsystems import TestSystem


class CustomSystem(TestSystem):

    def __init__(self, mass=39.9 * unit.amu, sigma=3.350 * unit.angstrom, epsilon=10.0 * unit.kilocalories_per_mole, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Store parameters
        self.mass = mass
        self.sigma = sigma
        self.epsilon = epsilon

        charge = 0.3 * unit.elementary_charge

        system = openmm.System()

        def _get_sterics_expression():
            exceptions_sterics_energy_expression = '4.0*lambda_sterics*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;'
            exceptions_sterics_energy_expression += 'reff_sterics = (softcore_alpha*sigma^softcore_n *(1.0-lambda_sterics^softcore_a) + r^softcore_n)^(1/softcore_n);'
            
            sterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
            
            return sterics_mixing_rules, exceptions_sterics_energy_expression
        
        def _get_sterics_expression_derivative():
            exceptions_sterics_energy_expression = '4.0*epsilon*x*(x-1.0) + lambda_sterics*4*epsilon*(dxdl*(x-1.0) + x*dxdl); x = (sigma/reff_sterics)^6;'
            exceptions_sterics_energy_expression += 'dxdl = -6*(sigma^6/reff_sterics^7) * drdl;'
            exceptions_sterics_energy_expression += 'drdl = -softcore_a*lambda_sterics^(softcore_a-1)*softcore_alpha*sigma^softcore_n * (1/softcore_n)*(softcore_alpha*sigma^softcore_n*(1.0-lambda_sterics^softcore_a)+r^softcore_n)^((1/softcore_n) - 1.0);'
            
            exceptions_sterics_energy_expression += 'reff_sterics = (softcore_alpha*sigma^softcore_n *(1.0-lambda_sterics^softcore_a) + r^softcore_n)^(1/softcore_n);'
            
            sterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
            
            return sterics_mixing_rules, exceptions_sterics_energy_expression
        
        mixing, expression = _get_sterics_expression()
        
        force = openmm.CustomNonbondedForce(expression + mixing)
        
        force.addGlobalParameter('softcore_alpha', 0.4)
        force.addGlobalParameter('softcore_beta', 0.0)
        force.addGlobalParameter('softcore_a', 2.0)
        force.addGlobalParameter('softcore_b', 2.0)
        force.addGlobalParameter('softcore_m', 1.0)
        force.addGlobalParameter('softcore_n', 6.0)
        force.addGlobalParameter('lambda_sterics', 1.0)
        
        force.addPerParticleParameter('sigma')
        force.addPerParticleParameter('epsilon')
        
        force.addEnergyParameterDerivative('lambda_sterics') 
        
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        
        force.setForceGroup(0)
        
        mixing, expression = _get_sterics_expression_derivative()
        
        force2 = openmm.CustomNonbondedForce(expression + mixing)
        
        force2.addGlobalParameter('softcore_alpha', 0.4)
        force2.addGlobalParameter('softcore_beta', 0.0)
        force2.addGlobalParameter('softcore_a', 2.0)
        force2.addGlobalParameter('softcore_b', 2.0)
        force2.addGlobalParameter('softcore_m', 1.0)
        force2.addGlobalParameter('softcore_n', 6.0)
        force2.addGlobalParameter('lambda_sterics', 1.0)
        force2.addEnergyParameterDerivative('lambda_sterics') 
        
        force2.addPerParticleParameter('sigma')
        force2.addPerParticleParameter('epsilon')
        
        force2.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        
        force2.setForceGroup(1)
        
        # Create positions.
        positions = unit.Quantity(np.zeros([3, 3], np.float32), unit.angstrom)
        # Move the second particle along the x axis to be at the potential minimum.
        positions[1, 0] = 2.0 ** (1.0 / 6.0) * sigma
        positions[2, 0] = 2 * 2.0 ** (1.0 / 6.0) * sigma
        
        system.addParticle(mass)
        force.addParticle([ sigma, epsilon])
        force2.addParticle([ sigma, epsilon])
        
        system.addParticle(mass)
        force.addParticle([ sigma, epsilon])
        force2.addParticle([ sigma, epsilon])
        
        system.addParticle(mass)
        force.addParticle([ sigma, epsilon])
        force2.addParticle([ sigma, epsilon])
        
        system.addForce(force)
        system.addForce(force2)

        self.system, self.positions = system, positions

        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)

        self.topology = topology


test = CustomSystem()

system, positions, topology = test.system, test.positions, test.topology

integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

context = Context(system, integrator)

for distance in np.linspace(3.5, 5.0, 10):
    positions[1, 1] = distance * unit.angstroms
    
    context.setPositions(positions)
    
    state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))

    energy_derivs = state.getEnergyParameterDerivatives()
    
    print ("P.E :", state.getPotentialEnergy(), "dVdl", energy_derivs['lambda_sterics'])

    state = context.getState(getEnergy=True, groups=set([1]))
    
    print ("dV/dl :", state.getPotentialEnergy())
    
