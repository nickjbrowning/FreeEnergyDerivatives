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

        def _get_electrostatics_expression(k_rf, c_rf):
            
            dexceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*lambda_electrostatics*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);'
            dexceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
            dexceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
            dexceptions_electrostatics_energy_expression += 'reff_electrostatics=(softcore_beta*(1.0-lambda_electrostatics^softcore_b) + r^softcore_m)^(1/softcore_m);'
                
            delectrostatics_mixing_rules = 'chargeprod = charge1*charge2;'
        
            return delectrostatics_mixing_rules, dexceptions_electrostatics_energy_expression
        
        def _get_electrostatics_expression_derivative(k_rf, c_rf):
           
            drdl = 'drdl = -softcore_b*lambda_electrostatics^(softcore_b - 1.0)*softcore_beta*(1/softcore_m)*(softcore_beta*(1.0-lambda_electrostatics^softcore_b) +r^softcore_m)^((1/softcore_m) - 1.0);'
            
            dexceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf)'
            dexceptions_electrostatics_energy_expression += '+ ONE_4PI_EPS0*lambda_electrostatics*chargeprod*(-1.0*reff_electrostatics^(-2.0)*drdl + 2*k_rf*reff_electrostatics*drdl);'
            dexceptions_electrostatics_energy_expression += drdl
            dexceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
            dexceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
            
            dexceptions_electrostatics_energy_expression += 'reff_electrostatics=(softcore_beta*(1.0-lambda_electrostatics^softcore_b) + r^softcore_m)^(1/softcore_m);'
                
            delectrostatics_mixing_rules = 'chargeprod = charge1*charge2;'
        
            return delectrostatics_mixing_rules, dexceptions_electrostatics_energy_expression

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
        
        sterics = openmm.CustomNonbondedForce(expression + mixing)
        
        sterics.addGlobalParameter('softcore_alpha', 0.4)
        sterics.addGlobalParameter('softcore_beta', 0.0)
        sterics.addGlobalParameter('softcore_a', 2.0)
        sterics.addGlobalParameter('softcore_b', 2.0)
        sterics.addGlobalParameter('softcore_m', 1.0)
        sterics.addGlobalParameter('softcore_n', 6.0)
        sterics.addGlobalParameter('lambda_sterics', 1.0)
        
        sterics.addPerParticleParameter('sigma')
        sterics.addPerParticleParameter('epsilon')
        
        sterics.addEnergyParameterDerivative('lambda_sterics') 
        
        sterics.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        
        sterics.setForceGroup(0)
        
        mixing, expression = _get_sterics_expression_derivative()
        
        sterics_deriv = openmm.CustomNonbondedForce(expression + mixing)
        
        sterics_deriv.addGlobalParameter('softcore_alpha', 0.4)
        sterics_deriv.addGlobalParameter('softcore_beta', 0.0)
        sterics_deriv.addGlobalParameter('softcore_a', 2.0)
        sterics_deriv.addGlobalParameter('softcore_b', 2.0)
        sterics_deriv.addGlobalParameter('softcore_m', 1.0)
        sterics_deriv.addGlobalParameter('softcore_n', 6.0)
        sterics_deriv.addGlobalParameter('lambda_sterics', 1.0)
        
        sterics_deriv.addPerParticleParameter('sigma')
        sterics_deriv.addPerParticleParameter('epsilon')
        
        sterics_deriv.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        
        sterics_deriv.setForceGroup(1)
        
        mixing, expression = _get_electrostatics_expression(0.5, 1.5)
        
        electrostatics = openmm.CustomNonbondedForce(expression + mixing)
        
        electrostatics.addGlobalParameter('softcore_alpha', 0.4)
        electrostatics.addGlobalParameter('softcore_beta', 0.0)
        electrostatics.addGlobalParameter('softcore_a', 2.0)
        electrostatics.addGlobalParameter('softcore_b', 2.0)
        electrostatics.addGlobalParameter('softcore_m', 1.0)
        electrostatics.addGlobalParameter('softcore_n', 6.0)
        electrostatics.addGlobalParameter('lambda_electrostatics', 1.0)
        
        electrostatics.addPerParticleParameter('charge')
        
        electrostatics.addEnergyParameterDerivative('lambda_electrostatics') 
        
        electrostatics.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        
        electrostatics.setForceGroup(0)
        
        mixing, expression = _get_electrostatics_expression_derivative(0.5, 1.5)
        
        electrostatics_deriv = openmm.CustomNonbondedForce(expression + mixing)
        
        electrostatics_deriv.addGlobalParameter('softcore_alpha', 0.4)
        electrostatics_deriv.addGlobalParameter('softcore_beta', 0.0)
        electrostatics_deriv.addGlobalParameter('softcore_a', 2.0)
        electrostatics_deriv.addGlobalParameter('softcore_b', 2.0)
        electrostatics_deriv.addGlobalParameter('softcore_m', 1.0)
        electrostatics_deriv.addGlobalParameter('softcore_n', 6.0)
        electrostatics_deriv.addGlobalParameter('lambda_electrostatics', 1.0)
        
        electrostatics_deriv.addPerParticleParameter('charge')
        
        electrostatics_deriv.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        
        electrostatics_deriv.setForceGroup(2)
        
        # Create positions.
        positions = unit.Quantity(np.zeros([3, 3], np.float32), unit.angstrom)
        # Move the second particle along the x axis to be at the potential minimum.
        positions[1, 0] = 2.0 ** (1.0 / 6.0) * sigma
        positions[2, 0] = 2 * 2.0 ** (1.0 / 6.0) * sigma
        
        system.addParticle(mass)
        sterics.addParticle([ sigma, epsilon])
        sterics_deriv.addParticle([ sigma, epsilon])
        electrostatics.addParticle([charge])
        electrostatics_deriv.addParticle([charge])
        
        system.addParticle(mass)
        sterics.addParticle([ sigma, epsilon])
        sterics_deriv.addParticle([ sigma, epsilon])
        electrostatics.addParticle([charge])
        electrostatics_deriv.addParticle([charge])
        
        system.addParticle(mass)
        sterics.addParticle([ sigma, epsilon])
        sterics_deriv.addParticle([ sigma, epsilon])
        electrostatics.addParticle([charge])
        electrostatics_deriv.addParticle([charge])
        
        system.addForce(sterics)
        system.addForce(sterics_deriv)
        system.addForce(electrostatics)
        system.addForce(electrostatics_deriv)
        
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
    
    print (energy_derivs)

    state = context.getState(getEnergy=True, groups=set([1]))
    
    dvdl_sterics = state.getPotentialEnergy()._value
    
    state = context.getState(getEnergy=True, groups=set([2]))
    
    dvdl_electrostatics = state.getPotentialEnergy()._value
    
    print ("dV/dl_electrostatics:", state.getPotentialEnergy())
    
    print ("P.E :", state.getPotentialEnergy(), "dVdl_sterics diff", energy_derivs['lambda_sterics'] - dvdl_sterics, "dVdl_electrostatics diff", energy_derivs['lambda_electrostatics'] - dvdl_electrostatics)
