# This script just confirms addEnergyParameterDerivative works as intended for a simple diatomic LJ system.

from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from simtk.openmm import *
from openmmtools.testsystems import  LennardJonesPair
from openmmtools import alchemy
import numpy as np
from simtk.openmm.vec3 import Vec3
from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator

platform = openmm.Platform.getPlatformByName('CPU')
# platform.setPropertyDefaultValue('Precision', 'double')

'''
---SYSTEM PREPARATION---
'''

test_system = LennardJonesPair()
[system, topology, positions] = [test_system.system, test_system.topology , test_system.positions]

reference_force = system.getForces()[0]

sterics_mixing_rules = ()

expression = ('U_sterics;'
              'U_sterics = (1.0-lambda)*4*epsilon*x*(x-1.0);'
              'x = (sigma/reff_sterics)^6;'
              'reff_sterics = (softcore_alpha*lambda*(sigma^softcore_n) + r^softcore_n)^(1.0/softcore_n);'
              'epsilon = sqrt(epsilon1*epsilon2);'
              'sigma = 0.5*(sigma1 + sigma2);')

force = openmm.CustomNonbondedForce(expression)

force.addGlobalParameter('lambda', 1.0)
force.addGlobalParameter('softcore_alpha', 0.5)
force.addGlobalParameter('softcore_n', 6.0)

force.addEnergyParameterDerivative('lambda')

force.addPerParticleParameter("sigma")
force.addPerParticleParameter("epsilon") 

force.setUseSwitchingFunction(reference_force.getUseSwitchingFunction())
force.setCutoffDistance(reference_force.getCutoffDistance())
force.setSwitchingDistance(reference_force.getSwitchingDistance())
force.setUseLongRangeCorrection(reference_force.getUseDispersionCorrection())
force.setNonbondedMethod(reference_force.getNonbondedMethod())
            
for particle_index in range(reference_force.getNumParticles()):
    [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
    force.addParticle([sigma, epsilon])
    print ('Adding Particle:', charge, sigma, epsilon)
  
sigmas = []
epsilons = []

for particle_index in range(reference_force.getNumParticles()):
    [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
    reference_force.setParticleParameters(particle_index, abs(0.0 * charge), sigma, abs(0 * epsilon))
    sigmas.append(sigma)
    epsilons.append(epsilon)

force.addInteractionGroup([0], [1])
          
# remove existing force
system.removeForce(0)

system.addForce(force)

integrator = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

context = Context(system, integrator, platform)
context.setPositions(positions)

integrator.step(100)

print ("lambda", "<dV/dl>")

distance = np.linspace(1.0, 3.0, 20)

context.setParameter('lambda', 0.5)


def compute_sterics_dvdl(sigma_1, sigma_2, epsilon_1, epsilon_2, r, l=0.5):
                            
    sigma = 0.5 * (sigma_1 + sigma_2)
    epsilon = np.sqrt(epsilon_1 * epsilon_2)
    
    reff = (0.5 * l * sigma ** 6 + r ** 6) ** (1 / 6)
    
    x = (sigma / reff) ** 6
    
    dreffdl = 1 / 6 * (0.5 * l * sigma ** 6 + r ** 6) ** (-5 / 6) * 0.5 * sigma ** 6
    
    dxdl = -6 * (sigma / reff) ** 5 * sigma / (reff ** 2) * dreffdl
    
    dudl = -4 * epsilon * x * (x - 1) + (1 - l) * 4 * epsilon * (dxdl * (x - 1) + x * dxdl)
    
    return dudl

    
for r in distance:
    
    context.setPositions([Vec3(x=0.0, y=0.0, z=0.0), Vec3(x=r, y=0.0, z=0.0)])

    state = context.getState(getEnergy=True, getParameterDerivatives=True)
    energy = state.getPotentialEnergy()
    energy_derivs = state.getEnergyParameterDerivatives()
        
    energy_deriv = energy_derivs['lambda']
    
    print (energy_deriv, compute_sterics_dvdl(sigmas[0]._value, sigmas[1]._value, epsilons[0]._value, epsilons[1]._value, r))
