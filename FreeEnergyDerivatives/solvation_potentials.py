from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from sys import stdout
from openmmtools.constants import ONE_4PI_EPS0
from openmmtools import forces


def create_alchemical_system_rxnfield(system, solute_indicies, cutoff=9.0 * unit.angstroms, switching_distance=1.0 * unit.angstroms):
    alchemical_atoms = set(solute_indicies)
    chemical_atoms = set(range(system.getNumParticles())) - alchemical_atoms
    
    for force_index, rforce in enumerate(system.getForces()):
        if (rforce.__class__.__name__ == "NonbondedForce"):
            reference_force = rforce
            remove_index = force_index
            break
        
    softcore_lj_function = '4.0*lambda_sterics^2*epsilon*x*(x-1.0); x = (1.0/reff_sterics);'
    softcore_lj_function += 'reff_sterics = (0.5*(1.0-lambda_sterics) + ((r/sigma)^6));'
    softcore_lj_function += 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)'
    softcore_lj = CustomNonbondedForce(softcore_lj_function)
    softcore_lj.addPerParticleParameter('sigma')
    softcore_lj.addPerParticleParameter('epsilon')
    softcore_lj.addGlobalParameter('lambda_sterics', 1.0) 
    softcore_lj.addEnergyParameterDerivative('lambda_sterics')
    
    epsilon_solvent = reference_force.getReactionFieldDielectric()
    r_cutoff = reference_force.getCutoffDistance()
    k_rf = r_cutoff ** (-3) * ((epsilon_solvent - 1) / (2 * epsilon_solvent + 1))
    k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
      
    softcore_electrostatics_function = 'ONE_4PI_EPS0*lambda_electrostatics^2*charge*(r^(-1) + k_rf*r^2);'
    softcore_electrostatics_function += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    softcore_electrostatics_function += 'k_rf = {k_rf};'.format(k_rf=k_rf)
    softcore_electrostatics_function += 'charge = charge1*charge2'
    softcore_electrostatics = CustomNonbondedForce(softcore_electrostatics_function)
    softcore_electrostatics.addPerParticleParameter('charge')
    softcore_electrostatics.addGlobalParameter('lambda_electrostatics', 1.0)
    softcore_electrostatics.addEnergyParameterDerivative('lambda_electrostatics')
    
    solute_electrostatics_function = 'ONE_4PI_EPS0*charge*(r^(-1) + k_rf*r^2);'
    solute_electrostatics_function += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    solute_electrostatics_function += 'k_rf = {k_rf};'.format(k_rf=k_rf)
    solute_electrostatics_function += 'charge = charge1*charge2'
    solute_electrostatics = CustomNonbondedForce(solute_electrostatics_function)
    solute_electrostatics.addPerParticleParameter('charge') 
    
    solute_lj_function = '4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;'
    solute_lj_function += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
    solute_lj = CustomNonbondedForce(solute_lj_function)
    solute_lj.addPerParticleParameter('sigma')
    solute_lj.addPerParticleParameter('epsilon')
    
    solvent_electrostatics_function = 'ONE_4PI_EPS0*charge*(r^(-1) + k_rf*r^2);'
    solvent_electrostatics_function += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    solvent_electrostatics_function += 'k_rf = {k_rf};'.format(k_rf=k_rf)
    solvent_electrostatics_function += 'charge = charge1*charge2'
    solvent_electrostatics = CustomNonbondedForce(solvent_electrostatics_function)
    solvent_electrostatics.addPerParticleParameter('charge') 
    
    solvent_lj_function = '4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;'
    solvent_lj_function += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
    solvent_lj = CustomNonbondedForce(solvent_lj_function)
    solvent_lj.addPerParticleParameter('sigma')
    solvent_lj.addPerParticleParameter('epsilon')
    
    softcore_lj_derivative_function = '4.0 * epsilon*((2.0*lambda_sterics*x*(x-1.0)) + lambda_sterics^2*(dxdl*(x-1.0) + x*dxdl));'
    softcore_lj_derivative_function += 'x = (1.0/reff_sterics);'
    softcore_lj_derivative_function += 'dxdl = -(1.0/reff_sterics^2) * 0.5;'
    softcore_lj_derivative_function += 'reff_sterics = (0.5*(1.0-lambda_sterics) + ((r/sigma)^6));'
    softcore_lj_derivative_function += 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)'
    softcore_lj_derivative = CustomNonbondedForce(softcore_lj_derivative_function)
    softcore_lj_derivative.addPerParticleParameter('sigma')
    softcore_lj_derivative.addPerParticleParameter('epsilon')
    softcore_lj_derivative.addGlobalParameter('lambda_sterics', 1.0) 
    
    softcore_electrostatics_derivative_function = 'ONE_4PI_EPS0*2.0*lambda_electrostatics*charge*(r^(-1) + k_rf*r^2);'
    softcore_electrostatics_derivative_function += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    softcore_electrostatics_derivative_function += 'k_rf = {k_rf};'.format(k_rf=k_rf)
    softcore_electrostatics_derivative_function += 'charge = charge1*charge2'
    softcore_electrostatics_derivative = CustomNonbondedForce(softcore_electrostatics_derivative_function)
    softcore_electrostatics_derivative.addPerParticleParameter('charge')
    softcore_electrostatics_derivative.addGlobalParameter('lambda_electrostatics', 1.0)
    
    for ind in range(system.getNumParticles()):
        # Get current parameters in non-bonded force
        [charge, sigma, epsilon] = reference_force.getParticleParameters(ind)
        # Make sure that sigma is not set to zero! Fine for some ways of writing LJ energy, but NOT OK for soft-core!
        if sigma / unit.nanometer == 0.0:
          newsigma = 0.3 * unit.nanometer  # This 0.3 is what's used by GROMACS as a default value for sc-sigma
        else:
          newsigma = sigma
        # Add the particle to the soft-core force (do for ALL particles)
        softcore_lj.addParticle([newsigma, epsilon])
        softcore_electrostatics.addParticle([charge])
        
        softcore_lj_derivative.addParticle([newsigma, epsilon])
        softcore_electrostatics_derivative.addParticle([charge])
        
        solute_lj.addParticle([newsigma, epsilon])
        solute_electrostatics.addParticle([charge])
        
        solvent_lj.addParticle([newsigma, epsilon])
        solvent_electrostatics.addParticle([charge])
        
        if ind in solute_indicies:
            reference_force.setParticleParameters(ind, charge * 0.0, sigma, epsilon * 0.0) 

    # Now we need to handle exceptions carefully
    for ind in range(reference_force.getNumExceptions()):
        [p1, p2, excCharge, excSig, excEps] = reference_force.getExceptionParameters(ind)
        # For consistency, must add exclusions where we have exceptions for custom forces
        softcore_lj.addExclusion(p1, p2)
        softcore_electrostatics.addExclusion(p1, p2)
        
        softcore_lj_derivative.addExclusion(p1, p2)
        softcore_electrostatics_derivative.addExclusion(p1, p2)
        
        solute_lj.addExclusion(p1, p2)
        solute_electrostatics.addExclusion(p1, p2)
        
        solvent_lj.addExclusion(p1, p2)
        solvent_electrostatics.addExclusion(p1, p2)
        
    softcore_lj.addInteractionGroup(alchemical_atoms, chemical_atoms)
    softcore_electrostatics.addInteractionGroup(alchemical_atoms, chemical_atoms)
    
    softcore_lj_derivative.addInteractionGroup(alchemical_atoms, chemical_atoms)
    softcore_electrostatics_derivative.addInteractionGroup(alchemical_atoms, chemical_atoms)
    
    solute_lj.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    solute_electrostatics.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    solvent_lj.addInteractionGroup(chemical_atoms, chemical_atoms)
    solvent_electrostatics.addInteractionGroup(chemical_atoms, chemical_atoms)
    
    softcore_lj.setCutoffDistance(cutoff)
    softcore_lj.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    softcore_lj.setUseLongRangeCorrection(False)
    softcore_lj.setUseSwitchingFunction(True)
    softcore_lj.setSwitchingDistance(switching_distance)
    softcore_lj.setForceGroup(0)
    
    system.addForce(softcore_lj)
    
    softcore_lj_derivative.setCutoffDistance(cutoff)
    softcore_lj_derivative.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    softcore_lj_derivative.setUseLongRangeCorrection(False)
    softcore_lj_derivative.setUseSwitchingFunction(True)
    softcore_lj_derivative.setSwitchingDistance(switching_distance)
    softcore_lj_derivative.setForceGroup(1)
    
    system.addForce(softcore_lj_derivative)
    
    softcore_electrostatics.setCutoffDistance(cutoff)
    softcore_electrostatics.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    softcore_electrostatics.setUseLongRangeCorrection(False) 
    softcore_electrostatics.setUseSwitchingFunction(True)
    softcore_electrostatics.setSwitchingDistance(switching_distance)
    softcore_electrostatics.setForceGroup(0)
    
    system.addForce(softcore_electrostatics)

    softcore_electrostatics_derivative.setCutoffDistance(cutoff)
    softcore_electrostatics_derivative.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    softcore_electrostatics_derivative.setUseLongRangeCorrection(False) 
    softcore_electrostatics_derivative.setUseSwitchingFunction(True)
    softcore_electrostatics_derivative.setSwitchingDistance(switching_distance)
    softcore_electrostatics_derivative.setForceGroup(1)
    
    system.addForce(softcore_electrostatics_derivative)
        
    solute_electrostatics.setCutoffDistance(cutoff)
    solute_electrostatics.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    solute_electrostatics.setUseLongRangeCorrection(False) 
    solute_electrostatics.setUseSwitchingFunction(True)
    solute_electrostatics.setSwitchingDistance(switching_distance)
    solute_electrostatics.setForceGroup(0)
    
    system.addForce(solute_electrostatics)
    
    solute_lj.setCutoffDistance(cutoff)
    solute_lj.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    solute_lj.setUseLongRangeCorrection(False) 
    solute_lj.setUseSwitchingFunction(True)
    solute_lj.setSwitchingDistance(switching_distance)
    solute_lj.setForceGroup(0)
    system.addForce(solute_lj)
    
    solvent_electrostatics.setCutoffDistance(cutoff)
    solvent_electrostatics.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    solvent_electrostatics.setUseLongRangeCorrection(False) 
    solvent_electrostatics.setUseSwitchingFunction(True)
    solvent_electrostatics.setSwitchingDistance(switching_distance)
    solvent_electrostatics.setForceGroup(0)
    system.addForce(solvent_electrostatics)
    
    solvent_lj.setCutoffDistance(cutoff)
    solvent_lj.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    solvent_lj.setUseLongRangeCorrection(False) 
    solvent_lj.setUseSwitchingFunction(True)
    solvent_lj.setSwitchingDistance(switching_distance)
    solvent_lj.setForceGroup(0)
    system.addForce(solvent_lj)
    
    system.removeForce(remove_index)
    
