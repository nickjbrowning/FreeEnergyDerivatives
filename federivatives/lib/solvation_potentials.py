from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from openmmtools.constants import ONE_4PI_EPS0
from openmmtools import forces
import numpy as np
import copy


def _get_sterics_expression():
    exceptions_sterics_energy_expression = '4.0*lambda_sterics*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;'
    exceptions_sterics_energy_expression += 'reff_sterics = (softcore_alpha*sigma^softcore_n *(1.0-lambda_sterics^softcore_a) + r^softcore_n)^(1/softcore_n);'
    
    sterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
    
    return sterics_mixing_rules, exceptions_sterics_energy_expression


def _get_sterics_expression_derivative():
    dexceptions_sterics_energy_expression = '4.0*epsilon*x*(x-1.0) + lambda_sterics*4*epsilon*(dxdl*(x-1.0) + x*dxdl); x = (sigma/reff_sterics)^6;'
    dexceptions_sterics_energy_expression += 'dxdl = -6*(sigma^6/reff_sterics^7) * drdl;'
    dexceptions_sterics_energy_expression += 'drdl = -softcore_a*lambda_sterics^(softcore_a-1)*softcore_alpha*sigma^softcore_n * (1/softcore_n)*(softcore_alpha*sigma^softcore_n*(1.0-lambda_sterics^softcore_a)+r^softcore_n)^((1/softcore_n) - 1.0);'
    
    dexceptions_sterics_energy_expression += 'reff_sterics = (softcore_alpha*sigma^softcore_n *(1.0-lambda_sterics^softcore_a) + r^softcore_n)^(1/softcore_n);'
    
    dsterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
    
    return dsterics_mixing_rules, dexceptions_sterics_energy_expression


def _get_electrostatics_expression(reference_force):
    
    epsilon_solvent = reference_force.getReactionFieldDielectric()
    rcut = reference_force.getCutoffDistance()

    k_rf = rcut ** (-3) * ((epsilon_solvent - 1) / (2 * epsilon_solvent + 1))
    k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
     
    c_rf = rcut ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + 1))
    c_rf = c_rf.value_in_unit_system(unit.md_unit_system)

    exceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*lambda_electrostatics*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);'
    exceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    exceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
    exceptions_electrostatics_energy_expression += 'reff_electrostatics=(softcore_beta*(1.0-lambda_electrostatics^softcore_b) + r^softcore_m)^(1/softcore_m);'
        
    electrostatics_mixing_rules = 'chargeprod = charge1*charge2;'

    return electrostatics_mixing_rules, exceptions_electrostatics_energy_expression


def _get_electrostatics_expression_derivative(reference_force):
    epsilon_solvent = reference_force.getReactionFieldDielectric()
    rcut = reference_force.getCutoffDistance()
    
    k_rf = rcut ** (-3) * ((epsilon_solvent - 1) / (2 * epsilon_solvent + 1))
    k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
     
    c_rf = rcut ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + 1))
    c_rf = c_rf.value_in_unit_system(unit.md_unit_system)
    
    drdl = 'drdl = -softcore_b*lambda_electrostatics^(softcore_b - 1.0)*softcore_beta*(1/softcore_m)*(softcore_beta*(1.0-lambda_electrostatics^softcore_b) +r^softcore_m)^((1/softcore_m) - 1.0);'
    
    dexceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf)'
    dexceptions_electrostatics_energy_expression += '+ ONE_4PI_EPS0*lambda_electrostatics*chargeprod*(-1.0*reff_electrostatics^(-2.0)*drdl + 2*k_rf*reff_electrostatics*drdl);'
    dexceptions_electrostatics_energy_expression += drdl
    dexceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    dexceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
    
    dexceptions_electrostatics_energy_expression += 'reff_electrostatics=(softcore_beta*(1.0-lambda_electrostatics^softcore_b) + r^softcore_m)^(1/softcore_m);'
        
    delectrostatics_mixing_rules = 'chargeprod = charge1*charge2;'

    return delectrostatics_mixing_rules, dexceptions_electrostatics_energy_expression


def create_force(force_constructor, energy_expression, is_lambda_controlled=False, lambda_var=None, request_derivative=True):
   
    if (is_lambda_controlled):
        force = force_constructor(energy_expression)
        force.addGlobalParameter(lambda_var, 1.0)
        if (request_derivative):
            print ("Requesting derivative on: ", force_constructor, lambda_var)
            force.addEnergyParameterDerivative(lambda_var)  # also request that we compute dE/dlambda
    else:
        energy_expression = energy_expression + lambda_var + '=1.0;'
        force = force_constructor(energy_expression)
    return force


def create_alchemical_system(system, solute_indicies, compute_solvation_response=False,
                                      annihilate_sterics=False, annihilate_electrostatics=False,
                                      disable_alchemical_dispersion_correction=False, softcore_alpha=0.5, softcore_beta=0.0, softcore_m=1.0, softcore_n=6.0, softcore_a=2.0, softcore_b=2.0):
    
    new_system = copy.deepcopy(system)
    
    alchemical_atoms = set(solute_indicies)
    
    chemical_atoms = set(range(system.getNumParticles())).difference(alchemical_atoms)
    
    integration_group = []
    
    for force in new_system.getForces():
        # group 0 will be used as integration group, so move all existing forces here
        force.setForceGroup(0)
        
    integration_group.append(0)
    
    force_idx, reference_force = forces.find_forces(new_system, openmm.NonbondedForce, only_one=True)

    nonbonded_force = copy.deepcopy(reference_force)
    
    sterics_mixing_roles, exceptions_sterics_energy_expression = _get_sterics_expression()
    
    sterics_energy_expression = exceptions_sterics_energy_expression + sterics_mixing_roles
    
    electrostatics_mixing_rules, exceptions_electrostatics_energy_expression = _get_electrostatics_expression(reference_force)
    
    electrostatics_energy_expression = exceptions_electrostatics_energy_expression + electrostatics_mixing_rules
    
    na_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                            True, 'lambda_electrostatics')
    
    aa_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                            annihilate_electrostatics, 'lambda_electrostatics', request_derivative=annihilate_electrostatics)
    
    all_electrostatics_custom_nonbonded_forces = [na_electrostatics_custom_nonbonded_force, aa_electrostatics_custom_nonbonded_force]
    
    na_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                       True, 'lambda_electrostatics')
    aa_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                       annihilate_electrostatics, 'lambda_electrostatics', request_derivative=annihilate_electrostatics)
    
    all_electrostatics_custom_bond_forces = [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]
    
    na_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                    True, 'lambda_sterics')
    aa_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                    annihilate_sterics, 'lambda_sterics', request_derivative=annihilate_sterics)
    
    all_sterics_custom_nonbonded_forces = [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]
    
    # CustomBondForces represent exceptions not picked up by exclusions 
    na_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                True, 'lambda_sterics')
    aa_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                annihilate_sterics, 'lambda_sterics', request_derivative=annihilate_sterics)
    
    all_sterics_custom_bond_forces = [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]

    for force in all_sterics_custom_nonbonded_forces:
        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon") 
        force.setUseSwitchingFunction(reference_force.getUseSwitchingFunction())
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setSwitchingDistance(reference_force.getSwitchingDistance())
        
        if disable_alchemical_dispersion_correction:
            force.setUseLongRangeCorrection(False)
        else:
            force.setUseLongRangeCorrection(reference_force.getUseDispersionCorrection())
    
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        
    for force in all_electrostatics_custom_nonbonded_forces:
        force.addPerParticleParameter("charge")
        force.setUseSwitchingFunction(False)
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setUseLongRangeCorrection(False)  
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    
    for force in all_sterics_custom_bond_forces:
        force.addPerBondParameter("sigma")  
        force.addPerBondParameter("epsilon")
    
    for force in all_electrostatics_custom_bond_forces:
        force.addPerBondParameter("chargeprod")  # charge product
        # force.addPerBondParameter("sigma") 
    
    # fix any missing values that can screw things up
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
        
        if sigma == 0.0 * unit.angstrom:
            warning_msg = 'particle %d has Lennard-Jones sigma = 0 (charge=%s, sigma=%s, epsilon=%s); setting sigma=1A'
            logger.warning(warning_msg % (particle_index, str(charge), str(sigma), str(epsilon)))
            sigma = 3.0 * unit.angstrom
            nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
            
    # also do the same for exceptions
    for exception_index in range(reference_force.getNumExceptions()):
  
        [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)

        if sigma == 0.0 * unit.angstrom:
            warning_msg = 'exception %d has Lennard-Jones sigma = 0 (iatom=%d, jatom=%d, chargeprod=%s, sigma=%s, epsilon=%s); setting sigma=1A'
            logger.warning(warning_msg % (exception_index, iatom, jatom, str(chargeprod), str(sigma), str(epsilon)))
            sigma = 3.0 * unit.angstrom
            # Fix it.
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)
    
    # add all particles to all custom forces...
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
        
        for force in all_sterics_custom_nonbonded_forces:
            force.addParticle([sigma, epsilon])
      
        for force in all_electrostatics_custom_nonbonded_forces:
            force.addParticle([charge])

    # now turn off interactions from alchemically-modified particles in unmodified nonbonded force
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)

        if particle_index in solute_indicies:
            nonbonded_force.setParticleParameters(particle_index, 0.0, sigma, 0.0)
            
    # Now restrict pairwise interactions to their respective groups
    na_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    na_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    # now lets handle exclusions and exceptions
    all_custom_nonbonded_forces = all_electrostatics_custom_nonbonded_forces + all_sterics_custom_nonbonded_forces 
    
    for exception_index in range(reference_force.getNumExceptions()):
    
        iatom, jatom, chargeprod, sigma, epsilon = reference_force.getExceptionParameters(exception_index)
    
        # All non-bonded forces must have same number of exceptions/exclusions on CUDA
        for force in all_custom_nonbonded_forces:
            force.addExclusion(iatom, jatom)

        is_exception_epsilon = abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 0.0
        is_exception_chargeprod = abs(chargeprod.value_in_unit_system(unit.md_unit_system)) > 0.0
        
        both_alchemical = iatom in alchemical_atoms and jatom in alchemical_atoms
        at_least_one_alchemical = iatom in alchemical_atoms or jatom in alchemical_atoms
        only_one_alchemical = at_least_one_alchemical and not both_alchemical
        
        # If exception (and not exclusion), add special CustomBondForce terms to handle alchemically modified LJ and reactionfield electrostatics
        if both_alchemical:
            if is_exception_epsilon:
                aa_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
            if is_exception_chargeprod:
                aa_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])

        # When this is a single region we model the exception between alchemical
        # and non-alchemical particles using a single custom bond.
        
        elif only_one_alchemical:
            if is_exception_epsilon:
                na_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
            if is_exception_chargeprod:
                na_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])
        # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce
        
        # remove this exception in original reference force
        if at_least_one_alchemical:
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, 0.0, sigma, 0.0)
    
    all_custom_forces = (all_electrostatics_custom_nonbonded_forces + all_electrostatics_custom_bond_forces + all_sterics_custom_nonbonded_forces + all_sterics_custom_bond_forces)
    
    names = ["na_electrostatics_nbf", "aa_electrostatics_nbf", "na_electrostatics_bf", "aa_electrostatics_bf",
             "na_sterics_nbf", "aa_sterics_nbf", "na_sterics_bf", "aa_sterics_bf" ]

    def add_global_parameters(force):
        force.addGlobalParameter('softcore_alpha', softcore_alpha)
        force.addGlobalParameter('softcore_beta', softcore_beta)
        force.addGlobalParameter('softcore_a', softcore_a)
        force.addGlobalParameter('softcore_b', softcore_b)
        force.addGlobalParameter('softcore_m', softcore_m)
        force.addGlobalParameter('softcore_n', softcore_n)

    # add all forces representing alchemical interactions
    for i, force in enumerate(all_custom_forces):
        add_global_parameters(force)
        force.setForceGroup(i + 1)
        integration_group.append(i + 1)
        new_system.addForce(force)
        
    groups = {'integration': set(integration_group)}
    
    if (compute_solvation_response):
        
        start_idx = new_system.getNumForces() + 1
        
        # Add dV/dl energy components, will give us d2V/dldR as well
        forces_to_add, deriv_groups = _get_alchemical_response(new_system, reference_force, solute_indicies,
                                      disable_alchemical_dispersion_correction, softcore_alpha, softcore_beta, softcore_m, softcore_n, softcore_a, softcore_b, start_idx)
        
        for i, force in enumerate(forces_to_add):
            add_global_parameters(force)
            new_system.addForce(force)
        
        groups.update(deriv_groups)
            
    # remove the original non-bonded force
    new_system.removeForce(force_idx)
    
    # add the new non-bonded force with alchemical interactions removed
    nonbonded_force.setForceGroup(0)
    new_system.addForce(nonbonded_force)
    
    ''' Return a dictionary containing force group sets as well as the new system with modified alchemical interactions
    
        integration : set corresponding to the forces that need to be used for system integration, **including** hybrid-hamiltonian components
        lambda_electrostatics: set corresponding to electrostatic dV/dl contributions (**not** for integration)
        lambda_sterics: set corresponding to steric dV/dl contributions (**not** for integration)
        
         python3 groups
         >>> {'integration', set[integration groups], 'lambda_electrostatics', set[dV_dl_{electrostatics}], 'lambda_sterics, set[dV_dl_{sterics}]'}
    '''
    
    return new_system, groups


def _get_alchemical_response(system, reference_force, solute_indicies, disable_alchemical_dispersion_correction=False,
                                       softcore_alpha=0.4, softcore_beta=0.0, softcore_m=1, softcore_n=6, softcore_a=2, softcore_b=2, group_id_start=10):

    alchemical_atoms = set(solute_indicies)
    chemical_atoms = set(range(system.getNumParticles())).difference(alchemical_atoms)
    
    dsterics_mixing_roles, dexceptions_sterics_energy_expression = _get_sterics_expression_derivative()
    
    dsterics_energy_expression = dexceptions_sterics_energy_expression + dsterics_mixing_roles
    
    delectrostatics_mixing_rules, dexceptions_electrostatics_energy_expression = _get_electrostatics_expression_derivative(reference_force)
    
    delectrostatics_energy_expression = dexceptions_electrostatics_energy_expression + delectrostatics_mixing_rules
    
    dna_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, dsterics_energy_expression,
                                                    True, 'lambda_sterics', False)
    
    dall_sterics_custom_nonbonded_forces = [dna_sterics_custom_nonbonded_force]
    
    dna_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, delectrostatics_energy_expression,
                                                            True, 'lambda_electrostatics', False)
    dall_electrostatics_custom_nonbonded_forces = [dna_electrostatics_custom_nonbonded_force]
    
    # CustomBondForces represent exceptions not picked up by exclusions 
    dna_sterics_custom_bond_force = create_force(openmm.CustomBondForce, dexceptions_sterics_energy_expression,
                                                True, 'lambda_sterics', False)
    
    dall_sterics_custom_bond_forces = [dna_sterics_custom_bond_force]
    
    dna_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, dexceptions_electrostatics_energy_expression,
                                                       True, 'lambda_electrostatics', False)
   
    dall_electrostatics_custom_bond_forces = [dna_electrostatics_custom_bond_force]
    
    electrostatics_group = []
    sterics_group = []
    
    for force in dall_electrostatics_custom_nonbonded_forces:
        force.addPerParticleParameter("charge")
        force.setUseSwitchingFunction(False)
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setUseLongRangeCorrection(False)  
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setForceGroup(group_id_start)
        
        electrostatics_group.append(group_id_start)
        
    for force in dall_electrostatics_custom_bond_forces:
        force.addPerBondParameter("chargeprod")  # charge product
        force.setForceGroup(group_id_start + 1)
        
        electrostatics_group.append(group_id_start + 1)
        
    for force in dall_sterics_custom_nonbonded_forces:
        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon") 
        force.setUseSwitchingFunction(reference_force.getUseSwitchingFunction())
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setSwitchingDistance(reference_force.getSwitchingDistance())
        if disable_alchemical_dispersion_correction:
            force.setUseLongRangeCorrection(False)
        else:
            force.setUseLongRangeCorrection(reference_force.getUseDispersionCorrection())
    
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setForceGroup(group_id_start + 2)
        
        sterics_group.append(group_id_start + 2)
        
    for force in dall_sterics_custom_bond_forces:
        force.addPerBondParameter("sigma")  
        force.addPerBondParameter("epsilon")
        force.setForceGroup(group_id_start + 3)
        
        sterics_group.append(group_id_start + 3)
    
    # add all particles to all custom forces...
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
      
        for force in dall_sterics_custom_nonbonded_forces:
            force.addParticle([sigma, epsilon])
      
        for force in dall_electrostatics_custom_nonbonded_forces:
            force.addParticle([charge])
            
    # Now restrict pairwise interactions to their respective groups
    
    dna_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    dna_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)

    # now lets handle exclusions and exceptions
    
    dall_custom_nonbonded_forces = dall_electrostatics_custom_nonbonded_forces + dall_sterics_custom_nonbonded_forces
    
    for exception_index in range(reference_force.getNumExceptions()):
    
        iatom, jatom, chargeprod, sigma, epsilon = reference_force.getExceptionParameters(exception_index)
    
        # All non-bonded forces must have same number of exceptions/exclusions on CUDA
        for force in dall_custom_nonbonded_forces:
            force.addExclusion(iatom, jatom)

        is_exception_epsilon = abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 0.0
        is_exception_chargeprod = abs(chargeprod.value_in_unit_system(unit.md_unit_system)) > 0.0
        
        both_alchemical = iatom in alchemical_atoms and jatom in alchemical_atoms
        at_least_one_alchemical = iatom in alchemical_atoms or jatom in alchemical_atoms
        only_one_alchemical = at_least_one_alchemical and not both_alchemical

        if only_one_alchemical:
            if is_exception_epsilon:
                dna_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
            if is_exception_chargeprod:
                dna_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])
    
    dall_custom_forces = [dna_electrostatics_custom_nonbonded_force, dna_electrostatics_custom_bond_force, dna_sterics_custom_nonbonded_force, dna_sterics_custom_bond_force]
    
    groups = {'lambda_electrostatics': set(electrostatics_group), 'lambda_sterics': set(sterics_group)}
    
    return dall_custom_forces, groups


def create_end_state_system(system, solute_indicies):
    
    new_system = copy.deepcopy(system)
    
    alchemical_atoms = set(solute_indicies)
    
    chemical_atoms = set(range(system.getNumParticles())).difference(alchemical_atoms)
    
    integration_group = []
    
    for force in new_system.getForces():
        # group 0 will be used as integration group, so move all existing forces here
        force.setForceGroup(0)
        
    integration_group.append(0)
    
    force_idx, reference_force = forces.find_forces(new_system, openmm.NonbondedForce, only_one=True)
    
    exceptions_sterics_energy_expression = '4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;'
    sterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
    
    epsilon_solvent = reference_force.getReactionFieldDielectric()
    rcut = reference_force.getCutoffDistance()

    k_rf = rcut ** (-3) * ((epsilon_solvent - 1) / (2 * epsilon_solvent + 1))
    k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
     
    c_rf = rcut ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + 1))
    c_rf = c_rf.value_in_unit_system(unit.md_unit_system)

    exceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*chargeprod*(r^(-1) + k_rf*r^2 - c_rf);'
    exceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    exceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
    electrostatics_mixing_rules = 'chargeprod = charge1*charge2;'
    
    sterics_energy_expression = exceptions_sterics_energy_expression + sterics_mixing_rules
    electrostatics_energy_expression = exceptions_electrostatics_energy_expression + electrostatics_mixing_rules

    nonbonded_force = copy.deepcopy(reference_force)
    
    na_electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce(electrostatics_energy_expression)
    aa_electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce(electrostatics_energy_expression)
    
    na_electrostatics_custom_bond_force = openmm.CustomBondForce(exceptions_electrostatics_energy_expression)
    aa_electrostatics_custom_bond_force = openmm.CustomBondForce(exceptions_electrostatics_energy_expression)
    
    na_sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(sterics_energy_expression)
    aa_sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(sterics_energy_expression)
   
    # CustomBondForces represent exceptions not picked up by exclusions 
    na_sterics_custom_bond_force = openmm.CustomBondForce(exceptions_sterics_energy_expression)
    aa_sterics_custom_bond_force = openmm.CustomBondForce(exceptions_sterics_energy_expression)

    all_electrostatics_custom_nonbonded_forces = [na_electrostatics_custom_nonbonded_force, aa_electrostatics_custom_nonbonded_force]
    all_electrostatics_custom_bond_forces = [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]
    all_sterics_custom_nonbonded_forces = [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]
    all_sterics_custom_bond_forces = [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]
    
    for force in all_sterics_custom_nonbonded_forces:
        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon") 
        force.setUseSwitchingFunction(reference_force.getUseSwitchingFunction())
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setSwitchingDistance(reference_force.getSwitchingDistance())
        
        force.setUseSwitchingFunction(False)
        # force.setUseLongRangeCorrection(reference_force.getUseDispersionCorrection())
    
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        
    for force in all_electrostatics_custom_nonbonded_forces:
        force.addPerParticleParameter("charge")
        force.setUseSwitchingFunction(False)
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setUseLongRangeCorrection(False)  
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    
    for force in all_sterics_custom_bond_forces:
        force.addPerBondParameter("sigma")  
        force.addPerBondParameter("epsilon")
    
    for force in all_electrostatics_custom_bond_forces:
        force.addPerBondParameter("chargeprod")  # charge product
        # force.addPerBondParameter("sigma") 
    
    # fix any missing values that can screw things up
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
        
        if sigma == 0.0 * unit.angstrom:
            warning_msg = 'particle %d has Lennard-Jones sigma = 0 (charge=%s, sigma=%s, epsilon=%s); setting sigma=1A'
            logger.warning(warning_msg % (particle_index, str(charge), str(sigma), str(epsilon)))
            sigma = 3.0 * unit.angstrom
            nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
            
    # also do the same for exceptions
    for exception_index in range(reference_force.getNumExceptions()):
  
        [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)

        if sigma == 0.0 * unit.angstrom:
            warning_msg = 'exception %d has Lennard-Jones sigma = 0 (iatom=%d, jatom=%d, chargeprod=%s, sigma=%s, epsilon=%s); setting sigma=1A'
            logger.warning(warning_msg % (exception_index, iatom, jatom, str(chargeprod), str(sigma), str(epsilon)))
            sigma = 3.0 * unit.angstrom
            # Fix it.
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)
    
    # add all particles to all custom forces...
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
        
        for force in all_sterics_custom_nonbonded_forces:
            force.addParticle([sigma, epsilon])
      
        for force in all_electrostatics_custom_nonbonded_forces:
            force.addParticle([charge])

    # now turn off interactions from alchemically-modified particles in unmodified nonbonded force
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)

        if particle_index in solute_indicies:
            nonbonded_force.setParticleParameters(particle_index, 0.0, sigma, 0.0)
            
    # Now restrict pairwise interactions to their respective groups
    na_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    na_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    # now lets handle exclusions and exceptions
    all_custom_nonbonded_forces = all_electrostatics_custom_nonbonded_forces + all_sterics_custom_nonbonded_forces 
    
    for exception_index in range(reference_force.getNumExceptions()):
    
        iatom, jatom, chargeprod, sigma, epsilon = reference_force.getExceptionParameters(exception_index)
    
        # All non-bonded forces must have same number of exceptions/exclusions on CUDA
        for force in all_custom_nonbonded_forces:
            force.addExclusion(iatom, jatom)

        is_exception_epsilon = abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 0.0
        is_exception_chargeprod = abs(chargeprod.value_in_unit_system(unit.md_unit_system)) > 0.0
        
        both_alchemical = iatom in alchemical_atoms and jatom in alchemical_atoms
        at_least_one_alchemical = iatom in alchemical_atoms or jatom in alchemical_atoms
        only_one_alchemical = at_least_one_alchemical and not both_alchemical
        
        # If exception (and not exclusion), add special CustomBondForce terms to handle alchemically modified LJ and reactionfield electrostatics
        if both_alchemical:
            if is_exception_epsilon:
                aa_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
            if is_exception_chargeprod:
                aa_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])

        # When this is a single region we model the exception between alchemical
        # and non-alchemical particles using a single custom bond.
        
        elif only_one_alchemical:
            if is_exception_epsilon:
                na_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
            if is_exception_chargeprod:
                na_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])
        # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce
        
        # remove this exception in original reference force
        if at_least_one_alchemical:
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, 0.0, sigma, 0.0)
    
    all_custom_forces = (all_electrostatics_custom_nonbonded_forces + all_electrostatics_custom_bond_forces + all_sterics_custom_nonbonded_forces + all_sterics_custom_bond_forces)

    all_na_forces = [na_electrostatics_custom_nonbonded_force, na_electrostatics_custom_bond_force, na_sterics_custom_nonbonded_force, na_sterics_custom_bond_force]
    
    names = ["na_electrostatics_nbf", "aa_electrostatics_nbf", "na_electrostatics_bf", "aa_electrostatics_bf",
             "na_sterics_nbf", "aa_sterics_nbf", "na_sterics_bf", "aa_sterics_bf" ]
    
    interaction_forces = []
    
    for i, force in enumerate(all_custom_forces):
        force.setForceGroup(i + 1)
        integration_group.append(i + 1)
        if (force in all_na_forces):
            interaction_forces.append(i + 1)
        new_system.addForce(force)
        
    groups = {'integration': set(integration_group), 'interaction': set(interaction_forces)}
            
    # remove the original non-bonded force
    new_system.removeForce(force_idx)
    
    # add the new non-bonded force with alchemical interactions removed
    nonbonded_force.setForceGroup(0)
    new_system.addForce(nonbonded_force)
    
    return new_system, groups

        
def decompose_energy(context, system, include_derivatives=True):
    
    print ("NUM_FORCES: ", system.getNumForces())
    
    def getGlobalParameterInfo(force):
        s = "GLOBAL PARAMS:"
        for i in range(force.getNumGlobalParameters()):
            s += " " + str(force.getGlobalParameterName(i))
        return s

    def get_forces_with_group(system, group_id):
        forces = system.getForces()
        
        force_list = []
        
        for force in forces:
            if(force.getForceGroup() == group_id):
                force_list.append(force)
        
        return force_list
            
    for i in range(0, 32):
        forces = get_forces_with_group(system, i)
        
        if (len(forces) > 0):
            print ("FORCE GROUP:", i, "num_forces_with_group:", len(forces))
            print (">> Force Classes:")
            for v in forces:
                print (">>", v.__class__.__name__)
                if ("Custom" in v.__class__.__name__):
                    print (">>", getGlobalParameterInfo(v))
                    print (">>", v.getEnergyFunction())
            print (">>")
            
            state = context.getState(getEnergy=True, getParameterDerivatives=include_derivatives, groups=set([i]))
            
            print ("-PE: ", state.getPotentialEnergy())
            
            if (include_derivatives):
                energy_derivs = state.getEnergyParameterDerivatives()
                print ("-Derivatives:")
                print (energy_derivs.keys())
                print (energy_derivs.values())
            print ("------")
            print("")
    
