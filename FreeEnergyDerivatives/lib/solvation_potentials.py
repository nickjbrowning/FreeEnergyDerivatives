from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from openmmtools.constants import ONE_4PI_EPS0
from openmmtools import forces

import copy


def _get_sterics_expression():
    exceptions_sterics_energy_expression = '4.0*lambda_sterics*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;'
    exceptions_sterics_energy_expression += 'reff_sterics = (softcore_alpha*sigma^softcore_n *(1.0-lambda_sterics^softcore_a) + r^softcore_n)^(1/softcore_n);'
    
    sterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
    
    return sterics_mixing_rules, exceptions_sterics_energy_expression


def _get_sterics_expression_derivative():
        exceptions_sterics_energy_expression = '4.0*epsilon*x*(x-1.0) + lambda_sterics*4*epsilon*(dxdl*(x-1.0) + x*dxdl); x = (sigma/reff_sterics)^6;'
        exceptions_sterics_energy_expression += 'dxdl = -6.0*(sigma^6/reff_sterics^7) * drdl;'
        exceptions_sterics_energy_expression += 'drdl = -1.0 * softcore_a*lambda_sterics^(softcore_a - 1.0)*softcore_alpha*sigma^softcore_n * (1.0/softcore_n)*((1.0-lambda_sterics^softcore_a)*softcore_alpha*sigma^softcore_n + r^softcore_n)^((1.0/softcore_n) - 1.0);'
        
        exceptions_sterics_energy_expression += 'reff_sterics = ((1.0-lambda_sterics^softcore_a)*softcore_alpha*sigma^softcore_n  + r^softcore_n)^(1.0/softcore_n);'
        
        sterics_mixing_rules = 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);'
        
        return sterics_mixing_rules, exceptions_sterics_energy_expression


def _get_electrostatics_expression(reference_force):
    
    epsilon_solvent = reference_force.getReactionFieldDielectric()
    rcut = reference_force.getCutoffDistance()
    eps = (1.0 / ONE_4PI_EPS0) * 4 * np.pi
    
    k_rf = rcut ** (-3) * ((epsilon_solvent - eps) / (2 * epsilon_solvent + eps))
    k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
    
    c_rf = rcut ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + eps))
    c_rf = c_rf.value_in_unit_system(unit.md_unit_system)
    
#     k_rf = rcut ** (-3) * ((epsilon_solvent - 1) / (2 * epsilon_solvent + 1))
#     k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
#     
#     c_rf = rcut ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + 1))
#     c_rf = c_rf.value_in_unit_system(unit.md_unit_system)
    
    exceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*lambda_electrostatics*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);'
    exceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    exceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
    exceptions_electrostatics_energy_expression += 'reff_electrostatics=(softcore_beta*(1.0-lambda_electrostatics^softcore_b) + r^softcore_m)^(1/softcore_m);'
        
    electrostatics_mixing_rules = 'chargeprod = charge1*charge2;'

    return electrostatics_mixing_rules, exceptions_electrostatics_energy_expression


def _get_electrostatics_expression_derivative(reference_force):
    epsilon_solvent = reference_force.getReactionFieldDielectric()
    rcut = reference_force.getCutoffDistance()
    
    eps = 1 / ONE_4PI_EPS0 * 4 * np.pi
    
    k_rf = rcut ** (-3) * ((epsilon_solvent - eps) / (2 * epsilon_solvent + eps))
    k_rf = k_rf.value_in_unit_system(unit.md_unit_system)  
    
    c_rf = rcut ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + eps))
    c_rf = c_rf.value_in_unit_system(unit.md_unit_system)
    
    drdl = 'drdl = -softcore_b*lambda_electrostatics^(softcore_b - 1.0)*softcore_beta*(1/softcore_m)*(softcore_beta*(1.0-lambda_electrostatics^softcore_b) +r^softcore_m)^((1/softcore_m) - 1.0);'
    
    exceptions_electrostatics_energy_expression = 'ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf)'
    exceptions_electrostatics_energy_expression += '+ ONE_4PI_EPS0*lambda_electrostatics*chargeprod*(-reff_electrostatics^(-2.0)*drdl + 2*k_rf*reff_electrostatics*drdl);'
    exceptions_electrostatics_energy_expression += drdl
    exceptions_electrostatics_energy_expression += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
    exceptions_electrostatics_energy_expression += 'k_rf = {k_rf};c_rf = {c_rf};'.format(k_rf=k_rf, c_rf=c_rf)
    
    exceptions_electrostatics_energy_expression += 'reff_electrostatics=(softcore_beta*(1.0-lambda_electrostatics^softcore_b) + r^softcore_m)^(1/softcore_m);'
        
    electrostatics_mixing_rules = 'chargeprod = charge1*charge2;'

    return electrostatics_mixing_rules, exceptions_electrostatics_energy_expression


def create_force(force_constructor, energy_expression, is_lambda_controlled=False, lambda_var=None):
   
    if (is_lambda_controlled):
        force = force_constructor(energy_expression)
        force.addGlobalParameter(lambda_var, 1.0)
        force.addEnergyParameterDerivative(lambda_var)  # also request that we compute dE/dlambda
    else:
        energy_expression = energy_expression + lambda_var + '=1.0;'
        force = force_constructor(energy_expression)
    return force


def create_alchemical_system(system, solute_indicies, compute_solvation_response=False,
                                      annihilate_sterics=False, annihilate_electrostatics=False,
                                      disable_alchemical_dispersion_correction=False, softcore_alpha=0.4, softcore_beta=(2.0 * unit.angstroms) ** 6.0, softcore_m=6.0, softcore_n=6.0, softcore_a=1.0, softcore_b=1.0):
    
    new_system = copy.deepcopy(system)
    
    alchemical_atoms = set(solute_indicies)
    chemical_atoms = set(range(system.getNumParticles())).difference(alchemical_atoms)
    
    for force in new_system.getForces():
        # group 0 will be used as integration group
        force.setForceGroup(0)
        
    force_idx, reference_force = forces.find_forces(new_system, openmm.NonbondedForce, only_one=True)
    
    nonbonded_force = copy.deepcopy(reference_force)
    
    sterics_mixing_roles, exceptions_sterics_energy_expression = _get_sterics_expression()
    
    sterics_energy_expression = exceptions_sterics_energy_expression + sterics_mixing_roles
    
    electrostatics_mixing_rules, exceptions_electrostatics_energy_expression = _get_electrostatics_expression(reference_force)
    
    electrostatics_energy_expression = exceptions_electrostatics_energy_expression + electrostatics_mixing_rules
    
    na_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                    True, 'lambda_sterics')
    aa_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                    annihilate_sterics, 'lambda_sterics')
    
    all_sterics_custom_nonbonded_forces = [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]
    
    na_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                            True, 'lambda_electrostatics')
    
    aa_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                            annihilate_electrostatics, 'lambda_electrostatics')
    
    all_electrostatics_custom_nonbonded_forces = [na_electrostatics_custom_nonbonded_force, aa_electrostatics_custom_nonbonded_force]
    
    # CustomBondForces represent exceptions not picked up by exclusions 
    na_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                True, 'lambda_sterics')
    aa_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                annihilate_sterics, 'lambda_sterics')
    
    all_sterics_custom_bond_forces = [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]
    
    na_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                       True, 'lambda_electrostatics')
    aa_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                       annihilate_electrostatics, 'lambda_electrostatics')
    
    all_electrostatics_custom_bond_forces = [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]

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
        # force.addPerParticleParameter("sigma") 
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
            nonbonded_force.setParticleParameters(particle_index, abs(0.0 * charge), sigma, abs(0 * epsilon))
            
    # Now restrict pairwise interactions to their respective groups
    na_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    na_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    # now lets handle exclusions and exceptions
    all_custom_nonbonded_forces = all_sterics_custom_nonbonded_forces + all_electrostatics_custom_nonbonded_forces
    
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
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, abs(0.0 * chargeprod), sigma, abs(0.0 * epsilon))
    
    all_custom_forces = (all_custom_nonbonded_forces + all_sterics_custom_bond_forces + all_electrostatics_custom_bond_forces)
    
    def add_global_parameters(force):
        force.addGlobalParameter('softcore_alpha', softcore_alpha)
        force.addGlobalParameter('softcore_beta', softcore_beta)
        force.addGlobalParameter('softcore_a', softcore_a)
        force.addGlobalParameter('softcore_b', softcore_b)
        force.addGlobalParameter('softcore_m', softcore_m)
        force.addGlobalParameter('softcore_n', softcore_n)
    
    # add all forces representing alchemical interactions
    for force in all_custom_forces:
        add_global_parameters(force)
        force.setForceGroup(0)  # integration force group
        new_system.addForce(force)
    
    if (compute_solvation_response):
        # Add dV/dl energy components
        _add_alchemical_response(new_system, reference_force, solute_indicies,
                                      annihilate_sterics, annihilate_electrostatics,
                                      disable_alchemical_dispersion_correction, softcore_alpha, softcore_beta, softcore_m, softcore_n, softcore_a, softcore_b)
    
    # remove the original non-bonded force
    new_system.removeForce(force_idx)
    # add the new non-bonded force with alchemical interactions removed
    nonbonded_force.setForceGroup(0)
    new_system.addForce(nonbonded_force)
    
    return new_system


def _add_alchemical_response(system, reference_force, solute_indicies, annihilate_sterics=False, annihilate_electrostatics=False, disable_alchemical_dispersion_correction=False,
                                       softcore_alpha=0.4, softcore_beta=(2.0 * unit.angstroms) ** 6, softcore_m=6, softcore_n=6, softcore_a=1, softcore_b=1):
    
    alchemical_atoms = set(solute_indicies)
    chemical_atoms = set(range(system.getNumParticles())).difference(alchemical_atoms)
    
    sterics_mixing_roles, exceptions_sterics_energy_expression = _get_sterics_expression_derivative()
    
    sterics_energy_expression = exceptions_sterics_energy_expression + sterics_mixing_roles
    
    electrostatics_mixing_rules, exceptions_electrostatics_energy_expression = _get_electrostatics_expression_derivative(reference_force)
    
    electrostatics_energy_expression = exceptions_electrostatics_energy_expression + electrostatics_mixing_rules
    
    na_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                    True, 'lambda_sterics')
    aa_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                    annihilate_sterics, 'lambda_sterics')
    
    all_sterics_custom_nonbonded_forces = [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]
    
    na_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                            True, 'lambda_electrostatics')
    
    aa_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                            annihilate_electrostatics, 'lambda_electrostatics')
    
    all_electrostatics_custom_nonbonded_forces = [na_electrostatics_custom_nonbonded_force, aa_electrostatics_custom_nonbonded_force]
    
    # CustomBondForces represent exceptions not picked up by exclusions 
    na_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                True, 'lambda_sterics')
    aa_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                annihilate_sterics, 'lambda_sterics')
    
    all_sterics_custom_bond_forces = [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]
    
    na_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                       True, 'lambda_electrostatics')
    aa_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                       annihilate_electrostatics, 'lambda_electrostatics')
    
    all_electrostatics_custom_bond_forces = [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]

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
        
        force.setForceGroup(2)
        
    for force in all_sterics_custom_bond_forces:
        force.addPerBondParameter("sigma")  
        force.addPerBondParameter("epsilon")
        
        force.setForceGroup(2)
        
    for force in all_electrostatics_custom_nonbonded_forces:
        force.addPerParticleParameter("charge")
        # force.addPerParticleParameter("sigma") 
        force.setUseSwitchingFunction(False)
        force.setCutoffDistance(reference_force.getCutoffDistance())
        force.setUseLongRangeCorrection(False)  
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        
        force.setForceGroup(1)
        
    for force in all_electrostatics_custom_bond_forces:
        force.addPerBondParameter("chargeprod")  # charge product
        # force.addPerBondParameter("sigma") 
        
        force.setForceGroup(1)
    
    # add all particles to all custom forces...
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
      
        for force in all_sterics_custom_nonbonded_forces:
            force.addParticle([sigma, epsilon])
      
        for force in all_electrostatics_custom_nonbonded_forces:
            force.addParticle([charge])
            
    # Now restrict pairwise interactions to their respective groups
    na_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    na_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    # now lets handle exclusions and exceptions
    all_custom_nonbonded_forces = all_sterics_custom_nonbonded_forces + all_electrostatics_custom_nonbonded_forces
    
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

        elif only_one_alchemical:
            if is_exception_epsilon:
                na_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
            if is_exception_chargeprod:
                na_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])
        # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce
    
    all_custom_forces = (all_custom_nonbonded_forces + all_sterics_custom_bond_forces + all_electrostatics_custom_bond_forces)
    
    def add_global_parameters(force):
        force.addGlobalParameter('softcore_alpha', softcore_alpha)
        force.addGlobalParameter('softcore_beta', softcore_beta)
        force.addGlobalParameter('softcore_a', softcore_a)
        force.addGlobalParameter('softcore_b', softcore_b)
        force.addGlobalParameter('softcore_m', softcore_m)
        force.addGlobalParameter('softcore_n', softcore_n)
    
    # add all forces representing alchemical interactions
    for force in all_custom_forces:
        add_global_parameters(force)
        system.addForce(force)
