

def create_alchemical_system2(system, solute_indicies, compute_solvation_response=False,
                                      annihilate_sterics=False, annihilate_electrostatics=False,
                                      disable_alchemical_dispersion_correction=False, softcore_alpha=0.4, softcore_beta=0.0, softcore_m=1.0, softcore_n=6.0, softcore_a=2.0, softcore_b=2.0):
    
    new_system = copy.deepcopy(system)
    
    alchemical_atoms = set(solute_indicies)
    chemical_atoms = set(range(system.getNumParticles())).difference(alchemical_atoms)
    
    for force in new_system.getForces():
        # group 0 will be used as integration group
        force.setForceGroup(0)
        
    force_idx, reference_force = forces.find_forces(system, openmm.NonbondedForce, only_one=True)
    
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
    
    if (compute_solvation_response):
        dsterics_mixing_roles, dexceptions_sterics_energy_expression = _get_sterics_expression_derivative()
    
        dsterics_energy_expression = dexceptions_sterics_energy_expression + dsterics_mixing_roles
        
        delectrostatics_mixing_rules, dexceptions_electrostatics_energy_expression = _get_electrostatics_expression_derivative(reference_force)
        
        delectrostatics_energy_expression = dexceptions_electrostatics_energy_expression + delectrostatics_mixing_rules
        
        dna_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, dsterics_energy_expression,
                                                        True, 'lambda_sterics', False)
        
        dna_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, delectrostatics_energy_expression,
                                                                True, 'lambda_electrostatics', False)
        
        # CustomBondForces represent exceptions not picked up by exclusions 
        dna_sterics_custom_bond_force = create_force(openmm.CustomBondForce, dexceptions_sterics_energy_expression,
                                                    True, 'lambda_sterics', False)
        
        dna_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, dexceptions_electrostatics_energy_expression,
                                                           True, 'lambda_electrostatics', False)

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
        force.addPerBondParameter("chargeprod")  
        
    if (compute_solvation_response):

        dna_sterics_custom_nonbonded_force.addPerParticleParameter("sigma")
        dna_sterics_custom_nonbonded_force.addPerParticleParameter("epsilon") 
        dna_sterics_custom_nonbonded_force.setUseSwitchingFunction(reference_force.getUseSwitchingFunction())
        dna_sterics_custom_nonbonded_force.setCutoffDistance(reference_force.getCutoffDistance())
        dna_sterics_custom_nonbonded_force.setSwitchingDistance(reference_force.getSwitchingDistance())
        if disable_alchemical_dispersion_correction:
            dna_sterics_custom_nonbonded_force.setUseLongRangeCorrection(False)
        else:
            dna_sterics_custom_nonbonded_force.setUseLongRangeCorrection(reference_force.getUseDispersionCorrection())
        dna_sterics_custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
       
        dna_electrostatics_custom_nonbonded_force.addPerParticleParameter("charge")
        dna_electrostatics_custom_nonbonded_force.setUseSwitchingFunction(False)
        dna_electrostatics_custom_nonbonded_force.setCutoffDistance(reference_force.getCutoffDistance())
        dna_electrostatics_custom_nonbonded_force.setUseLongRangeCorrection(False)  
        dna_electrostatics_custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        dna_sterics_custom_bond_force.addPerBondParameter("sigma")  
        dna_sterics_custom_bond_force.addPerBondParameter("epsilon")

        dna_electrostatics_custom_bond_force.addPerBondParameter("chargeprod")  
        
    # fix any missing values that can screw things up
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
        
        if sigma == 0.0 * unit.angstrom:
            print ("YO")
            warning_msg = 'particle %d has Lennard-Jones sigma = 0 (charge=%s, sigma=%s, epsilon=%s); setting sigma=1A'
            logger.warning(warning_msg % (particle_index, str(charge), str(sigma), str(epsilon)))
            sigma = 3.0 * unit.angstrom
            nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
            
    # also do the same for exceptions
    for exception_index in range(reference_force.getNumExceptions()):
        
        [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)

        if sigma == 0.0 * unit.angstrom:
            print ("YO2")
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
            
        if (compute_solvation_response):
            dna_sterics_custom_nonbonded_force.addParticle([sigma, epsilon])
            dna_electrostatics_custom_nonbonded_force.addParticle([charge])

    # now turn off interactions from alchemically-modified particles in unmodified nonbonded force
    for particle_index in range(reference_force.getNumParticles()):

        [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)

        if particle_index in alchemical_atoms:
            nonbonded_force.setParticleParameters(particle_index, 0.0, sigma, 0.0)
            
    # Now restrict pairwise interactions to their respective groups
    na_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    na_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atoms, alchemical_atoms)
    
    if (compute_solvation_response):
        dna_sterics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
        dna_electrostatics_custom_nonbonded_force.addInteractionGroup(chemical_atoms, alchemical_atoms)
    
    # now lets handle exclusions and exceptions
    all_custom_nonbonded_forces = all_sterics_custom_nonbonded_forces + all_electrostatics_custom_nonbonded_forces
        
    for exception_index in range(reference_force.getNumExceptions()):
    
        iatom, jatom, chargeprod, sigma, epsilon = reference_force.getExceptionParameters(exception_index)
    
        # All non-bonded forces must have same number of exceptions/exclusions on CUDA
        for force in all_custom_nonbonded_forces:
            force.addExclusion(iatom, jatom)
        
        if (compute_solvation_response):
                dna_sterics_custom_nonbonded_force.addExclusion(iatom, jatom)
                dna_electrostatics_custom_nonbonded_force.addExclusion(iatom, jatom)

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
                
            if (compute_solvation_response):
                if is_exception_epsilon:
                    dna_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                if is_exception_chargeprod:
                    dna_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod])
                
        # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce
        
        # remove this exception in original reference force
        if at_least_one_alchemical:
            nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, 0.0, sigma, 0.0)
    
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
        
        dna_electrostatics_custom_nonbonded_force.setForceGroup(1)
        add_global_parameters(dna_electrostatics_custom_nonbonded_force)
        new_system.addForce(dna_electrostatics_custom_nonbonded_force)
        
        dna_electrostatics_custom_bond_force.setForceGroup(1)
        add_global_parameters(dna_electrostatics_custom_bond_force)
        new_system.addForce(dna_electrostatics_custom_bond_force)
        
        dna_sterics_custom_nonbonded_force.setForceGroup(2)
        add_global_parameters(dna_sterics_custom_nonbonded_force)
        new_system.addForce(dna_sterics_custom_nonbonded_force)
        
        dna_sterics_custom_bond_force.setForceGroup(2)
        add_global_parameters(dna_sterics_custom_bond_force)
        new_system.addForce(dna_sterics_custom_bond_force)
        
    # remove the original non-bonded force
    new_system.removeForce(force_idx)
    
    # add the new non-bonded force with alchemical interactions removed
    nonbonded_force.setForceGroup(0)
    new_system.addForce(nonbonded_force)
    
    return new_system
