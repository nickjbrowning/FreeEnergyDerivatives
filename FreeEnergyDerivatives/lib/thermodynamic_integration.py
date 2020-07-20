import numpy as np
import scipy
from simtk import unit


def collect_dvdl_values(simulation, lambda_grid, nsamples, nsteps, solute_indexes, force_groups, lambda_var, debug=True, compute_forces_along_path=False):
    
    dV = np.zeros((len(lambda_grid), nsamples), dtype=np.float64)
    sample_forces = np.zeros((len(lambda_grid), nsamples, len(solute_indexes), 3), dtype=np.float64)

    if (debug):
        print ("lambda variable: %s nlambda %i nsamples %i" % (lambda_var, len(lambda_grid), nsamples))
        print ('force_groups[%s] = %s' % (lambda_var, force_groups[lambda_var]))
        print ("lambda", "mean(dV/dl)", "SEM(dV/dl)")
    
    for i, l in enumerate(lambda_grid):
        
        # assume lambda_grid has been generated in the direction coupled (\lambda=1.0) -> uncoupled (\lambda=0.0) 
        idx = len(lambda_grid) - i - 1
        
        simulation.context.setParameter(lambda_var, l)
        
        simulation.step(50000)  # equilibrate for 100ps before sampling data

        for iteration in range(nsamples):
            simulation.step(nsteps) 
            
            state = simulation.context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))

            energy_derivs = state.getEnergyParameterDerivatives()
            
            energy_deriv = energy_derivs[lambda_var]
            
            if (not compute_forces_along_path): 
                # be wary of using getEnergyParameterDerivatives() as of openMM 7.5, in my tests the electrostatic component
                # can be incorrectly split between lambda_electrostatics and lambda_sterics for some unknown reason (but the sum is correct)
                dV[idx, iteration] = energy_deriv  
            elif (compute_forces_along_path): 
                
                state_deriv = simulation.context.getState(getEnergy=True, getForces=True, groups=force_groups[lambda_var])
                
                dvdl = state_deriv.getPotentialEnergy()
                dV[idx, iteration] = dvdl.value_in_unit_system(unit.md_unit_system)

                forces = state_deriv.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]
                sample_forces[idx, iteration, :, :] = forces
        
        if (debug):
            print ("%5.2f %5.2f %8.5f" % (simulation.context.getParameter(lambda_var), np.mean(dV[idx, :]), scipy.stats.sem(dV[idx, :])))
    
    return dV, sample_forces
