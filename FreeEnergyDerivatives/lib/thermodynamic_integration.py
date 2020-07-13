import numpy as np
from simtk import unit


def collect_dvdl_values(simulation, lambda_grid, nsamples, nsteps, solute_indexes, lambda_var=None, debug=True, compute_forces=False):
    
    dV = np.zeros((len(lambda_grid), nsamples))
    sample_forces = np.zeros((len(lambda_grid), nsamples, len(solute_indexes), 3))
    
    if (debug):
        print ("lambda variable: ", lambda_var)
        print ("lambda", "mean(dV/dl)", "SEM(dV/dl)")
    
    for i, l in enumerate(lambda_grid):
        
        # assume lambda_grid has been generated in reverse (i.e coupled -> uncoupled) 
        idx = len(lambda_grid) - i - 1
        
        simulation.context.setParameter(lambda_var, l)
        
        simulation.step(50000)  # equilibrate for 100ps before sampling data
    
        for iteration in range(nsamples):
            simulation.step(nsteps) 
            
            state = simulation.context.getState(getEnergy=True, getParameterDerivatives=True)

            energy_derivs = state.getEnergyParameterDerivatives()
            
            energy_deriv = energy_derivs[lambda_var]
            
            dV[idx, iteration] = energy_deriv  
            
            if (compute_forces and i == 0):  # only collect forces at the end state
                state_deriv = simulation.context.getState(getForces=True)
                forces = state_deriv.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]
                sample_forces[idx, iteration, :, :] = forces
        
        if (debug):
            print ("%5.2f %5.2f %8.5f" % 
                   (simulation.context.getParameter(lambda_var), np.average(dV[idx, :]), np.std(dV[idx, :]) / np.sqrt(nsamples))
                   )
    
    return dV, sample_forces
