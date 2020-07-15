import numpy as np
from simtk import unit


def collect_dvdl_values(simulation, lambda_grid, nsamples, nsteps, solute_indexes, lambda_var=None, debug=True, compute_forces_along_path=False):
    
    dV = np.zeros((len(lambda_grid), nsamples))
    sample_forces = np.zeros((len(lambda_grid), nsamples, len(solute_indexes), 3))
    
    if (debug):
        print ("lambda variable: ", lambda_var)
        print ("lambda", "mean(dV/dl)", "SEM(dV/dl)")
    
    for i, l in enumerate(lambda_grid):
        
        # assume lambda_grid has been generated in the direction coupled (\lambda=1.0) -> uncoupled (\lambda=0.0) 
        idx = len(lambda_grid) - i - 1
        
        simulation.context.setParameter(lambda_var, l)
        
        simulation.step(50000)  # equilibrate for 100ps before sampling data
        
        t = []
        for iteration in range(nsamples):
            simulation.step(nsteps) 
            
            state = simulation.context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))

            energy_derivs = state.getEnergyParameterDerivatives()
            
            energy_deriv = energy_derivs[lambda_var]
            
            dV[idx, iteration] = energy_deriv  
            
            if (compute_forces_along_path): 
                lambda_group = -1
                if (lambda_var == 'lambda_electrostatics'):
                    lambda_group = 1
                elif (lambda_var == 'lambda_sterics'):
                    lambda_group = 2
                
                state_deriv = simulation.context.getState(getEnergy=True, getForces=True, groups=set([lambda_group]))
                
                dvdl = state_deriv.getPotentialEnergy()
                
                t.append(dvdl._value)
                
                forces = state_deriv.getForces(asNumpy=True).value_in_unit_system(unit.md_unit_system)[solute_indexes, :]
                sample_forces[idx, iteration, :, :] = forces
        
        if (debug):
            print ("%5.2f %5.2f %8.5f" % 
                   (simulation.context.getParameter(lambda_var), np.average(dV[idx, :]), np.std(dV[idx, :]) / np.sqrt(nsamples))
                   )
            print ("Debug, analytical dv/dl derivative:", np.average(t))
    
    return dV, sample_forces
