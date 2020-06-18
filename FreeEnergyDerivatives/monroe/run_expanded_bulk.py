#!/usr/bin/env python

from __future__ import division, print_function

import sys, os
import copy
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as u
import parmed as pmd
from parmed.openmm.reporters import NetCDFReporter
from pymbar import mbar
from openmm_surface_affinities_lib import *

# Given a topology and structure file, this script sets up an alchemical system and runs an expanded ensemble simulation


def main(args):
  # Get the structure and topology files from the command line
  # ParmEd accepts a wide range of file types (Amber, GROMACS, CHARMM, OpenMM... but not LAMMPS) 
  try:
    topFile = args[0]
    strucFile = args[1]
  except IndexError:
    print("Specify topology and structure files from the command line.")
    sys.exit(2)
  
  print("Using topology file: %s" % topFile)
  print("Using structure file: %s" % strucFile)
  
  print("\nSetting up system...")
  
  # Load in the files for initial simulations
  top = pmd.load_file(topFile)
  struc = pmd.load_file(strucFile)
  
  # Transfer unit cell information to topology object
  top.box = struc.box[:]
  
  # Set up some global features to use in all simulations
  temperature = 298.15 * u.kelvin
  
  # Define the platform (i.e. hardware and drivers) to use for running the simulation
  # This can be CUDA, OpenCL, CPU, or Reference 
  # CUDA is for NVIDIA GPUs
  # OpenCL is for CPUs or GPUs, but must be used for old CPUs (not SSE4.1 compatible)
  # CPU only allows single precision (CUDA and OpenCL allow single, mixed, or double)
  # Reference is a clear, stable reference for other code development and is very slow, using double precision by default
  platform = mm.Platform.getPlatformByName('CUDA')
  prop = {  # 'Threads': '2', #number of threads for CPU - all definitions must be strings (I think)
          'Precision': 'mixed',  # for CUDA or OpenCL, select the precision (single, mixed, or double)
          'DeviceIndex': '0',  # selects which GPUs to use - set this to zero if using CUDA_VISIBLE_DEVICES
          'DeterministicForces': 'True'  # Makes sure forces with CUDA and PME are deterministic
         }
  
  # Create the OpenMM system that can be used as a reference
  systemRef = top.createSystem(
                               nonbondedMethod=app.PME,  # Uses PME for long-range electrostatics, simple cut-off for LJ
                               nonbondedCutoff=12.0 * u.angstroms,  # Defines cut-off for non-bonded interactions
                               rigidWater=True,  # Use rigid water molecules
                               constraints=app.HBonds,  # Constrains all bonds involving hydrogens
                               flexibleConstraints=False,  # Whether to include energies for constrained DOFs
                               removeCMMotion=True,  # Whether or not to remove COM motion (don't want to if part of system frozen)
  )

  # Set up the integrator to use as a reference
  integratorRef = mm.LangevinIntegrator(
                                        temperature,  # Temperature for Langevin
                                        1.0 / u.picoseconds,  # Friction coefficient
                                        2.0 * u.femtoseconds,  # Integration timestep
  )
  integratorRef.setConstraintTolerance(1.0E-08)

  # Get solute atoms and solute heavy atoms separately
  soluteIndices = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        soluteIndices.append(atom.idx)

  # And define lambda states of interest
  lambdaVec = np.array(# electrostatic lambda - 1.0 is fully interacting, 0.0 is non-interacting
                       [[1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       # LJ lambdas - 1.0 is fully interacting, 0.0 is non-interacting
                        [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00] 
                       ])

  # We need to add a custom non-bonded force for the solute being alchemically changed
  # Will be helpful to have handle on non-bonded force handling LJ and coulombic interactions
  NBForce = None
  for frc in systemRef.getForces():
    if (isinstance(frc, mm.NonbondedForce)):
      NBForce = frc

  # Turn off dispersion correction since have interface
  NBForce.setUseDispersionCorrection(False)

  forceLabelsRef = getForceLabels(systemRef)

  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

  # Separate out alchemical and regular particles using set objects
  alchemicalParticles = set(soluteIndices)
  chemicalParticles = set(range(systemRef.getNumParticles())) - alchemicalParticles

  # Define the soft-core function for turning on/off LJ interactions
  # In energy expressions for CustomNonbondedForce, r is a special variable and refers to the distance between particles
  # All other variables must be defined somewhere in the function.
  # The exception are variables like sigma1 and sigma2.
  # It is understood that a parameter will be added called 'sigma' and that the '1' and '2' are to specify the combining rule.
  softCoreFunction = '4.0*lambdaLJ*epsilon*x*(x-1.0); x = (1.0/reff_sterics);'
  softCoreFunction += 'reff_sterics = (0.5*(1.0-lambdaLJ) + ((r/sigma)^6));'
  softCoreFunction += 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)'
  # Define the system force for this function and its parameters
  SoftCoreForce = mm.CustomNonbondedForce(softCoreFunction)
  SoftCoreForce.addGlobalParameter('lambdaLJ', 1.0)  # Throughout, should follow convention that lambdaLJ=1.0 is fully-interacting state
  SoftCoreForce.addPerParticleParameter('sigma')
  SoftCoreForce.addPerParticleParameter('epsilon')

  # Will turn off electrostatics completely in the original non-bonded force
  # In the end-state, only want electrostatics inside the alchemical molecule
  # To do this, just turn ON a custom force as we turn OFF electrostatics in the original force
  ONE_4PI_EPS0 = 138.935456  # in kJ/mol nm/e^2
  soluteCoulFunction = '(1.0-(lambdaQ^2))*ONE_4PI_EPS0*charge/r;'
  soluteCoulFunction += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
  soluteCoulFunction += 'charge = charge1*charge2'
  SoluteCoulForce = mm.CustomNonbondedForce(soluteCoulFunction)
  # Note this lambdaQ will be different than for soft core (it's also named differently, which is CRITICAL)
  # This lambdaQ corresponds to the lambda that scales the charges to zero
  # To turn on this custom force at the same rate, need to multiply by (1.0-lambdaQ**2), which we do
  SoluteCoulForce.addGlobalParameter('lambdaQ', 1.0) 
  SoluteCoulForce.addPerParticleParameter('charge')

  # Also create custom force for intramolecular alchemical LJ interactions
  # Could include with electrostatics, but nice to break up
  # We could also do this with a separate NonbondedForce object, but it would be a little more work, actually
  soluteLJFunction = '4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;'
  soluteLJFunction += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
  SoluteLJForce = mm.CustomNonbondedForce(soluteLJFunction)
  SoluteLJForce.addPerParticleParameter('sigma')
  SoluteLJForce.addPerParticleParameter('epsilon')
  
  # Loop over all particles and add to custom forces
  # As we go, will also collect full charges on the solute particles
  # AND we will set up the solute-solute interaction forces
  alchemicalCharges = [[0]] * len(soluteIndices)
  for ind in range(systemRef.getNumParticles()):
    # Get current parameters in non-bonded force
    [charge, sigma, epsilon] = NBForce.getParticleParameters(ind)
    # Make sure that sigma is not set to zero! Fine for some ways of writing LJ energy, but NOT OK for soft-core!
    if sigma / u.nanometer == 0.0:
      newsigma = 0.3 * u.nanometer  # This 0.3 is what's used by GROMACS as a default value for sc-sigma
    else:
      newsigma = sigma
    # Add the particle to the soft-core force (do for ALL particles)
    SoftCoreForce.addParticle([newsigma, epsilon])
    # Also add the particle to the solute only forces
    SoluteCoulForce.addParticle([charge])
    SoluteLJForce.addParticle([sigma, epsilon])
    # If the particle is in the alchemical molecule, need to set it's LJ interactions to zero in original force
    if ind in soluteIndices:
      NBForce.setParticleParameters(ind, charge, sigma, epsilon * 0.0)
      # And keep track of full charge so we can scale it right by lambda
      alchemicalCharges[soluteIndices.index(ind)] = charge

  # Now we need to handle exceptions carefully
  for ind in range(NBForce.getNumExceptions()):
    [p1, p2, excCharge, excSig, excEps] = NBForce.getExceptionParameters(ind)
    # For consistency, must add exclusions where we have exceptions for custom forces
    SoftCoreForce.addExclusion(p1, p2)
    SoluteCoulForce.addExclusion(p1, p2)
    SoluteLJForce.addExclusion(p1, p2)

  # Only compute interactions between the alchemical and other particles for the soft-core force
  SoftCoreForce.addInteractionGroup(alchemicalParticles, chemicalParticles)

  # And only compute alchemical/alchemical interactions for other custom forces
  SoluteCoulForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)
  SoluteLJForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)

  # Set other soft-core parameters as needed
  SoftCoreForce.setCutoffDistance(12.0 * u.angstroms)
  SoftCoreForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoftCoreForce.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoftCoreForce)

  # Set other parameters as needed - note that for the solute force would like to set no cutoff
  # However, OpenMM won't allow a bunch of potentials with cutoffs then one without...
  # So as long as the solute is smaller than the cut-off, won't have any problems!
  SoluteCoulForce.setCutoffDistance(12.0 * u.angstroms)
  SoluteCoulForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoluteCoulForce.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoluteCoulForce)

  SoluteLJForce.setCutoffDistance(12.0 * u.angstroms)
  SoluteLJForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoluteLJForce.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoluteLJForce)

  forceLabelsRef = getForceLabels(systemRef)

  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

  # Do NVT simulation
  stateFileNVT, stateNVT = doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, pos=struc.positions)

  # And do NPT simulation using state information from NVT
  stateFileNPT, stateNPT = doSimNPT(top, systemRef, integratorRef, platform, prop, temperature, inBulk=True, state=stateFileNVT)

  # And do production run in expanded ensemble!
  stateFileProd, stateProd, weightsVec = doSimExpanded(top, systemRef, integratorRef, platform, prop, temperature, 0, lambdaVec, soluteIndices, alchemicalCharges, inBulk=True, state=stateFileNPT, nSteps=6000000)


if __name__ == "__main__":
  main(sys.argv[1:])
