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

# This library is intended to support the project for patterning surfaces to adjust surface affinities
# All the functions defined below make running various simulations with OpenMM much more convenient
# Some just report information about the way the system or simulation is setup
# Others actually perform various simulations
# This library of functions must be loaded in order 


def systemInfo(topObj, sysObj):
  # Let's check the set-up a little bit
  # Start with topObjology in ParmEd, then move to checking OpenMM interpretation
  print("%i atoms" % len(topObj.atoms))
  print("%i residues" % len(topObj.residues))
  print("%i bonds" % len(topObj.bonds))
  print("%i angles" % len(topObj.angles))
  print("%i dihedrals" % len(topObj.dihedrals))
  print("%i rb_torsions" % len(topObj.rb_torsions))
  print("%i exclusions" % len(topObj.adjusts))
  print(topObj.adjusts)
  print("%i constraints" % sysObj.getNumConstraints())
  # for i in range(sysObj.getNumConstraints()):
  #  print(sysObj.getConstraintParameters(i))
  # for i in range(sysObj.getNumParticles()):
  #  print("Atom %i with mass %s, virtual site? %s"%(i, str(sysObj.getParticleMass(i)), str(sysObj.isVirtualSite(i))))


def groupForces(sysObj):
  # Assign each for a name and an index
  # Returns a dictionary with force names as keys and force indices in the system object as definitions
  # print('')
  forces = {}
  forcesCounts = {}
  for k, frc in enumerate(sysObj.getForces()):
    # frc.setForceGroup(k)
    frcName = frc.__class__.__name__
    if frcName in forces.keys():
      forcesCounts[frcName] += 1
      frcName += '_%i' % forcesCounts[frcName]
    else:
      forcesCounts[frcName] = 1
    # print("Assigning force %s to group %i (object is %s)" % (frcName, k, frc.__str__()))
    forces[frcName] = k
  # print('')
  return forces


def getForceInfo(forceObj):
  # For regular or custom nonbonded forces, reports a bunch of information
  # Will add other forces as necessary
  fname = forceObj.__class__.__name__
  print('')
  print(fname)
  if 'NonbondedForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tNonbonded method: " + str(forceObj.getNonbondedMethod()))
    print("\tNumber particles: %i" % forceObj.getNumParticles())
    for k in range(5):
      print("\t\tParticle %i params: " % (k) + str(forceObj.getParticleParameters(k)))
    print("\tUses PBCs? " + str(forceObj.usesPeriodicBoundaryConditions()))
    print("\tCutoff distance: " + str(forceObj.getCutoffDistance()))
    print("\tUsing switching function? " + str(forceObj.getUseSwitchingFunction()))
    if 'CustomNonbonded' in fname:
      print("\tDispersion correction? " + str(forceObj.getUseLongRangeCorrection()))
      print("\tNumber exclusions: %i" % forceObj.getNumExclusions())
      for l in range(forceObj.getNumExclusions()):
        if l < 5 or l > (forceObj.getNumExclusions() - 5):
          print("\t\t" + str(forceObj.getExclusionParticles(l)))
      print("\tEnergy function: " + str(forceObj.getEnergyFunction()))
      print("\tNumber interaction groups: %i" % forceObj.getNumInteractionGroups())
      print("\tNumber global params: %i" % forceObj.getNumGlobalParameters())
      for l in range(forceObj.getNumGlobalParameters()):
        print("\t\tGlobal parameter: " + str(forceObj.getGlobalParameterName(l)))
        print("\t\tGlobal param value: " + str(forceObj.getGlobalParameterDefaultValue(l)))
      print("\tNumber per-particle params: %i" % forceObj.getNumPerParticleParameters())
      for l in range(forceObj.getNumPerParticleParameters()):
        print("\t\tParameter name: " + str(forceObj.getPerParticleParameterName(l)))
    else:
      print("\tDispersion correction? " + str(forceObj.getUseDispersionCorrection()))
      print("\tNumber of exceptions: %i" % forceObj.getNumExceptions())
      for l in range(forceObj.getNumExceptions()):
        if l < 5 or l > (forceObj.getNumExceptions() - 5):
          print("\t\t" + str(forceObj.getExceptionParameters(l)))
      print("\tReciprocal force group: %i" % forceObj.getReciprocalSpaceForceGroup())
  if 'HarmonicBondForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tNumber bonds: %i" % forceObj.getNumBonds())
    for l in range(forceObj.getNumBonds()):
      if l < 5 or l > (forceObj.getNumBonds() - 5):
        print("\t\t" + str(forceObj.getBondParameters(l)))
  if 'HarmonicAngleForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tNumber angles: %i" % forceObj.getNumAngles())
    for l in range(forceObj.getNumAngles()):
      if l < 5 or l > (forceObj.getNumAngles() - 5):
        print("\t\t" + str(forceObj.getAngleParameters(l)))
  if 'TorsionForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tNumber torsions: %i" % forceObj.getNumTorsions())  
    for l in range(forceObj.getNumTorsions()):
      if l < 5 or l > (forceObj.getNumTorsions() - 5):
        print("\t\t" + str(forceObj.getTorsionParameters(l)))
  if 'CustomBondForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tEnergy function: %s" % forceObj.getEnergyFunction())
    print("\tNumber of bonds: %i" % forceObj.getNumBonds())
    for l in range(forceObj.getNumBonds()):
      bondinfo = forceObj.getBondParameters(l)
      print("\t\tBond %i:" % l)
      print("\t\t\tAtoms %s and %s have per-bond parameters %s" % (str(bondinfo[0]),
                                                                 str(bondinfo[1]), str(bondinfo[2])))
    print("\tNumber global params: %i" % forceObj.getNumGlobalParameters())
    for l in range(forceObj.getNumGlobalParameters()):
      print("\t\tGlobal parameter: " + str(forceObj.getGlobalParameterName(l)))
      print("\t\tGlobal param value: " + str(forceObj.getGlobalParameterDefaultValue(l)))
  if 'CustomCentroidBondForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tEnergy function: %s" % forceObj.getEnergyFunction())
    print("\tNumber of bonds: %i" % forceObj.getNumBonds())
    for l in range(forceObj.getNumBonds()):
      bondinfo = forceObj.getBondParameters(l)
      print("\t\tBond %i:" % l)
      print("\t\t\tGroups are %s, per-bond parameters are %s" % (str(bondinfo[0]), str(bondinfo[1])))
      for groupind in bondinfo[0]:
        groupinfo = forceObj.getGroupParameters(groupind)
        print("\t\t\tGroup %i atom indices and weights:" % groupind)
        print("\t\t\t%s" % str(groupinfo[0]))
        print("\t\t\t%s" % str(groupinfo[1]))
    print("\tNumber global params: %i" % forceObj.getNumGlobalParameters())
    for l in range(forceObj.getNumGlobalParameters()):
      print("\t\tGlobal parameter: " + str(forceObj.getGlobalParameterName(l)))
      print("\t\tGlobal param value: " + str(forceObj.getGlobalParameterDefaultValue(l)))
  if 'CustomExternalForce' in fname:
    print("\tForce group: %i" % forceObj.getForceGroup())
    print("\tEnergy function: %s" % forceObj.getEnergyFunction())
    print("\tNumber of particles: %i" % forceObj.getNumParticles())
    for l in range(forceObj.getNumParticles()):
      particleinfo = forceObj.getParticleParameters(l)
      print("\t\tParticle %i:" % l)
      print("\t\t\tIndex: %i" % particleinfo[0])
      print("\t\t\tPer-particle parameters are %s" % str(particleinfo[1]))
    print("\tNumber global params: %i" % forceObj.getNumGlobalParameters())
    for l in range(forceObj.getNumGlobalParameters()):
      print("\t\tGlobal parameter: " + str(forceObj.getGlobalParameterName(l)))
      print("\t\tGlobal param value: " + str(forceObj.getGlobalParameterDefaultValue(l)))
  print('')


class RestraintReporter(object):

  def __init__(self, file, reportInterval, separator=',', header='#Time (ps)  RefZ (nm)  Z-dist (nm)  Energy (kJ/mol)'):
    self._out = open(file, 'w')
    self._reportInterval = reportInterval
    self._separator = separator
    self._header = header
    if header is not None:
      self._out.write('%s\n' % header)

  def __del__(self):
    self._out.close()

  def describeNextReport(self, simulation):
    steps = self._reportInterval - simulation.currentStep % self._reportInterval
    return (steps, True, False, False, False, None)

  def report(self, simulation, state):
    # Unfortunately have to hard code the energy computation - should be pretty quick, though
    # This is because it's tricky to deconvolute the energy calculation done by OpenMM into individual forces
    # First get the force information from the appropriate force object
    refz = simulation.context.getParameter('refZ')
    kval = None
    group1Inds = None
    group1Weights = None
    group2Inds = None
    group2Weights = None
    boxVecs = state.getPeriodicBoxVectors()
    print(boxVecs)
    boxZ = boxVecs[-1][-1].value_in_unit(u.nanometer)
    print(boxZ)
    for frc in simulation.system.getForces():
      if (isinstance(frc, mm.CustomCentroidBondForce)):
        kval = frc.getBondParameters(0)[1][0]
        group1Inds, group1Weights = frc.getGroupParameters(0)
        group2Inds, group2Weights = frc.getGroupParameters(1)
    pos = np.array(state.getPositions().value_in_unit(u.nanometer))
    distancez = (np.average(pos[group2Inds, 2], weights=group2Weights) 
                -np.average(pos[group1Inds, 2], weights=group1Weights))
    # Make sure to get the minimum image distance
    distancez = abs(distancez - boxZ * np.rint(distancez / boxZ))
    energy = 0.5 * kval * ((distancez - refz) ** 2)
    simtime = state.getTime().value_in_unit(u.picosecond)
    self._out.write('%g %g %g %g\n' % (simtime, refz, distancez, energy))


def getForceLabels(sysObj):
  # Assigns labels to forces in a systematic manner
  # For a given force name, returns the force index in the system object
  forceLabs = {}
  for ind in range(sysObj.getNumForces()):
    frc = sysObj.getForce(ind)
    if isinstance(frc, mm.NonbondedForce):
      forceLabs['Nonbonded force'] = ind
    elif isinstance(frc, mm.CustomNonbondedForce):
      try:
        paramName = frc.getGlobalParameterName(0)
        if paramName == 'lambdaLJ':
          forceLabs['Soft-core force'] = ind
        if paramName == 'lambdaQ':
          forceLabs['Intramolecular electrostatic force'] = ind 
      except:
        forceLabs['Intramolecular LJ force'] = ind
    elif isinstance(frc, mm.CustomCentroidBondForce):
      forceLabs['Restraint force'] = ind
    else:
      forceLabs[frc.__class__.__name__] = ind
  return forceLabs


def decompEnergy(sysObj, stateObj, labels=None, verbose=False):
  # Gives instantaneous energy decomposition without reporter
  # labels is a dictionary with force names as keys and indices as definitions
  print("\n")
  system = copy.deepcopy(sysObj)
  groupcount = 0
  groupDict = {}
  for ind in range(system.getNumForces()):
    frc = system.getForce(ind)
    # getForceInfo(frc)
    frc.setForceGroup(copy.deepcopy(groupcount))
    fname = frc.__class__.__name__
    if labels is not None:
      try:
        for aname in labels.keys():
          if labels[aname] == ind:
            fname = aname
      except:
        pass
    groupDict[groupcount] = fname
    groupcount += 1
    try:
      frc.setReciprocalSpaceForceGroup(groupcount)
      groupDict[groupcount] = fname + '_reciprocal'
      groupcount += 1
    except:
      pass
    if verbose:
      getForceInfo(frc)

  integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
  context = mm.Context(system, integrator)
  # Want to have choice of giving state or positions
  # State is more accurate, but sometimes just want to evaluate positions
  try:
    context.setState(stateObj)
  except:
    context.setPositions(stateObj) 

  print("Potential energy: %s" % (str(context.getState(getEnergy=True).getPotentialEnergy())))
  for i in range(groupcount):
    thisenergy = context.getState(getEnergy=True, groups=2 ** i).getPotentialEnergy()
    print("%s (group %i): %s" % (groupDict[i], i, str(thisenergy)))
  print("\n")

  del context, integrator


# Define a functions for modifying and querying the alchemical state
def changeLambdaState(contextObj, forceObj, atomInds, fullCharges, states, oldInd, newInd):
  """Just changes the lambda state in OpenMM, modifying the context and force objects passed in.

Inputs:
     contextObj - OpenMM context object to modify
     forceObj - OpenMM force object that defines the electrostatic interactions in the system
     atomInds - atom indices for the molecule being decoupled
     fullCharges - full, unscaled charges (need as reference!)
     states - 2xN array defining lambda values for electrostatics (1st row) and LJ interactions (2nd row)
     oldInd - old index of the lambda state
     newInd - new index to switch to
Outputs:
     Nothing - it just modifies the passed in objects.

  """
  # Make sure the state is actually changed
  if newInd == oldInd:
    return

  # Update the LJ interaction lambda
  contextObj.setParameter('lambdaLJ', states[1, newInd])
  contextObj.setParameter('lambdaQ', states[0, newInd])
 
  # Only change electrostatics if we have to since more expensive
  if states[0, newInd] != states[0, oldInd]:
    lambdaElec = states[0, newInd]
    for k, ind in enumerate(atomInds):
      # print("For atom %i (%ith atom in solute), parameters are: "%(ind, k)+str(forceObj.getParticleParameters(ind)))
      [charge, sig, eps] = forceObj.getParticleParameters(ind)
      forceObj.setParticleParameters(ind, fullCharges[k] * lambdaElec, sig, eps)
    forceObj.updateParametersInContext(contextObj)

  # For some solutes, namely acetic acid, may have additional rows in states array
  # If this is the case, also adjust this other lambda parameter (doesn't matter what it's called)
  # Note that as this is coded, only ONE additionall lambda parameter may be accommodated
  # So setting if statement up to ignore any other situations
  if states.shape[0] == 3:
     globalParams = contextObj.getParameters()
     for param in globalParams:
        if param not in ['lambdaLJ', 'lambdaQ']:
          contextObj.setParameter(param, states[2, newInd])
 
  # getForceInfo(forceObj)
  # print(contextObj.getParameter('lambda'))


def getLambdaTransitionProbs(contextObj, forceObj, atomInds, fullCharges, states, weights, biases, oldIndex, newIndices, kBT):
  """For a list of proposed transitions from the current lambda state to new lambda states, the potential
energy difference is computed and the Metropolis probability for transition is returned for all proposed
states.

Inputs:
     contextObj - OpenMM context object - WILL MODIFY, SO SHOULD BE A COPY
     forceObj - OpenMM force object defining the electrostatics - MUST be associated with contextObj, or will through error
     atomInds - atom indices for the molecule being decoupled
     fullCharges - full, unscaled charges (need as reference!)
     states - 2xN array defining lambda values for electrostatics (1st row) and LJ interactions (2nd row)
     weights - the matrix of weights for PROPOSING a move from any state i (row) to state j (column)
     biases - the values of the biasing functions to improve sampling
     oldIndex - old index of the lambda state
     newIndices - list of new indices of lambda states to switch to
     kBT - kB*T in unit quantities; necessary to compute the probabilities correctly
Outputs:
     tProbs - transition probabilities to all specified states in newIndices
     deltaU - potential energy difference for all proposed changes of state
     deltaH - difference in the full Hamiltonian, including the biasing weight
     **The convention should be that when lambda is 1.0 the system is fully-interacting**

  """
  # Create arrays to hold potential energy differences and probabilities
  deltaU = np.zeros(len(newIndices))
  tProbs = np.zeros(len(newIndices))
  deltaH = np.zeros(len(newIndices))

  # Calculate the starting potential energy
  oldU = contextObj.getState(getEnergy=True).getPotentialEnergy()

  # Keep track of what state we set the system to
  currIndex = oldIndex

  # And loop over indices we are changing to!
  for j, newIndex in enumerate(newIndices):

    changeLambdaState(contextObj, forceObj, atomInds, fullCharges, states, currIndex, newIndex)

    # Necessary to make sure we change the state correctly (i.e. from what it's currently in)
    # Also lets us know what state we ended in for the system and context
    currIndex = newIndex

    # Get new potential energy
    newU = contextObj.getState(getEnergy=True).getPotentialEnergy()
    
    # Report energy difference
    thisdU = (newU - oldU) / kBT
    deltaU[j] = thisdU

    # And Hamiltonian difference
    thisdBias = biases[newIndex] - biases[oldIndex]
    thisdH = thisdU + thisdBias
    deltaH[j] = thisdH

    # Calculate the probability using the Metropolis criterion
    wNewOld = weights[newIndex, oldIndex]
    wOldNew = weights[oldIndex, newIndex]
    # Should NEVER have the case where one weight is zero and the other is non-zero!
    # Technically could have Markov model with this property, but breaks the detailed balance acceptance criteria?
    # (i.e. if wOldNew is zero and wNewOld is non-zero, get infinity!)
    if wOldNew == 0 and wNewOld != 0:
      print("Something is terribly wrong! Have zero probability of moving from old to new state, but non-zero from new to old!")
      print("Just setting transition probability to zero.")
      tProbs[j] = 0.0
    elif wNewOld == 0 and wOldNew == 0:
      tProbs[j] = 0.0
    else:
      tProbs[j] = np.min([1.0, wNewOld * np.exp(-thisdH) / wOldNew])

  return tProbs, deltaU, deltaH, currIndex


def getLambdaTransitionProbsGibbs(contextObj, forceObj, atomInds, fullCharges, states, biases, oldIndex, kBT):
  """For a list of proposed transitions from the current lambda state to new lambda states, the potential
energy difference is computed and the Metropolis probability for transition is returned for all proposed
states.

Inputs:
     contextObj - OpenMM context object - WILL MODIFY, SO SHOULD BE A COPY
     forceObj - OpenMM force object defining the electrostatics - MUST be associated with contextObj, or will through error
     atomInds - atom indices for the molecule being decoupled
     fullCharges - full, unscaled charges (need as reference!)
     states - 2xN array defining lambda values for electrostatics (1st row) and LJ interactions (2nd row)
     weights - the matrix of weights for PROPOSING a move from any state i (row) to state j (column)
     biases - the values of the biasing functions to improve sampling
     oldIndex - old index of the lambda state
     kBT - kB*T in unit quantities; necessary to compute the probabilities correctly
Outputs:
     tAccProbs - transition probabilities to all specified states in newIndices
     tWeights - transition proposal probabilites to move to each state
     allU - potential energies at all states
     allH - full Hamiltonian at all states, iqncluding the biasing weight
     **The convention should be that when lambda is 1.0 the system is fully-interacting**

  """
  # In this scheme, need all potential energy differences from all states
  newIndices = np.arange(len(biases))

  # So start by just getting all Hamiltonian energies
  allH = np.zeros(len(biases))

  # Will also store potential energies to report to MBAR
  allU = np.zeros(len(biases))

  # Keep track of what state we set the system to
  currIndex = oldIndex

  for j, newIndex in enumerate(newIndices):
    
    changeLambdaState(contextObj, forceObj, atomInds, fullCharges, states, currIndex, newIndex)

    # Necessary to make sure we change the state correctly (i.e. from what it's currently in)
    # Also lets us know what state we ended in for the system and context
    currIndex = newIndex

    # Get new potential energy
    newU = contextObj.getState(getEnergy=True).getPotentialEnergy() / kBT
    allU[j] = newU

    # And store Hamiltonian
    allH[j] = newU + biases[newIndex]

  # print("Current potential energy values:")
  # print(allU)

  # print("Current Hamiltonian values:")
  # print(allH)

  # Now can compute all conditional probabilites for each lambda state given sampled configurations
  # But be careful to avoid overflows
  minH = np.min(allH)
  condProbs = np.exp(-(allH - minH)) / np.sum(np.exp(-(allH - minH)))

  # print("Conditional probabilities given current configuration:")
  # print(condProbs)
  # print(np.sum(condProbs))

  # Finally can compute acceptance and proposal probabilites from old to all other states
  tWeights = condProbs / (1.0 - condProbs[oldIndex])
  tWeights[oldIndex] = 0.0
  tWeights /= np.sum(tWeights)  # Not sure if I can do this or need to draw random numbers differently... CHECK
  tAccProbs = (1.0 - condProbs[oldIndex]) / (1.0 - condProbs)
  tAccProbs[(tAccProbs >= 1.0)] = 1.0

  return tAccProbs, tWeights, condProbs, allU, allH, currIndex


def getLambdaPotentialEnergies(contextObj, forceObj, atomInds, fullCharges, states, oldIndex, newIndices, kBT):
  """For a list of states, computes all potential energies of those states. 

Inputs:
     contextObj - OpenMM context object - WILL MODIFY, SO SHOULD BE A COPY
     forceObj - OpenMM force object defining the electrostatics - MUST be associated with contextObj, or will throw error
     atomInds - atom indices for the molecule being decoupled
     fullCharges - full, unscaled charges (need as reference!)
     states - 2xN array defining lambda values for electrostatics (1st row) and LJ interactions (2nd row)
     oldIndex - old index of the lambda state
     newIndices - list of new indices of lambda states to switch to
     kBT - kB*T in unit quantities; necessary to compute the probabilities correctly
Outputs:
     energiesU - potential energies for all states in newIndices
     Note that we first copy the contextObj and forceObj so we don't modify the originals
     **The convention should be that when lambda is 1.0 the system is fully-interacting**

  """
  # Create an arrays to hold potential energies
  energiesU = np.zeros(len(newIndices))

  # print("Old index: %i"%oldIndex)
  # getForceInfo(forceObj)
  # print(contextObj.getParameter('lambda'))

  # Loop over indices we are changing to!
  for j, newIndex in enumerate(newIndices):

    changeLambdaState(contextObj, forceObj, atomInds, fullCharges, states, oldIndex, newIndex)

    # print("Old index: %i"%oldIndex)
    # print("New index: %i"%newIndex)
    # getForceInfo(forceObj)
    # print(contextObj.getParameter('lambda'))

    # Get new potential energy
    energiesU[j] = contextObj.getState(getEnergy=True).getPotentialEnergy() / kBT

    # To save time, update oldIndex to newIndex (helps make changing lambda states faster)
    oldIndex = newIndex
    
  return energiesU, oldIndex

# Define the simulation procedures we will use for each type of simulation we want to perform


def doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, state=None, pos=None, vels=None, nSteps=50000):
  # Input a topology object, structure object, reference system, integrator, platform, platform properties, list of reporters,
  # and optionally state file, positions, and velocities
  # If state is specified including positions and velocities and pos and vels are not None, the 
  # positions and velocities from the provided state will be overwritten
  # Does NVT and returns a simulation state object and state file that can be used to start other simulations

  # Copy the reference system and integrator objects
  system = copy.deepcopy(systemRef)
  integrator = copy.deepcopy(integratorRef)

  # Create the simulation object for NVT simulation
  sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

  # Set the particle positions
  if pos is not None:
    sim.context.setPositions(pos)

  # Apply constraints before starting the simulation
  sim.context.applyConstraints(1.0E-08)

  # Check starting energy decomposition if want
  # decompEnergy(sim.system, sim.context.getState(getPositions=True))

  # Minimize the energy
  print("\nMinimizing energy...")
  sim.minimizeEnergy(
                     tolerance=10.0 * u.kilojoule / u.mole,  # Energy threshold below which stops
                     maxIterations=1000  # Maximum number of iterations
  )

  # Initialize velocities if not specified
  if vels is not None:
    sim.context.setVelocities(vels)
  else:
    try:
      testvel = sim.context.getState(getVelocities=True).getVelocities()
      print("Velocities included in state, starting with 1st particle: %s" % str(testvel[0]))
      # If all the velocities are zero, then set them to the temperature
      if not np.any(testvel.value_in_unit(u.nanometer / u.picosecond)):
        print("Had velocities, but they were all zero, so setting based on temperature.")
        sim.context.setVelocitiesToTemperature(temperature)
    except:
      print("Could not find velocities, setting with temperature")
      sim.context.setVelocitiesToTemperature(temperature)

  # Set up the reporter to output energies, volume, etc.
  sim.reporters.append(app.StateDataReporter(
                                             'nvt_out.txt',  # Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                             1000,  # Number of steps between writes
                                             step=True,  # Write step number
                                             time=True,  # Write simulation time
                                             potentialEnergy=True,  # Write potential energy
                                             kineticEnergy=True,  # Write kinetic energy
                                             totalEnergy=True,  # Write total energy
                                             temperature=True,  # Write temperature
                                             volume=True,  # Write volume
                                             density=False,  # Write density
                                             speed=True,  # Estimate of simulation speed
                                             separator='  '  # Default is comma, but can change if want (I like spaces)
                                            )
  )

  # Set up reporter for printing coordinates (trajectory)
  sim.reporters.append(NetCDFReporter(
                                      'nvt.nc',  # File name to write trajectory to
                                      1000,  # Number of steps between writes
                                      crds=True,  # Write coordinates
                                      vels=False,  # Write velocities
                                      frcs=False  # Write forces
                                     )
  )

  # Run NVT dynamics
  print("\nRunning NVT simulation...")
  sim.context.setTime(0.0)
  sim.step(nSteps)

  # Save simulation state if want to extend, etc.
  sim.saveState('nvtState.xml')

  # Get the final positions and velocities
  return 'nvtState.xml', sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True)


def doSimNPT(top, systemRef, integratorRef, platform, prop, temperature, scalexy=False, inBulk=False, state=None, pos=None, vels=None, nSteps=250000):
  # Input a topology object, reference system, integrator, platform, platform properties, 
  # and optionally state file, positions, or velocities
  # If state is specified including positions and velocities and pos and vels are not None, the 
  # positions and velocities from the provided state will be overwritten
  # Does NPT and returns a simulation state object that can be used to start other simulations

  # Copy the reference system and integrator objects
  system = copy.deepcopy(systemRef)
  integrator = copy.deepcopy(integratorRef)

  # For NPT, add the barostat as a force
  # If not in bulk, use anisotropic barostat
  if not inBulk:
    system.addForce(mm.MonteCarloAnisotropicBarostat((1.0, 1.0, 1.0) * u.bar,
                                                     temperature,  # Temperature should be SAME as for thermostat
                                                     scalexy,  # Set with flag for flexibility
                                                     scalexy,
                                                     True,  # Only scale in z-direction
                                                     100  # Time-steps between MC moves
                                                    )
    )
  # If in bulk, have to use isotropic barostat to avoid any weird effects with box changing dimensions
  else:
    system.addForce(mm.MonteCarloBarostat(1.0 * u.bar,
                                          temperature,
                                          100
                                         )
    )

  # Create new simulation object for NPT simulation
  sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

  # Set the particle positions
  if pos is not None:
    sim.context.setPositions(pos)

  # Apply constraints before starting the simulation
  sim.context.applyConstraints(1.0E-08)

  # Check starting energy decomposition if want
  # decompEnergy(sim.system, sim.context.getState(getPositions=True))

  # Initialize velocities if not specified
  if vels is not None:
    sim.context.setVelocities(vels)
  else:
    try:
      testvel = sim.context.getState(getVelocities=True).getVelocities()
      print("Velocities included in state, starting with 1st particle: %s" % str(testvel[0]))
      # If all the velocities are zero, then set them to the temperature
      if not np.any(testvel.value_in_unit(u.nanometer / u.picosecond)):
        print("Had velocities, but they were all zero, so setting based on temperature.")
        sim.context.setVelocitiesToTemperature(temperature)
    except:
      print("Could not find velocities, setting with temperature")
      sim.context.setVelocitiesToTemperature(temperature)

  # Set up the reporter to output energies, volume, etc.
  sim.reporters.append(app.StateDataReporter(
                                             'npt_out.txt',  # Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                             1000,  # Number of steps between writes
                                             step=True,  # Write step number
                                             time=True,  # Write simulation time
                                             potentialEnergy=True,  # Write potential energy
                                             kineticEnergy=True,  # Write kinetic energy
                                             totalEnergy=True,  # Write total energy
                                             temperature=True,  # Write temperature
                                             volume=True,  # Write volume
                                             density=False,  # Write density
                                             speed=True,  # Estimate of simulation speed
                                             separator='  '  # Default is comma, but can change if want (I like spaces)
                                            )
  )

  # Set up reporter for printing coordinates (trajectory)
  sim.reporters.append(NetCDFReporter(
                                      'npt.nc',  # File name to write trajectory to
                                      1000,  # Number of steps between writes
                                      crds=True,  # Write coordinates
                                      vels=False,  # Write velocities
                                      frcs=False  # Write forces
                                     )
  )

  # Run NPT dynamics
  print("\nRunning NPT simulation...")
  sim.context.setTime(0.0)
  sim.step(nSteps)

  # And save the final state of the simulation in case we want to extend it
  sim.saveState('nptState.xml')

  # Get the final positions and velocities
  return 'nptState.xml', sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True)


def doSimExpanded(top, systemRef, integratorRef, platform, prop, temperature, lState, lambdaStates, alchemicalAtoms, alchemicalCharges, lambdaWeights=None, scalexy=False, inBulk=False, state=None, pos=None, vels=None, nSteps=3500000, equilSteps=1000000):
  # Input a topology object, reference system, integrator, platform, platform properties, lambda states, alchemical atom indices
  # and optionally state file, positions, or velocities
  # If state is specified including positions and velocities and pos and vels are not None, the 
  # positions and velocities from the provided state will be overwritten
  # Does expanded ensemble in NPT and returns a simulation state object that can be used to start other simulations

  # Copy the reference system and integrator objects
  system = copy.deepcopy(systemRef)
  integrator = copy.deepcopy(integratorRef)

  # For NPT, add the barostat as a force
  # If not in bulk, use anisotropic barostat
  if not inBulk:
    system.addForce(mm.MonteCarloAnisotropicBarostat((1.0, 1.0, 1.0) * u.bar,
                                                     temperature,  # Temperature should be SAME as for thermostat
                                                     scalexy,  # Set with flag for flexibility
                                                     scalexy,
                                                     True,  # Only scale in z-direction
                                                     250  # Time-steps between MC moves
                                                    )
    )
  # If in bulk, have to use isotropic barostat to avoid any weird effects with box changing dimensions
  else:
    system.addForce(mm.MonteCarloBarostat(1.0 * u.bar,
                                          temperature,
                                          250
                                         )
    )

  # Create new simulation object for NPT simulation
  sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

  # Get some names for forces and the non-bonded force with PME electrostatics
  forceLabels = getForceLabels(system)
  nbForce = system.getForce(forceLabels['Nonbonded force'])

  # Set the particle positions - should also figure out how to set box information!
  if pos is not None:
    sim.context.setPositions(pos)

  # Apply constraints before starting the simulation
  sim.context.applyConstraints(1.0E-08)

  # Check starting energy decomposition if want
  # decompEnergy(sim.system, sim.context.getState(getPositions=True), labels=forceLabels)

  # Initialize velocities if not specified
  if vels is not None:
    sim.context.setVelocities(vels)
  else:
    try:
      testvel = sim.context.getState(getVelocities=True).getVelocities()
      print("Velocities included in state, starting with 1st particle: %s" % str(testvel[0]))
      # If all the velocities are zero, then set them to the temperature
      if not np.any(testvel.value_in_unit(u.nanometer / u.picosecond)):
        print("Had velocities, but they were all zero, so setting based on temperature.")
        sim.context.setVelocitiesToTemperature(temperature)
    except:
      print("Could not find velocities, setting with temperature")
      sim.context.setVelocitiesToTemperature(temperature)

  # Set up the reporter to output energies, volume, etc.
  sim.reporters.append(app.StateDataReporter(
                                             'prod_out.txt',  # Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                             1000,  # Number of steps between writes
                                             step=True,  # Write step number
                                             time=True,  # Write simulation time
                                             potentialEnergy=True,  # Write potential energy
                                             kineticEnergy=True,  # Write kinetic energy
                                             totalEnergy=True,  # Write total energy
                                             temperature=True,  # Write temperature
                                             volume=True,  # Write volume
                                             density=False,  # Write density
                                             speed=True,  # Estimate of simulation speed
                                             separator='  '  # Default is comma, but can change if want (I like spaces)
                                            )
  )

  # Set up reporter for printing coordinates (trajectory)
  sim.reporters.append(NetCDFReporter(
                                      'prod.nc',  # File name to write trajectory to
                                      500,  # Number of steps between writes
                                      crds=True,  # Write coordinates
                                      vels=False,  # Write velocities
                                      frcs=False  # Write forces
                                     )
  )

  # Ready to perform expanded ensemble dynamics
  print("\nRunning production alchemical simulation...")
  sim.context.setTime(0.0)
  
  totSteps = nSteps  # 3500000 #Total number of simulation steps
  countSteps = 0
  swapFreq = 250  # Frequency to perform MC swaps in alchemical space
  writeFreq = 2  # Number of MC cycles to perform before writing information for MBAR
  alchemicalFile = open('alchemical_output.txt', 'w')
  alchemicalFile.write('#Step  Time (ps)  LambdaState  potential energies at all states (kB*T)   pV (kB*T) \n')

  Nstates = lambdaStates.shape[1]  # Total number of states

  ####### Only with local moves (neighbor moves) and WL #######
#  #Need a matrix to define probabilities of choosing moves to different states
#  #weightsMat = np.ones((Nstates, Nstates)) / float(Nstates)
#  weightsMat = (np.eye(Nstates, k=-1) + np.eye(Nstates, k=1)) * 0.5
#  weightsMat[0,1] = 1.0
#  weightsMat[-1, -2] = 1.0

  # And a vector to hold biases on states
  stateBias = np.zeros(Nstates)
  if lambdaWeights is not None:
    stateBias[:] = lambdaWeights[:]  # Doing it this way checks the shape of lambdaWeights automatically

  # Set initial bias amount (2 kB*T should be ok at first)
  # This is only used with Wang-Landau
  biasDelta = 2.0

  # Initialize a histogram to track sampling (with Wang-Landau approach)
  histWL = np.zeros(Nstates)

  # Need to also keep track of MC swaps taken
  countMC = 0

  # Set burn-in time for fancier update scheme
  # Start with zero, switch to the time it takes to reach an 80% flat histogram the first time
  tBurn = 0

  # Set lower cut-off for maximum state bias at which point turn off bias updates
  biasCut = 5.0E-3
  doBiasUpdate = True

  # And for more efficient update, use MBAR whenever change biasDelta
  # To do this, accumulate potential energy differences between updates, resetting after histogram is flat
  # Actually, don't - it doesn't help until weights essentially converged...
  mbarUkn = np.array([[]] * Nstates).T
  mbarCount = np.zeros(Nstates)

#  print("Matrix of transition proposal weights:")
#  print(weightsMat)

  print("Starting biasing weights:")
  print(stateBias)

#  print("Starting bias delta (in kB*T):")
#  print(biasDelta)

  print("Bias cut-off to stop adjusting weights:")
  print(biasCut)

  # Also need a variable to keep track of state as we move around to check potential energies
  currState = lState

  # And for convenience, define kB*T
  kBT = u.AVOGADRO_CONSTANT_NA * u.BOLTZMANN_CONSTANT_kB * temperature

  #############################For neighbor exchange######################
#  #Get all the potential energies, transition probabilites, etc.
#  thisProbs, thisU, thisH, endState = getLambdaTransitionProbs(sim.context, nbForce, 
#                                                               alchemicalAtoms, alchemicalCharges,
#                                                               lambdaStates, weightsMat, stateBias,
#                                                               currState, range(Nstates), 
#                                                               kBT
#                                                              )
#  
#  #Randomly pick a state to switch to based on weightsMat
#  newState = np.random.choice(Nstates, size=None, replace=False, p=weightsMat[currState, :])

  ############################For Gibbs sampling##########################
  # Real difference is that we don't use a pre-defined matrix for weightsMat, but compute it
  # Get all the potential energies, transition probabilities, etc.
  (thisProbs, thisWeights, thisStateProbs,
  thisU, thisH, endState) = getLambdaTransitionProbsGibbs(sim.context, nbForce,
                                                          alchemicalAtoms, alchemicalCharges,
                                                          lambdaStates, stateBias,
                                                          currState,
                                                          kBT
                                                          )

  # And switch back to original state before we start
  changeLambdaState(sim.context, nbForce, alchemicalAtoms, alchemicalCharges, lambdaStates, endState, lState)

  # Update biases 
  ##################### Simple Wang-Landau scheme #################
#  stateBias[currState] += biasDelta

  ##################### Fancy more optimal way from Tan ###########
  # Check burn-in time to determine update strategy
  if tBurn == 0:
    thisGamma = np.min([1.0 / Nstates, 1.0 / ((countMC + 1) ** (0.6))])
  else:
    thisGamma = np.min([1.0 / Nstates, 1.0 / ((countMC + 1) - tBurn + (tBurn ** 0.6))])
  thisAddBias = thisGamma * thisStateProbs * Nstates
  thisAddBias -= thisAddBias[0]
  stateBias += thisAddBias
  # print("Added biases for this step:")
  # print(thisAddBias)

  # And add count to histogram (track in both schemes)
  histWL[currState] += 1

  # Add to potential energies to use with MBAR
  mbarUkn = np.vstack((mbarUkn, thisU))

  # Now setup, start the loop for simulation
  while countSteps < totSteps:
    
    # Start with MD
    sim.step(swapFreq)

    countSteps += swapFreq
    countMC += 1

    #############################For neighbor exchange######################
#    #Get all the potential energies, transition probabilites, etc.
#    thisProbs, thisU, thisH, endState = getLambdaTransitionProbs(sim.context, nbForce, 
#                                                                 alchemicalAtoms, alchemicalCharges,
#                                                                 lambdaStates, weightsMat, stateBias,
#                                                                 currState, range(Nstates), 
#                                                                 kBT
#                                                                )
#    
#    #Randomly pick a state to switch to based on weightsMat
#    newState = np.random.choice(Nstates, size=None, replace=False, p=weightsMat[currState, :])

    ############################For Gibbs sampling##########################
    # Real difference is that we don't use a pre-defined matrix for weightsMat, but compute it
    # Get all the potential energies, transition probabilities, etc.
    (thisProbs, thisWeights, thisStateProbs,
    thisU, thisH, endState) = getLambdaTransitionProbsGibbs(sim.context, nbForce,
                                                            alchemicalAtoms, alchemicalCharges,
                                                            lambdaStates, stateBias,
                                                            currState,
                                                            kBT
                                                            )

    # print("Proposal and acceptance probabilites for the current configuration:")
    # print(thisWeights)
    # print(np.sum(thisWeights))
    # print(thisProbs)

    # Randomly pick a state to switch to based on thisWeights
    newState = np.random.choice(Nstates, size=None, replace=False, p=thisWeights)

    # Keep track of current/old state since write information to file after we switch
    oldState = currState

    # Compare to a random number
    thisRand = np.random.random()
    if thisProbs[newState] > thisRand:
      # Remember, as far as the computer is concerned, we're actually in endState, not currState
      changeLambdaState(sim.context, nbForce, alchemicalAtoms, alchemicalCharges, lambdaStates, endState, newState)
      currState = newState
    else:
      # If rejected, switch back!
      changeLambdaState(sim.context, nbForce, alchemicalAtoms, alchemicalCharges, lambdaStates, endState, currState)

    # Update biases 
    ##################### Simple Wang-Landau scheme #################
#    if doBiasUpdate:
#      stateBias[currState] += biasDelta

    ##################### Fancy more optimal way from Tan ###########
    if doBiasUpdate:
      # Check burn-in time to determine update strategy
      if tBurn == 0:
        thisGamma = np.min([1.0 / Nstates, 1.0 / ((countMC + 1) ** (0.6))])
      else:
        thisGamma = np.min([1.0 / Nstates, 1.0 / ((countMC + 1) - tBurn + (tBurn ** 0.6))])
      thisAddBias = thisGamma * thisStateProbs * Nstates
      thisAddBias -= thisAddBias[0]
      stateBias += thisAddBias
    # print("Added biases for this step:")
    # print(thisAddBias)

    # And add count to histogram (track in both schemes)
    histWL[currState] += 1

    # Add to potential energies to use with MBAR
    if doBiasUpdate:
      mbarUkn = np.vstack((mbarUkn, thisU))

    # Decide if need to write this frame
    # Only start writing if weights converged... that way, don't have to automate script to check for convergence
    if (countMC % writeFreq) == 0:
      if not doBiasUpdate:
        # Also want current PV term in kB*T
        thisPV = ((1.0 * u.bar) * sim.context.getState().getPeriodicBoxVolume()) / (u.BOLTZMANN_CONSTANT_kB * temperature)

        # Write to file needed for MBAR 
        alchemicalFile.write("%i  %10.1f  %i  %s  %f \n" % (countSteps,
                                                          sim.context.getState().getTime().value_in_unit(u.picosecond),
                                                          oldState,
                                                          "  ".join(map(str, thisU)),
                                                          thisPV
                                                         )
                            )

      else:
        print("\nAfter %i MC swaps (%i MD steps), biases are:" % (countMC, countSteps))
        print(stateBias)
        print("Histogram counts are:")
        print(histWL)
        print("Transition probabilities were:")
        print(thisProbs)

    # Decide if histogram is flat enough that we should reduce the weight update parameter
    # Should consider directly using MBAR to compute new biases at this point...
    # Will need to keep track of potential energies, then, which shouldn't take up too much memory
    # Running MBAR may be slow, but will have to see if really speeds up sampling
    # So MBAR does provide better estimates for smaller sample sizes
    # However, this doesn't matter because fluctuations still large...
    # Turns out when update weight fluctuations become small enough that the algorithm with MBAR converges,
    # then MBAR is doing very little at that point, and before it's not helping because everything still fluctuates
    # So to help save time and memory, won't use MBAR, EXCEPT...
    # Will use it once right before stop updating biases... at this point, it should actually help
    if doBiasUpdate:
      if np.all(((abs((histWL / np.sum(histWL)) - (1.0 / Nstates)) / (1.0 / Nstates)) < 0.3)):
#      if np.min(histWL) > 0.8*np.average(histWL):
        oldBiasDelta = biasDelta
        if countMC > 100 and biasDelta < (1.0 / countMC):
          biasDelta = 1.0 / countMC 
        else:
          biasDelta *= 0.7
        thisBiasCompare = np.max(abs(thisAddBias))  # Must be absolute value in case all negative except first state
#       thisBiasCompare = biasDelta 
        print("\nHistogram probabilities are all within 30%% of flat, with maximum added bias of %e - checking stopping criteria.\n" % (thisBiasCompare))
#        print("\nHistogram is flat within 80%%, reducing bias from %e to %e and updating weights with MBAR.\n"%(oldBiasDelta, biasDelta))
        if tBurn == 0:
          tBurn = countMC
          print("Have encountered flat histogram for first time, setting burn-in time to %i." % (tBurn))
        # And use MBAR to update weights
        # Currently think it's better to use ALL of the previous data, not just since the last time we ran MBAR
        # This is because small errors in sampling create small errors in weights, which create subtle bias in MBAR
        # We sort of fix this by including ALL sampled data because older sampling biases are cancelled by more recent biases
        mbarCount += histWL
        # thisMbarObj = mbar.MBAR(mbarUkn.T, mbarCount)
        # thisdG, thisdGerr, thisTheta = thisMbarObj.getFreeEnergyDifferences()
        # stateBias = -thisdG[0]
        histWL[:] = 0.0
        # Finally, check if should stop updating weights
        # Moved to preferring a fixed amount of time to equilibrate weights... less variability this way
#        if thisBiasCompare < biasCut:
#          doBiasUpdate = False
#          print("Maximum added bias of %e below threshold of %e - stopping weight updates."%(thisBiasCompare, biasCut))
#          thisMbarObj = mbar.MBAR(mbarUkn.T, mbarCount)
#          thisdG, thisdGerr, thisTheta = thisMbarObj.getFreeEnergyDifferences()
#          stateBias = -thisdG[0]
#          print("Final bias to use moving forward (computed with MBAR):")
#          print(stateBias)
#          if thisBiasCompare == 0.0:
#            print("Had max added bias of zero, so checking this out...")
#            print("State probabilities:")
#            print(thisStateProbs)
#            print("Gamma:")
#            print(thisGamma)
#            print("Added biases:")
#            print(thisAddBias)
      # In method of Tan, histogram counts can be off, even though have converged more quickly and accurately to biases
      # At some point, just use our best estimate to the weights and make sure we get some usable samples
      # Typically, convergence takes 1-2 ns, so using 2 ns cutoff as default - can change if want, though
      elif countSteps >= equilSteps:
        thisBiasCompare = np.max(abs(thisAddBias))  # Must be absolute value in case all negative except first state
        mbarCount += histWL
        histWL[:] = 0.0
        doBiasUpdate = False
        print("Have not reach convergence criteria (flat histogram AND minimum added bias), but stopping weight updates.")
        print("Maximum added bias of %e - compare to cutoff threshold of %e." % (thisBiasCompare, biasCut))
        thisMbarObj = mbar.MBAR(mbarUkn.T, mbarCount)
        thisdG, thisdGerr, thisTheta = thisMbarObj.getFreeEnergyDifferences()
        stateBias = -thisdG[0]
        print("Final bias to use moving forward (computed with MBAR):")
        print(stateBias)

  # Clean up
  alchemicalFile.close()

  # And save the final state of the simulation in case we want to extend it
  sim.saveState('prodState.xml')

  # Get the final positions and velocities
  return 'prodState.xml', sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True), stateBias


def doSimPull(top, systemRef, integratorRef, platform, prop, temperature, state=None, pos=None, vels=None):
  # Input a topology object, structure object, reference system, integrator, platform, platform properties, list of reporters,
  # and optionally state file, positions, and velocities
  # If state is specified including positions and velocities and pos and vels are not None, the 
  # positions and velocities from the provided state will be overwritten
  # Does NVT and returns a simulation state object and state file that can be used to start other simulations
  # Uses CustomCentroidBondForce to apply harmonic restraint on solute and pull towards surface over time

  # Copy the reference system and integrator objects
  system = copy.deepcopy(systemRef)
  integrator = copy.deepcopy(integratorRef)

  # Create the simulation object for NVT simulation
  sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

  # Set the particle positions
  if pos is not None:
    sim.context.setPositions(pos)

  # Apply constraints before starting the simulation
  sim.context.applyConstraints(1.0E-08)

  # Check starting energy decomposition if want
  # decompEnergy(sim.system, sim.context.getState(getPositions=True))

  # Initialize velocities if not specified
  if vels is not None:
    sim.context.setVelocities(vels)
  else:
    try:
      testvel = sim.context.getState(getVelocities=True).getVelocities()
      print("Velocities included in state, starting with 1st particle: %s" % str(testvel[0]))
      # If all the velocities are zero, then set them to the temperature
      if not np.any(testvel.value_in_unit(u.nanometer / u.picosecond)):
        print("Had velocities, but they were all zero, so setting based on temperature.")
        sim.context.setVelocitiesToTemperature(temperature)
    except:
      print("Could not find velocities, setting with temperature")
      sim.context.setVelocitiesToTemperature(temperature)

  # Set up the reporter to output energies, volume, etc.
  sim.reporters.append(app.StateDataReporter(
                                             'pull_out.txt',  # Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                             1000,  # Number of steps between writes
                                             step=True,  # Write step number
                                             time=True,  # Write simulation time
                                             potentialEnergy=True,  # Write potential energy
                                             kineticEnergy=True,  # Write kinetic energy
                                             totalEnergy=True,  # Write total energy
                                             temperature=True,  # Write temperature
                                             volume=True,  # Write volume
                                             density=False,  # Write density
                                             speed=True,  # Estimate of simulation speed
                                             separator='  '  # Default is comma, but can change if want (I like spaces)
                                            )
  )

  # Set up reporter for printing coordinates (trajectory)
  sim.reporters.append(NetCDFReporter(
                                      'pull.nc',  # File name to write trajectory to
                                      1000,  # Number of steps between writes
                                      crds=True,  # Write coordinates
                                      vels=False,  # Write velocities
                                      frcs=False  # Write forces
                                     )
  )

  # Add a reporter for the restraining potential - also gives solute centroid over time
  sim.reporters.append(RestraintReporter(
                                         'pull_restraint.txt',  # File name with restraint info
                                         1000,  # Number of steps between writes
                                        )
  )

  # Run non-equilibrium pulling dynamics
  print("\nRunning pulling simulation...")
  sim.context.setTime(0.0)

  # Set total simulation time
  # Will be 180 ps, which should pull over 1.8 nm
  totSteps = 90000
  
  # Define pulling rate and frequency to adjust the reference distance
  pullRate = 0.01  # nm/ps
  shiftFreq = 10  # MD steps, so will shift every 20 fs
  timeStep = sim.integrator.getStepSize().value_in_unit(u.picosecond)

  # Need to know the starting value of refZ
  initRef = None
  for frc in sim.system.getForces():
    if (isinstance(frc, mm.CustomCentroidBondForce)):
      initRef = frc.getGlobalParameterDefaultValue(0)

  countSteps = 0

  while countSteps < totSteps:
    
    countSteps += shiftFreq
    
    sim.step(shiftFreq)

    # Change the reference distance to be closer to the surface
    sim.context.setParameter('refZ', initRef - pullRate * countSteps * timeStep)

  # Save simulation state if want to extend, etc.
  sim.saveState('pullState.xml')

  # Get the final positions and velocities
  return 'pullState.xml', sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True)


def doSimUmbrella(top, systemRef, integratorRef, platform, prop, temperature, state=None, pos=None, vels=None, nSteps=5000000):
  # Input a topology object, reference system, integrator, platform, platform properties, 
  # and optionally state file, positions, or velocities
  # If state is specified including positions and velocities and pos and vels are not None, the 
  # positions and velocities from the provided state will be overwritten
  # Does NPT and returns a simulation state object that can be used to start other simulations

  # Copy the reference system and integrator objects
  system = copy.deepcopy(systemRef)
  integrator = copy.deepcopy(integratorRef)

  # For NPT, add the barostat as a force
  system.addForce(mm.MonteCarloAnisotropicBarostat((1.0, 1.0, 1.0) * u.bar,
                                                   temperature,  # Temperature should be SAME as for thermostat
                                                   False,
                                                   False,
                                                   True,  # Only scale in z-direction
                                                   250  # Time-steps between MC moves
                                                  )
  )

  # Create new simulation object for NPT simulation
  sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

  # Set the particle positions
  if pos is not None:
    sim.context.setPositions(pos)

  # Apply constraints before starting the simulation
  sim.context.applyConstraints(1.0E-08)

  # Check starting energy decomposition if want
  # decompEnergy(sim.system, sim.context.getState(getPositions=True))

  # Initialize velocities if not specified
  if vels is not None:
    sim.context.setVelocities(vels)
  else:
    try:
      testvel = sim.context.getState(getVelocities=True).getVelocities()
      print("Velocities included in state, starting with 1st particle: %s" % str(testvel[0]))
      # If all the velocities are zero, then set them to the temperature
      if not np.any(testvel.value_in_unit(u.nanometer / u.picosecond)):
        print("Had velocities, but they were all zero, so setting based on temperature.")
        sim.context.setVelocitiesToTemperature(temperature)
    except:
      print("Could not find velocities, setting with temperature")
      sim.context.setVelocitiesToTemperature(temperature)

  # Set up the reporter to output energies, volume, etc.
  sim.reporters.append(app.StateDataReporter(
                                             'prod_out.txt',  # Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                             500,  # Number of steps between writes
                                             step=True,  # Write step number
                                             time=True,  # Write simulation time
                                             potentialEnergy=True,  # Write potential energy
                                             kineticEnergy=True,  # Write kinetic energy
                                             totalEnergy=True,  # Write total energy
                                             temperature=True,  # Write temperature
                                             volume=True,  # Write volume
                                             density=False,  # Write density
                                             speed=True,  # Estimate of simulation speed
                                             separator='  '  # Default is comma, but can change if want (I like spaces)
                                            )
  )

  # Set up reporter for printing coordinates (trajectory)
  sim.reporters.append(NetCDFReporter(
                                      'prod.nc',  # File name to write trajectory to
                                      500,  # Number of steps between writes
                                      crds=True,  # Write coordinates
                                      vels=False,  # Write velocities
                                      frcs=False  # Write forces
                                     )
  )

  # Add a reporter for the restraining potential - also gives solute centroid over time
  sim.reporters.append(RestraintReporter(
                                         'prod_restraint.txt',  # File name with restraint info
                                         500,  # Number of steps between writes
                                        )
  )

  # Run NPT dynamics
  print("\nRunning NPT umbrella production simulation...")
  sim.context.setTime(0.0)
  sim.step(nSteps)

  # And save the final state of the simulation in case we want to extend it
  sim.saveState('prodState.xml')

  # Get the final positions and velocities
  return 'prodState.xml', sim.context.getState(getPositions=True, getVelocities=True, getParameters=True, enforcePeriodicBox=True)

