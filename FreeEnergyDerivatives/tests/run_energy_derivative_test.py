from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from simtk.openmm import app
from simtk.openmm.app import PDBFile, Modeller, PDBFile
from mdtraj.reporters import NetCDFReporter   
from openmmtools import alchemy
import numpy as np
from time import time

from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator
from openmmtools.forces import find_forces
from openmmtools.constants import ONE_4PI_EPS0
import copy
from openmmtools import forcefactories
from openmmtools import forces

from openmmtools.alchemy import  *

from lib import solvation_potentials as sp
from openmmtools.testsystems import TestSystem


class CustomSystem(TestSystem):

    def __init__(self, mass=39.9 * unit.amu, sigma=3.350 * unit.angstrom, epsilon=10.0 * unit.kilocalories_per_mole, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Store parameters
        self.mass = mass
        self.sigma = sigma
        self.epsilon = epsilon

        charge = 0.3 * unit.elementary_charge

        system = openmm.System()

        force = openmm.NonbondedForce()
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        # Create positions.
        positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstrom)
        # Move the second particle along the x axis to be at the potential minimum.
        positions[1, 0] = 2.0 ** (1.0 / 6.0) * sigma

        system.addParticle(mass)
        force.addParticle(charge, sigma, epsilon)

        system.addParticle(mass)
        force.addParticle(charge, sigma, epsilon)

        system.addForce(force)

        self.system, self.positions = system, positions

        self.ligand_indices = [0]
        self.receptor_indices = [1]

        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
        self.topology = topology


test = CustomSystem()

system, positions, topology = test.system, test.positions, test.topology

alchemical_system = sp.create_alchemical_system(system, solute_indicies=[0, 1], softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=True)
integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

context = Context(alchemical_system, integrator)

positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstrom)

for distance in np.linspace(2.5, 5.0, 10):
    positions[0, 0] = distance * unit.angstroms
    
    context.setPositions(positions)
    
    state = context.getState(getEnergy, getEnergyParameterDerivatives=True, group=set([0]))
    
    state = simulation.context.getState(getEnergy=True, getParameterDerivatives=True)

    energy_derivs = state.getEnergyParameterDerivatives()
    
    state_deriv_electrostatics = simulation.context.getState(getEnergy=True, groups=set([1]))
    state_deriv_sterics = simulation.context.getState(getEnergy=True, groups=set([2]))
            
    energy_deriv_electrostatics = energy_derivs['lambda_electrostatics']
    energy_deriv_sterics = energy_derivs['lambda_sterics']
    
    print ("Electrostatics Diff:", energy_deriv_electrostatics, state_deriv_electrostatics.getPotentialEnergy()._value)
    print ("Sterics Diff:", energy_deriv_sterics, state_deriv_sterics.getPotentialEnergy()._value)
    
