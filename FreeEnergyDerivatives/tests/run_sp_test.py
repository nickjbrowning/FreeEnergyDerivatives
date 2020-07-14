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
        
        force.setForceGroup(0)
   
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

system = sp.create_alchemical_system(system, [0], softcore_beta=0.0, softcore_m=1.0, compute_solvation_response=True)

integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)

context = Context(system, integrator)

for distance in np.linspace(3.5, 5.0, 10):
    positions[1, 0] = distance * unit.angstroms
    
    context.setPositions(positions)
    
    state = context.getState(getEnergy=True, getParameterDerivatives=True, groups=set([0]))

    energy_derivs = state.getEnergyParameterDerivatives()
    
    print ("P.E :", state.getPotentialEnergy())
    
    state = context.getState(getEnergy=True, groups=set([1]))
    print ("electrostatic dVdl", energy_derivs['lambda_electrostatics'], state.getPotentialEnergy())
    
    state = context.getState(getEnergy=True, groups=set([2]))
    
    print ("steric dV/dl :", energy_derivs['lambda_sterics'], state.getPotentialEnergy())
    
