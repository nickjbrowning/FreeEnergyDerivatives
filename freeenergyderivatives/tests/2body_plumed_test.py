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

from freeenergyderivatives.lib import solvation_potentials as sp
from freeenergyderivatives.lib import utils

from openmmtools.testsystems import TestSystem, WaterBox

from openmmplumed import *

platform = openmm.Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')


def test_diatomic_system():
    
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
            force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            
            # Create positions.
            positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstrom)
            
            positions[1, 0] = 4.5 * unit.angstrom
            
            system.addParticle(mass)
            force.addParticle(charge, sigma, epsilon)
            
            system.addParticle(mass)
            force.addParticle(charge, sigma, epsilon)

            system.addForce(force)
            
            script = """
            d: DISTANCE ATOMS=1,2
            METAD ARG=d SIGMA=0.2 HEIGHT=0.3 PACE=500"""
            system.addForce(PlumedForce(script))
    
            self.system, self.positions = system, positions

            topology = app.Topology()
            element = app.Element.getBySymbol('Ar')
            chain = topology.addChain()
            
            for i in range(len(positions)):
                residue = topology.addResidue('Ar', chain)
                topology.addAtom('Ar', element, residue)
          
            self.topology = topology
            
    test = CustomSystem()

    system, positions, topology = test.system, test.positions, test.topology

    integrator = LangevinIntegrator(298.15 * unit.kelvin, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    
    context = Context(new_system, integrator, platform)

    context.setPositions(positions)
    
    sp.decompose_energy(context, new_system)


if __name__ == "__main__":
    print ("Diatomic System")
    test_diatomic_system()
