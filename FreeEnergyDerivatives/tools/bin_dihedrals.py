'''
Code to uniformly sample configurations from the distribution of a given dihedral over the course of an MD trajectory
'''

from netCDF4 import Dataset
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-netcdf', type=str, default=None)
parser.add_argument('-indexes', type=int, nargs='+', default=None)


def calc_dihedral(coordinates):
   
    p0 = coordinates[0]
    p1 = coordinates[1]
    p2 = coordinates[2]
    p3 = coordinates[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


args = parser.parse_args()

ncin = Dataset(args.netcdf, 'r', format='NETCDF4')

atom_indexes = np.array(args.indexes)

coordinates = ncin.variables['coordinates'][:]

dihedrals = np.zeros(coordinates.shape[0])

for i in coordinates.shape[0]:
    dihedral_coords = coordinates[i][atom_indexes, :]
    
    dihedral = calc_dihedral(dihedral_coords)
    
    dihedrals[i] = dihedral
    
import matplotlib.pyplot as plt
plt.hist(dihedrals, bins=50)
plt.show()
    
