'''
Code to uniformly sample configurations from the distribution of a given dihedral over the course of an MD trajectory
'''

from netCDF4 import Dataset
import argparse
import numpy as np
from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('-netcdf', type=str, default=None)
parser.add_argument('-indexes', type=int, nargs='+', default=None)
parser.add_argument('-nsamples', type=int, default=None)
parser.add_argument('-xyz', type=str, default=None)


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

dihedrals = np.array([calc_dihedral(coordinates[i][atom_indexes, :]) for i in range(coordinates.shape[0])])

sort_indexes = np.argsort(dihedrals)

sorted_coordinates = coordinates[sort_indexes, :, :]
sorted_dihedrals = dihedrals[sort_indexes]
    
# bin_edges has length nbins + 1
hist, bin_edges = np.histogram(sorted_dihedrals, bins=50)

# get the bin index for each dihedral
bin_indexes = np.digitize(sorted_dihedrals, bin_edges)

# import matplotlib.pyplot as plt

# plt.hist(sorted_dihedrals, bins=50)

# plt.show()

print (bin_indexes)
 
if args.nsamples != None:
     
    if (args.nsamples < 51):
        print ("args.nsamples must be > nbins")
        exit()
 
    sample_counter = 0
    
    samples_per_bin = np.int(args.nsamples / 50)
     
    for i in range(50):
         
        indexes = np.where(bin_indexes == i + 1)[0]
        
        chosen_indexes = np.random.choice(indexes, size=samples_per_bin)
        
        for j in range(len(chosen_indexes)):
            
            print ("selecting index %i from bin %i" % (chosen_indexes[j], i))
            
            coordinates = sorted_coordinates[chosen_indexes[j]]
            
            if (args.xyz != None):  # interpretting this as wanting to output samples to disk
                elements, _ = utils.read_xyz(args.xyz)
                utils.write_xyz('sample_' + str(sample_counter) + '.xyz', elements, coordinates)
                
            sample_counter += 1
            
    print ("picked %i configuration samples" % sample_counter)

