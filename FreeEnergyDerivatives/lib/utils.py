import numpy as np
import netCDF4 as nc


def read_xyz(file_path):
    fobj = open(file_path, 'r')
    lines = fobj.readlines()
    fobj.close()

    natoms = np.int(lines[0].rstrip())
    
    elements = []
    coordinates = []
    
    for i in range(natoms):
        
        idx = 2 + i
        
        data = lines[idx].rstrip().split()
        
        element = data[0]
        
        position = np.array([np.float(data[i]) for i in [1, 2, 3]])
        
        elements.append(element)
        coordinates.append(position)
        
    elements = np.array(elements)
    coordinates = np.array(coordinates)
    
    return elements, coordinates


def write_xyz(file_path, elements, coordinates, comment=''):
    
    fobj = open(file_path, 'w')
    
    fobj.write(str(len(coordinates)) + '\n')
    fobj.write(comment + '\n')
    
    for i in range(len(coordinates)):
        fobj.write(elements[i] + " " + " ".join([str(pos) for pos in coordinates[i]]) + '\n')
        
    fobj.close()

    
def collect_solute_indexes(topology):
    soluteIndices = []
    for res in topology.residues():
        resname = res.name.upper()
        if (resname != 'HOH' and resname != 'WAT'and resname != 'CL'and resname != 'NA'):
            for atom in res.atoms():
                soluteIndices.append(atom.index)
    return soluteIndices


def strip_netcdf(incdf, outcdf, atom_indexes):
    with nc.Dataset("orig.nc") as src, nc.Dataset("filtered.nc", "w") as dst:
        # copy attributes
        for name in src.ncattrs():
            dst.setncattr(name, src.getncattr(name))
        # copy dimensions
        for name, dimension in src.dimensions.iteritems():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited else None))
        # copy all file data except for the excluded
        for name, variable in src.variables.iteritems():
            if (name == "coordinates"):
                pass
            else:
                x = dst.createVariable(name, variable.datatype, variable.dimension)

        
def display_netcdf(incdf):
    with nc.Dataset(incdf) as src:
        # copy attributes
        print ("--ATTRIBUTES--")
        for name in src.ncattrs():
            print (name, src.getncattr(name))
        # copy dimensions
        print ("--DIMENSIONS--")
        for name, dimension in src.dimensions.iteritems():
            print (name, (len(dimension) if not dimension.isunlimited else None))
            
        print ("--VARIABLES--")
        
        for name, variable in src.variables.iteritems():
            print (name, variable.datatype, variable.dimension)
 
