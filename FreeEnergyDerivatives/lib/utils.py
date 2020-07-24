import numpy as np
import netCDF4 as nc

# TODO could maybe turn xyz writing/reading to dictionary writing/reading


def read_xyz(file_path):
    
    fobj = open(file_path, 'r')
    lines = fobj.readlines()
    fobj.close()

    natoms = np.int(lines[0].rstrip())
    
    elements = []
    coordinates = []
    forces = []
    
    comment = lines[1]
    
    for i in range(natoms):
        
        idx = 2 + i
        
        data = lines[idx].rstrip().split()
        
        element = data[0]
        
        position = np.array([np.float(data[i]) for i in [1, 2, 3]])
        
        if (len(data) > 4):  # assume we have force info as well
            force = np.array([np.float(data[i]) for i in [4, 5, 6]])
            forces.append(force)
            
        elements.append(element)
        coordinates.append(position)
        
    elements = np.array(elements)
    coordinates = np.array(coordinates)
    
    if (len(forces) > 0):
        forces = np.array(forces)
        return elements, coordinates, forces, comment
    else:
        return elements, coordinates, comment


def write_xyz(file_path, elements, coordinates, forces=None, comment=''):
    
    fobj = open(file_path, 'w')
    
    fobj.write(str(len(coordinates)) + '\n')
    fobj.write(comment + '\n')
    
    for i in range(len(coordinates)):
        fobj.write(elements[i] + " " + " ".join([str(pos) for pos in coordinates[i]]))
        
        if (forces is not None):
            fobj.write(" " + " ".join([str(pos) for pos in forces[i]]))
            
        fobj.write("\n")
        
    fobj.close()

    
def collect_solute_indexes(topology, solvent=['WAT', 'HOH'], pos_ions=['NA', 'K', 'Li'], neg_ions=['F', 'CL', 'BR', 'I']):
    soluteIndices = []
    
    for res in topology.residues():
        resname = res.name.upper()
        if (resname not in solvent and resname not in neg_ions and resname not in pos_ions):
            for atom in res.atoms():
                soluteIndices.append(atom.index)
    return soluteIndices


def slice_netcdf(incdf, outcdf, atom_indexes, centre=True):
    '''NETCDF3_64BIT_OFFSET is the amber standard, had trouble getting the default format to work with vmd...'''
    with nc.Dataset(incdf, "r") as src, nc.Dataset(outcdf, "w", format='NETCDF3_64BIT_OFFSET') as dst:
        # copy attributes
        for name in src.ncattrs():
            dst.setncattr(name, src.getncattr(name))
            
        # copy dimensions except for atom
        for name, dimension in src.dimensions.items():
            if (name == "atom"):
                dst.createDimension(name, size=(len(atom_indexes) if not dimension.isunlimited() else None))
            else:
                dst.createDimension(name, size=(len(dimension) if not dimension.isunlimited() else None))
                
        # copy all file data except for coordinates
        for name, variable in src.variables.items():
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            if name == "coordinates":
                min = np.min(src.variables[name][:, atom_indexes, : ], axis=1)
                if (centre):
                    dst.variables[name][:] = src.variables[name][:, atom_indexes, : ] - min[:, np.newaxis, :]
                else:
                    dst.variables[name][:] = src.variables[name][:, atom_indexes, : ]
            else:
                dst.variables[name][:] = src.variables[name][:]
            
            # copy varaible attributes
            for attrname in variable.ncattrs():
                dst.variables[name].setncattr(attrname, variable.getncattr(attrname))
            
        
def display_netcdf(incdf):
    with nc.Dataset(incdf) as src:
        # copy attributes
        print ("--ATTRIBUTES--")
        for name in src.ncattrs():
            print (name, src.getncattr(name))
        # copy dimensions
        print ("--DIMENSIONS--")
        for name, dimension in src.dimensions.items():
            print (name, dimension, dimension.isunlimited())
            
        print ("--VARIABLES--")
        for name, variable in src.variables.items():
            print (name, variable)
            print (">> Variable attributes")
            for name2 in variable.ncattrs():
                print (">>", name2, variable.getncattr(name2))
                
            print ("--")

            
def merge_netcdf(innetcdfs, outnetcdf):
    
    dst = nc.Dataset(outnetcdf, "w", format='NETCDF3_64BIT_OFFSET')
    
    with nc.Dataset(innetcdfs[0], "r") as src:
        # copy attributes
        for name in src.ncattrs():
            dst.setncattr(name, src.getncattr(name))
            
        # copy dimensions except for atom
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, size=(len(dimension) if not dimension.isunlimited() else None))
                
        for name, variable in src.variables.items():
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            
            # copy varaible attributes
            for attrname in variable.ncattrs():
                dst.variables[name].setncattr(attrname, variable.getncattr(attrname))
                
    for netcdf in innetcdfs:
        with nc.Dataset(innetcdfs[0], "r") as src:
            # copy all file data except for coordinates
            for name, variable in src.variables.items():
                if ("frame" in variable.dimensions):
                    dst.variables[name][:] = np.concatenate((dst.variables[name][:], src.variables[name][:]))
