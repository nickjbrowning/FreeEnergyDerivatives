import numpy as np


def read_xyz(file_path):
    fobj = open(file_path, 'r')
    lines = fobj.readlines()
    fobj.close()

    natoms = np.int(lines[0].rstrip())
    
    elements = []
    coordinates = []
    
    for i in range(natoms):
        
        idx = 2 + i
        
        data = lines[idx].rstrip().split(" ")
        
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
        
