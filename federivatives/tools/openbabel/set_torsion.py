import openbabel as ob
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', type=str)
parser.add_argument('-o', type=str)
parser.add_argument('-dihedral_atoms', type=int, nargs='+', help='(N,4) dimensional list containing dihedral indexes. Note: OpenBabel indexes start from 1')
parser.add_argument('-angles', type=float, nargs='+', help='(N) array containing dihedral angles in radians')

args = parser.parse_args()

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats(args.i.split('.')[1], args.o.split('.')[1])

mol = ob.OBMol()

obConversion.ReadFile(mol, args.i)

dihedral_atoms = np.array(args.dihedral_atoms, dtype=np.int)
dihedral_atoms = dihedral_atoms.reshape((np.int(len(dihedral_atoms) / 4), 4))

for i in range(dihedral_atoms.shape[0]):
    
    iw, ix, iy, iz = dihedral_atoms[i]
    
    angle_rad = args.angles[i]

    mol.SetTorsion(mol.GetAtom(int(iw)), mol.GetAtom(int(ix)), mol.GetAtom(int(iy)), mol.GetAtom(int(iz)), angle_rad)

obConversion.WriteFile(mol, args.o)
