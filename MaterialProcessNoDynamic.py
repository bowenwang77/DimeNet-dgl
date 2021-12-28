import os
import numpy as np
from numpy.core import numeric
import scipy.sparse as sp
import torch
import dgl

from tqdm import trange
from dgl.data import QM9Dataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.convert import graph as dgl_graph
import pandas

num_rows=3013
# doping_file=pandas.read_csv('Material.csv',nrows=num_rows)
doping_file=pandas.read_csv('dataset/csv/WithDynamic.csv',nrows=num_rows)
lattice=doping_file.loc[:,'lattice0':'lattice8'].to_numpy()
# Doping_pos=doping_file.loc[:,'dopx':'dopz'].to_numpy().reshape(num_rows,1,3)
# C_pos=doping_file.loc[:,'Cx':'Cz'].to_numpy().reshape(num_rows,1,3)
# O_pos=doping_file.loc[:,'Ox':'Oz'].to_numpy().reshape(num_rows,1,3)
# Cu_pos=doping_file.loc[:,'Cu0x':'Cu70z'].to_numpy().reshape(num_rows,71,3)
Pos=doping_file.loc[:,'Ag0x':'Cu70z'].to_numpy().reshape(num_rows,74,3)

Energy=doping_file.loc[:,'E'].to_numpy().reshape(num_rows,1)
Doping_atomic_num=doping_file.loc[:,'en'].to_numpy().reshape(num_rows,1)

##compute the cartisan coordinate
# Doping_cpos = Doping_pos@lattice
# Cu_pos=Cu_pos@lattice
# O_cpos=O_pos@lattice
# C_cpos=C_pos@lattice
Cartisan_pos=Pos@lattice.reshape(num_rows,3,3)

# Cartisan_pos[7][~np.isnan(Cartisan_pos[7][:,0])]

##Final data

##Material_dataset loop: [Carbon, Oxygen,Doping element,  Cu]


Material_dataset={'N':[],'Z':[],'R':[],'Energy':[],'lattice':[]}
for i in range(num_rows):
    if (Energy[i].item()>-5) & (Energy[i].item()<1):
        Num_atom=(~np.isnan(Pos[i][:,0])).sum()
        
        Material_dataset['N'].append(Num_atom)#num of atom in each molecule
        Material_dataset['R'].append(Pos[i][:Num_atom])#3D coordinate
        Material_dataset['Z'].append(Doping_atomic_num[i].item())
        Material_dataset['Z'].append(6)#Atomic number
        Material_dataset['Z'].append(8)

        Material_dataset['Z'].extend([29]*(Num_atom-3))
        Material_dataset['Energy'].append(Energy[i].item())
        Material_dataset['lattice'].append(lattice[i])

Material_dataset['N']=np.asarray(Material_dataset['N'])
Material_dataset['R']=np.concatenate(Material_dataset['R'],axis=0)
Material_dataset['Z']=np.asarray(Material_dataset['Z'])
Material_dataset['Energy']=np.asarray(Material_dataset['Energy'])
Material_dataset['lattice']=np.asarray(Material_dataset['lattice'])

# np.savez('/root/.dgl/MaterialNoDynamic.npz',
#     N=Material_dataset['N'],R=Material_dataset['R'],Z=Material_dataset['Z'],Energy=Material_dataset['Energy'],lattice=Material_dataset['lattice'])
np.savez('dataset/npz/MaterialWithDynamic.npz',
    N=Material_dataset['N'],R=Material_dataset['R'],Z=Material_dataset['Z'],Energy=Material_dataset['Energy'],lattice=Material_dataset['lattice'])

# Material_dataset['R']=np.concatenate(Material_dataset['R'],axis=0)
# npz_path='/root/.dgl/qm9_eV.npz'
# data_dict = np.load(npz_path, allow_pickle=True)

print()