import os
import numpy as np
from numpy.core import numeric
import scipy.sparse as sp
import torch
import dgl
import pdb

from tqdm import trange
from dgl.data import QM9Dataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.convert import graph as dgl_graph
import pandas

Correct_periodic_coordinate=True

file_name="NoDynamic_Clean"
relax_file_name = "WithDynamic_Clean"
doping_file = pandas.read_csv('dataset/csv_complete/'+file_name+'.csv')
num_rows = doping_file.shape[0]
lattice=doping_file.loc[:,'lattice0':'lattice8'].to_numpy()
Pos=doping_file.loc[:,'B0x':'Cu70z'].to_numpy().reshape(num_rows,74,3)

Energy=doping_file.loc[:,'E'].to_numpy().reshape(num_rows,1)
Doping_atomic_num=doping_file.loc[:,'en'].to_numpy().reshape(num_rows,1)

##compute the cartisan coordinate and fractional coordinate
Cartisan_pos = Pos
doping_file_relax = pandas.read_csv('dataset/csv_complete/'+relax_file_name+'.csv')
relax_num_rows = doping_file_relax.shape[0]
Cartisan_pos_relax = doping_file_relax.loc[:,'B0x':'Cu70z'].to_numpy().reshape(num_rows,74,3)
lattice_inv = np.linalg.inv(lattice.reshape(num_rows,3,3))
Frac_pos = Cartisan_pos@lattice_inv
Frac_pos_relax = Cartisan_pos_relax@lattice_inv
# Cartisan_pos=Pos@lattice.reshape(num_rows,3,3)


##Final data

##dataset loop: [Carbon, Oxygen,Doping element,  Cu]

# pdb.set_trace()
dataset={'N':[],'Z':[],'R':[],'R_relax':[],'Frac_R':[],'Frac_R_relax':[],'Energy':[],'lattice':[]}
for i in range(num_rows):
    # pdb.set_trace()
    # if (Energy[i].item()>-5) & (Energy[i].item()<1):
    if True:
        Num_atom=(~np.isnan(Cartisan_pos[i][:,0])).sum()
        
        dataset['N'].append(Num_atom)#num of atom in each molecule
        dataset['R'].append(Cartisan_pos[i][:Num_atom])#3D coordinate

        dataset['Frac_R'].append(Frac_pos[i][:Num_atom])
        dataset['Frac_R_relax'].append(Frac_pos_relax[i][:Num_atom])
        dataset['Z'].append(Doping_atomic_num[i].item())
        dataset['Z'].append(6)#Atomic number
        dataset['Z'].append(8)
        # dataset['Z'].append(Doping_atomic_num[i].item())
        dataset['Z'].extend([29]*(Num_atom-3))
        dataset['Energy'].append(Energy[i].item())
        dataset['lattice'].append(lattice[i])
        # pdb.set_trace()
        if Correct_periodic_coordinate:
            print(len(dataset['Frac_R_relax']), 'id ', i)
            Drift=dataset['Frac_R_relax'][i]-dataset['Frac_R'][i]
            rowid, colid = np.where(np.abs(Drift)>0.8)
            for ids in range(rowid.shape[0]):
                if dataset['Frac_R_relax'][i][rowid[ids],colid[ids]]>0.5:
                    dataset['Frac_R_relax'][i][rowid[ids],colid[ids]]-=1
                else:
                    dataset['Frac_R_relax'][i][rowid[ids],colid[ids]]+=1
            dataset['R_relax'].append(dataset['Frac_R_relax'][i]@(dataset['lattice'][i].reshape(3,3)))
        else:
            dataset['R_relax'].append(Cartisan_pos_relax[i][:Num_atom])


# pdb.set_trace()
dataset['N']=np.asarray(dataset['N'])
dataset['R']=np.concatenate(dataset['R'],axis=0)
dataset['R_relax']=np.concatenate(dataset['R_relax'],axis=0)
dataset['Frac_R']=np.concatenate(dataset['Frac_R'],axis=0)
dataset['Frac_R_relax']=np.concatenate(dataset['Frac_R_relax'],axis=0)

dataset['Z']=np.asarray(dataset['Z'])
dataset['Energy']=np.asarray(dataset['Energy'])
dataset['lattice']=np.asarray(dataset['lattice'])
# pdb.set_trace()

np.savez('dataset/npz_correct/'+file_name+'.npz',
    N=dataset['N'],R=dataset['R'],R_relax=dataset['R_relax'],Frac_R=dataset['Frac_R'],Frac_R_relax=dataset['Frac_R_relax'],Z=dataset['Z'],Energy=dataset['Energy'],lattice=dataset['lattice'])

##Correct coordinates of atom that passed the periodic boundary during relax
#Identify weird coordinates
# row,column=np.where(np.abs(dataset['Frac_R_relax']-dataset['Frac_R'])>0.8)

# np.savez('dataset/npz_combine/'+file_name+'.npz',
#     N=dataset['N'],R=dataset['R'],R_relax=dataset['R_relax'],Frac_R=dataset['Frac_R'],Frac_R_relax=dataset['Frac_R_relax'],Z=dataset['Z'],Energy=dataset['Energy'],lattice=dataset['lattice'])

print()