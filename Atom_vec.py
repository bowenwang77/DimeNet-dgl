import sys
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt

AtomVecs=[]
with open("Atom2Vec/atoms_vec.txt", 'r') as f:
    lines=f.readlines()
    for i in range(len(lines)):
        AtomVecs.append(np.array(lines[i].split()).astype(np.float))
AtomIds=[]
with open("Atom2Vec/atoms_index.txt", 'r') as f:
    lines=f.readlines()
    for i in range(len(lines)):
        AtomIds.append(np.array(lines[i].split()).astype(np.int))
AtomVecdict={}
for i in range(len(AtomVecs)):
    AtomVecdict[AtomIds[i].item()]=AtomVecs[i]

import pickle

with open('Atom2Vec_dict.pickle','wb') as f:
    pickle.dump(a,f)