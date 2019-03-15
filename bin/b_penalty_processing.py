import numpy as np
import pickle

def calc_t(comm):
    nd_comm = np.array(comm)
    ind_update = [0] + [np.argwhere(nd_comm==el)[-1][0]+1 for el in set(nd_comm)]
    t_tuples = zip(ind_update[:-1],ind_update[1:])
    # t_tuples = list(t_tuples)[:-1] # remove last tuple as the end of the list is not an update
    t = [tup[1]-tup[0] for tup in t_tuples]
    return t