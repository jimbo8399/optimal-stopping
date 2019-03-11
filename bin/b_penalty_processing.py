import numpy as np
import pickle

def calc_t(comm):
    nd_comm = np.array(comm)
    ind_update = [0] + [np.argwhere(nd_comm==el)[-1][0]+1 for el in set(nd_comm)]
    t_tuples = zip(ind_update[:-1],ind_update[1:])
    t = [tup[1]-tup[0] for tup in t_tuples]
    # data = (dataset,kernel_name,ostPenalty,t)
    return t