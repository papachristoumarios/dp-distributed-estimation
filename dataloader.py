import networkx as nx
import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

def load_dataset(args):

    name = args.name
    signals = None

    if name == 'karate_club':
        G = nx.karate_club_graph()
    elif name == 'ieee_33_bus':
        mat = scipy.io.loadmat('datasets/33bus_modified_ybus.mat')
        A = (mat['xx'] != 0).astype(np.float64)
        G = nx.from_numpy_array(A)
    elif name == 'us_power_grid':
        df = pd.read_csv('datasets/us_power_grid.txt', sep=' ', skiprows=4, header=None, names=['u', 'v', 'w'])
        G = nx.from_pandas_edgelist(df, source='u', target='v')
    elif name == 'germany_consumption':
        signals = np.loadtxt('datasets/germany-consumption/log_signals.txt.gz')
        n = signals.shape[0]
        G = nx.random_geometric_graph(n, 0.1, seed=0)
    else:
        raise Exception('Invalid dataset name')


    return G, signals