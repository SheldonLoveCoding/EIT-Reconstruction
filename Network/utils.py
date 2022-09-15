import torch
from scipy.io import loadmat



def dataLoader(path):
    datamat = loadmat(path)
    dataset = datamat['allRegionDataset']
    return dataset