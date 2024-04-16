import numpy as np

def popInitialization(NP,size):
    # for i in range(NP):
    #     for j in range(size):
    #         xPopulation[i, j] = xMin + rd.random() * (xMax - xMin)
    xPopulation=np.zeros((NP,size))+np.random.rand(NP,size)
    return xPopulation