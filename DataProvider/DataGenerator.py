import numpy as np
import os


def Fetch_filelist(root):
    filelist = sorted(os.listdir(root))
    individuals = {}

    for filename in filelist:
        # print(filename)
        individual = filename[:5]
        if individual not in individuals:
            individuals[individual] = []
        individuals[individual].append(filename)
    return individuals


