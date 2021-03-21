#!/usr/bin/env python3

import numpy as np
import pandas as pd

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def simple_gradient(x, y, theta):
    if type(theta) != np.ndarray or type(x) != np.ndarray or x.ndim != 1\
            or theta.ndim != 1 or len(theta) != 2 or type(y) != np.ndarray\
            or y.ndim != 1 or len(y) != len(x):
        return None
    else:
        X = np.concatenate(((np.ones((len(x), 1))), x.reshape(len(x), 1)),
                           axis=1)
        h0 = X @ theta
        J = ((h0 - y) @ X)/len(x)
        return((J))

def fit(x, y, theta, alpha=0.005, max_iter=1500):
    if type(theta) != np.ndarray or type(x) != np.ndarray or x.ndim != 1\
            or theta.ndim != 1 or len(theta) != 2 or type(y) != np.ndarray\
            or y.ndim != 1 or len(y) != len(x) or type(alpha) != float or\
        type(max_iter) != int or max_iter < 0 :
        return None
    else:
        new_theta = np.array([float(theta[0]), float(theta[1])])
        for i in range(max_iter):
            J = simple_gradient(x, y, new_theta)
            new_theta[0] = new_theta[0] - alpha * J[0]
            new_theta[1] = new_theta[1] - alpha * J[1]
        return(new_theta)

def normalizer(data, file):
    if type(data) != np.ndarray or data.ndim != 1:
        return None
    else:
        data_max = np.max(data)
        data_min = np.min(data)
        file.writelines(f"min = {data_min}\n")
        file.writelines(f"max = {data_max}\n")
        norm_data = []
        norm_data = (data - data_min) / (data_max - data_min)
        return (np.array(norm_data))


def train_model():
    filename = input(f"{colors.OKBLUE}Please, introduce a data file:\n" +
        colors.ENDC)
    if not filename:
        filename = 'data.csv'
        print("data.csv")
    try:
        file = pd.read_csv(filename)
        data = np.asarray(file).transpose()
        x = data[0]
        y = data[1]
    except:
        print(colors.WARNING + "Invalid data file")
        return
    theta = []
    try:
        theta_file = open('theta.txt', 'r+')
        lines = theta_file.readlines()
        for i in range(1, 3):
            words = lines[i].split()
            theta.append(float(words[2]))
        print(f"{colors.OKBLUE}Theta file found, starting training with thetha"
         + f" = {theta}{colors.ENDC}")
    except:
        print("No theta file, starting training with theta [0.0, 0.0]")
        theta = [0.0, 0.0]

    print(x)
    norm_x = normalizer(x, theta_file)
    print(norm_x)
    new_theta = fit(norm_x, y, np.array(theta))
    print(new_theta)

if __name__ == "__main__":
   train_model()
