#!/usr/bin/env python3

import numpy as np
import pandas as pd

file = pd.read_csv('data.csv')
arr = np.asarray(file)
theta = []
try:
    theta_file = open('theta.txt', 'r+')
    lines = theta_file.readlines()
    print(lines[1])
    print(lines[2])
    for i in range(1, 3):
        words = lines[i].split()
        theta.append(float(words[2]))
    print(f"Theta file found, starting training with {theta}")
except:
    print("No theta file, starting training with [1.0, 1.0]")
    theta_file = open("theta.txt","w+")
    theta_file.write("Thetha training data:\nt0 = 1.0\nt1 = 1.0\n")
    theta_file.seek(0)



if __name__ == "__main__":
    print("Hola")
    #print(file)
    #print(arr)
