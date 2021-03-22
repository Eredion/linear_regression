#! /usr/bin/env python3
from train import get_theta

if __name__ == "__main__":
    theta = get_theta()
    if theta == [0, 0]:
        print('\033[93m' + "Model not trained, run ./train first.")
        exit()

    n = input("\033[94mPlease, introduce a mileage in Km:\n" + '\033[0m')
    try:
        n = float(n)
        if n < 0:
            raise ValueError
    except:
        print('\033[93m' + "Invalid milleage! Please, introduce a valid number.")
        exit()
    y = (theta[0]) + (theta[1] * n)
    if y < 0:
        y = 0
    print(f"\033[92mEstimated price: {round(y, 2)} $")
