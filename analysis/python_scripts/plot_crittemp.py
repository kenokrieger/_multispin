#! /usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
from os.path import join, basename


if __name__ == "__main__":
    magnetisation = dict()
    data_files = glob.glob(argv[1] + "/*.mag")

    for file in data_files:
        with open(file.replace(".mag", ".conf"), "r") as f:
            beta = [float(line.split("=")[1]) for line in f.readlines() if "beta" in line][0]
        if beta in magnetisation.keys():
            magnetisation[beta].append(np.ptp(np.loadtxt(file)))
        else:
            magnetisation[beta] = [np.ptp(np.loadtxt(file))]

    for temp in magnetisation.keys():
        plt.scatter(temp, np.amax(np.abs(magnetisation[temp])), marker="x", color="k")

    plt.title("Magnetisation as a function of the inverse temperature")
    plt.xlabel("Inverse temperature")
    plt.ylabel("Magnetisation")
    plt.savefig("results/sid={}_critical_temperature.png".format(basename(argv[1])))
