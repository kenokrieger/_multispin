#! /usr/bin/env python3
"""Plots a live view of the magnetisation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


with open("../multising.conf") as f:
    for element in f.readlines():
        if "grid_height" in element:
            LATTICE_SIZE = int(element.replace("grid_height = ", "")) ** 3
            break
    else:
        print("Internal Error")
        exit(0)


if __name__ == "__main__":

    magnetisation = np.loadtxt("magnetisation.dat")
    plt.plot(magnetisation[:, 0] / LATTICE_SIZE)

    plt.title("Relative Magnetisation")
    plt.savefig("magnetisation.png")
    plt.close()

    # the very first value of the difference is random
    plt.plot(magnetisation[1:, 1] / LATTICE_SIZE)
    plt.title("Relative magnetisation difference between updates")
    plt.savefig("magnetisation_difference.png")
    plt.close()
