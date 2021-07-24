#! /usr/bin/env python3
"""Plots a live view of the magnetisation"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


with open("multising.conf") as f:
    for element in f.readlines():
        if "grid_height" in element:
            LATTICE_SIZE = int(element.replace("grid_height = ", "")) ** 3
            break
    else:
        print("Internal Error")
        exit(0)


if __name__ == "__main__":
    fig, ax = plt.subplots()

    def animate(i):
        magnetisation = np.loadtxt("magnetisation.dat", max_rows = 1)
        idx = np.arange(0, len(magnetisation))
        plt.cla()
        ax.plot(idx, magnetisation / LATTICE_SIZE)
        plt.title("Relative Magnetisation")

    #ani = FuncAnimation(fig, animate, interval=500)
    animate(1)
    plt.savefig("magnetisation.png")
