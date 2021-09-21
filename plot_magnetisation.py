import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    magnetisation = np.loadtxt("magnetisation.dat")
    plt.plot(magnetisation)
    plt.show()

    plt.plot(np.diff(np.log(magnetisation)))
    plt.show()
