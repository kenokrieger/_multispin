import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.plot(np.loadtxt("magnetisation.dat"))
    plt.show()
