#!/usr/bin/env python3
from random import randrange
from numpy import arange
from sys import argv

TSTART = 0.2
TEND = 0.6
TSTEP = 0.02
TESTRUNS = 5
SIZESTART = 256
SIZEEND = 16_384
SIZEMULTIPLY = 4

if __name__ == "__main__":
    multising_config = """#
GPU = 1
grid_width = {0}
grid_height = {0}
alpha = 0.0
j = 1.0
total_updates = 5000
beta = {1}
seed = {2}
"""

    for run in range(TESTRUNS):
        size = SIZESTART
        seed = randrange(1, 1000000)

        while size <= SIZEEND:
            for temp in arange(TSTART, TEND, TSTEP):

                file_id = "{}/run={}_size={}_temp={:.3f}.conf".format(argv[1], run, size, temp)
                with open(file_id, "w") as f:
                    f.write(multising_config.format(size, temp, seed))

            size *= SIZEMULTIPLY
