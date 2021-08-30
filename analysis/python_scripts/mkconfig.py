#!/usr/bin/env python3
from random import randrange
from numpy import arange
from sys import argv

if __name__ == "__main__":
    multising_config = ""
    with open("temperature.conf", "r") as f:
        for line in f.readlines():
            if "TSTART" in line:
                tstart = float(line.split("=")[1].strip())
            elif "TEND" in line:
                tend = float(line.split("=")[1].strip())
            elif "TSTEP" in line:
                tstep = float(line.split("=")[1].strip())
            elif "TESTRUNS" in line:
                testruns = int(line.split("=")[1].strip())
            else:
                multising_config += line

    for run in range(testruns):
        seed = f"\nseed = {randrange(1, 1000000)}"
        for temp in arange(tstart, tend, tstep):
            config_temp = "\nbeta = {}".format(temp)

            file_id = "{}/run={}_temp={:.3f}.conf".format(argv[1], run, temp)
            with open(file_id, "w") as f:
                f.write(multising_config + seed + config_temp)
