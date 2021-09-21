#!/usr/bin/env python3
from random import randrange
from numpy import arange
from sys import argv
from os.path import exists

CONFIG_FILE = "temperature.conf"


if __name__ == "__main__":
    multising_config = ""

    if not exists(CONFIG_FILE):
        print(f"Configuration file '{CONFIG_FILE}' not found.")
        print("Manually create file or abort? [m/A]")
        procedure = input().strip()

        if procedure == "m":
            input("Press enter, when you are ready to resume")
        elif procedure == "A":
            print("Aborting...")
            exit()

    tstart, tend, tstep, testruns = -1
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

    if tstart == -1 or tend == -1 or tstep == -1 or testruns == -1:
        print("Missing parameter in configuration file.")
        exit()
        
    for run in range(testruns):
        seed = f"\nseed = {randrange(1, 1000000)}"
        for temp in arange(tstart, tend, tstep):
            config_temp = "\nbeta = {}".format(temp)

            file_id = "{}/run={}_temp={:.3f}.conf".format(argv[1], run, temp)
            with open(file_id, "w") as f:
                f.write(multising_config + seed + config_temp)
