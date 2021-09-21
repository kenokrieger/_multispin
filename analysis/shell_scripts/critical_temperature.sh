#! /usr/bin/bash

python3 analysis/python_scripts/mkconfig.py $1
for CONFIG_FILE in $1/*.conf; do
  ./build/multising $CONFIG_FILE
  mv magnetisation.dat "$1/$(basename "$CONFIG_FILE" .conf).mag"
  mv log "$1/$(basename "$CONFIG_FILE" .conf).log"
done

python3 analysis/python_scripts/plot_crittemp.py $1
