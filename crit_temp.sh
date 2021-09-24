#! /usr/bin/bash

python3 mkconfig.py $1
for CONFIG_FILE in $1/*.conf; do
  ./build/multising $CONFIG_FILE
  mv magnetisation.dat "$1/$(basename "$CONFIG_FILE" .conf).mag"
  mv log "$1/$(basename "$CONFIG_FILE" .conf).log"
done
