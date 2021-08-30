#! /usr/bin/bash
cd crit_temperature/
for run in {0..9}
do
  mkdir run$run && cd run$run
  for scale in {0..14}
  do
    mkdir temp$scale && cd temp$scale
    ../../mkconfig.py $scale
    ../../../build/multising
    cd ../
  done
  cd ../
done
