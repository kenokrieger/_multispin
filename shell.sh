#! /usr/bin/bash
cd crit_temperature/
for run in {32..50}
do
  mkdir run$run && cd run$run
  rm -r temp*
  for scale in {0..15}
  do
    mkdir temp$scale && cd temp$scale
    ../../mkconfig.py $scale
    ../../../build/multising
    cd ../
  done
  cd ../
done
shutdown now
