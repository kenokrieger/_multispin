#! /usr/bin/bash
ffmpeg -framerate 5 -i iteration_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p simulation.mp4
