#!/bin/bash

#set directory to the leaf model, then run source (necessary at the start of each session
cd /home/m/malone/GitHub/leaf-project/vlab/oofs/ext/NPHLeafModels_1.01/$1
source ./../../../../bin/sourceme.sh

#generate the parameter file using the Input python script
#python3 ~/Desktop/Input.py

#prints the command before executing
set -o xtrace
#runs the model on the .h and .l files specified by leaffinder
./../../../../bin/lpfg -w 512 512 plantFig6_to_Fig7.map plant.a $2 plant.v -out $3

#for running lpfg on plant.l and thus MyParameters.h
#./../../../../bin/lpfg -w 512 512 plantFig6_to_Fig7.map plant.a plant.l plant.v -out leaf.png

#The command below generates a black and white image
#./../../../bin/lpfg -b plantFig6_to_Fig7.map plant.a plant.l plant.v -out leaf.png

