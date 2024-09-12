#!/bin/bash
source project.bashrc
ocean -nograph -replay pll_vco.ocn -log pll_vco.log
FILE=./simulation
while [ ! -d "$FILE" ]
do
    ./clear.sh
    ocean -nograph -replay pll_vco.ocn -log pll_vco.log
done
