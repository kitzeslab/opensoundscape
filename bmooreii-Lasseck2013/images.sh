#!/usr/bin/env bash

files=($(tail -n +2 /home/bmooreii/projects/openbird/openbird_data/sylatr/train.csv | cut -d, -f1))

for file in ${files[@]}; do
    image_name=sylatr-$(echo $file | cut -d'/' -f2 | sed 's/^nips4b_birds_trainfile\([0-9]*\)\.wav$/\1/').png
    echo $file $image_name
    ./openbird.py view $file $image_name -i sylatr_small.ini -s
done
