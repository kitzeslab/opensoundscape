#!/usr/bin/env bash

# Simple BASH script which will generate a PDF of many image files
# 1. It needs to draw a list of files from somewhere, the
# -> `train.csv` file below only contains birds labeled as Sylatr_call
data_dir=/home/bmooreii/projects/opensoundscape/opensoundscape_data/sylatr
files=($(tail -n +2 $data_dir/train.csv | cut -d, -f1))

# Image prefix can be whatever you like
image_prefix=sylatr
# The database definitions, can be different than the `train.csv` above!
# -> However, labels must be available between the 2
ini=../nips4b_small.ini
output=$image_prefix-templates.pdf

rm $output

for file in ${files[@]}; do
    image_name=$image_prefix-$(echo $file | cut -d'/' -f2 | sed 's/^nips4b_birds_trainfile\([0-9]*\)\.wav$/\1/').png
    echo $file $image_name
    ../opensoundscape.py view $file $image_name -i $ini
done

convert $image_prefix*.png $output

rm $image_prefix*.png
