#!/usr/bin/env bash

data_dir=/home/bmooreii/projects/openbird/openbird_data/serser
files=($(tail -n +2 $data_dir/train.csv | cut -d, -f1))
image_prefix=serser
ini=serser_small.ini
output=$image_prefix-$ini-templates.pdf

rm $output

for file in ${files[@]}; do
    image_name=$image_prefix-$(echo $file | cut -d'/' -f2 | sed 's/^nips4b_birds_trainfile\([0-9]*\)\.wav$/\1/').png
    echo $file $image_name
    ./openbird.py view $file $image_name -i $ini
done

convert $image_prefix*.png $output

rm $image_prefix*.png
