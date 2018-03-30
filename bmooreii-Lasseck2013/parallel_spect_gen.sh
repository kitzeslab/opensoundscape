#!/usr/bin/env bash

find ../../openbird_data/nips4b/train -name "*.wav" | parallel --bar "./openbird.py spect_gen {}"
