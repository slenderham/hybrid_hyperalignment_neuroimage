#!/bin/bash

for i in {1..5}; do 
python run_anatomical_benchmarks.py budapest $i
done
