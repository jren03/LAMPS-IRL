#!/bin/bash

# Loop through all .sub files in the current directory
for file in *.sub
do
    sbatch --requeue "$file"
done