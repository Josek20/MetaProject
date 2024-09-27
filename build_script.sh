#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --nodes=1 --cores-per-socket=8
#SBATCH --mem=64G
#SBATCH -p amdfast
#SBATCH --error=/home/shuhaole/logs/tree_simp.err
#SBATCH --out=/home/shuhaole/logs/tree_simp.out

ml Julia/1.10.2-linux-x86_64
julia -p 10 --project=./example example/tree_simp_example.jl