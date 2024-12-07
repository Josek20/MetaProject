#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --nodes=1 --cores-per-socket=8
#SBATCH --mem=64G
#SBATCH -p amdfast
#SBATCH --error=/home/shuhaole/logs/tree_simp_%j.err
#SBATCH --out=/home/shuhaole/logs/tree_simp_%j.out

ml Julia/1.10.2-linux-x86_64
julia -p 1 --project=./test/bemchmarking test/benchmarking/benchmark.jl
# julia -p 10 --project=./example example/tree_simp_example.jl