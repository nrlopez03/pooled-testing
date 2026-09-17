#!/bin/bash
#SBATCH --job-name=python_sum_job # Job name
#SBATCH --output=job_output-%j.txt # Output and error log
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=16 # Number of CPU cores per task
#SBATCH --mem=8G # Total memory requirement (8 GB)

# Load Spack setup script (modify the path according to your Spack installation)
source /shared/spack/share/spack/setup-env.sh

# Activate a Spack-installed package if needed (uncomment line and adjust accordingly)
# spack load python@3.8.0

# We're using Conda installed by Spack so we need to activate that environment before we can use any conda commands
spack env activate conda

# Activate the conda environment you want to run with your script
conda init
source /shared/home/nil527/.bashrc
conda activate apmth91r

# Run the Python script
/shared/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-14.1.0/miniconda3-24.3.0-zxx5jostrj4myhf7bi3oap3ylkmegd3a/envs/apmth91r/bin/python greedyDynamicSample.py

# Optionally: Deactivate the Conda environment after running the script
conda deactivate
