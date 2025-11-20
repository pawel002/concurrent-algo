### Running scripts

Runing python example:

```bash
piexec -n 5 python example.py
```

Compiling and running C locally:

```bash
mpicc -std=c11 -O2 sieve.c -o sieve
mpiexec -n 5 ./sieve
```

On ares:

```bash
# load modules
module load gcc/13.2.0
module load openmpi/5.0.2-gcc-13.2.0
# compile
mpicc -std=c11 -O2 sieve.c -o sieve
# run SLURM
srun --overlap -A plgar2025-cpu -N 1 --tasks-per-node=4 -p plgrid -t 20:00 --pty /bin/bash
# execute
./sieve
```
