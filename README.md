# Parallel Image Blur: MPI vs Pthreads

Parallel implementations of an iterative 3x3 mean blur on a large ASCII PGM (P2) greyscale image, written in C for XJTLU DTS202TC (Fundamentals of Parallel Computing), individual assessment 2, December 2024. 并行计算课程作业：同一网格计算的 MPI 与 Pthread 两种实现及性能分析。

## What it does

Each pixel is replaced by the mean of its 8 neighbours (boundary pixels unchanged), repeated for 20 iterations, and the run time, speedup and parallel efficiency are measured for 1, 2 and 4 processes / threads.

- **MPI** (`mpi/blur_mpi.c`): rows are block-distributed across ranks; each iteration exchanges **ghost rows** with neighbouring ranks (`MPI_Sendrecv`), uses **double buffering** (swap input/output arrays) and an `MPI_Barrier` per iteration.
- **Pthreads** (`pthread/blur_pthread.c`): rows are split among threads, with a `pthread_barrier` between iterations and AVX2 vectorisation for the main part of each row; it times 1, 2 and 4 threads and prints speedup and efficiency.
- **Serial reference** (`serial/blur_serial.c`) and a small PGM comparison script for output correctness (`tools/compare_pgm.py`).

## Results

From the coursework report: with 4 MPI processes the speedup is about **3.6x (about 90% parallel efficiency)**, and the Pthread version scales less well. Timings depend on the machine; run the programs on your own hardware to reproduce them.

## Build and run (not re-verified)

```bash
mpicc -O2 -o blur_mpi mpi/blur_mpi.c && mpirun -np 4 ./blur_mpi        # reads im.pgm from the working directory
gcc -O2 -mavx2 -pthread -o blur_pthread pthread/blur_pthread.c && ./blur_pthread im.pgm   # runs 1, 2, 4 threads
```
Requires an MPI implementation (e.g. OpenMPI), gcc with pthreads, and an AVX2-capable CPU for the Pthread version (`-mavx2`). I did not compile or run the code when preparing this repository.

## Tech stack
C, MPI, POSIX threads, AVX2, Python (comparison script).

## Not included
The input image (`im.pgm`, about 50 MB, provided with the coursework), output images, executables and the coursework report/specification.
