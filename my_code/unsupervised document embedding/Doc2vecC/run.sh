#! /bin/bash

gcc doc2vecc.c -o doc2vecc -lm -pthread -O3 -march=native -funroll-loops