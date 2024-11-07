#!/bin/bash

export AMREX_HOME=/work2/07638/aduarteg/frontera/research/repositories/plasma/amrexclean/amrex
export PELE_PHYSICS_HOME=/work2/07638/aduarteg/frontera/research/repositories/plasma/amrexclean/PelePhysics
export AMREX_HYDRO_HOME=/work2/07638/aduarteg/frontera/research/repositories/plasma/amrexclean/AMReX-Hydro
module load gcc

idev -p development -N 1 -n 8 -m 30 -A CTS22014
