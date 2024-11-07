#!/bin/bash

# Declare parameters
T0="300.0"
P0="1013250.0"
r0="150e-4"
Q="2000.0"

# Values in cgs
declare -a Qarr=("3000" "4320" "6000" "9600")
declare -a sigmaarr=("12.5e-4" "18.0e-4" "25.0e-4" "40.0e-4")
declare -a gaparr=("0.125" "0.18" "0.25" "0.4")

# Values in mks
declare -a Qval=("300e-6" "432e-6" "600e-6" "960e-6")
declare -a gap_d=("1.25e-3" "1.8e-3" "2.5e-3" "4.0e-3")
density_0="1.1768"

# Modify the input file accordingly
sed -i "s|.*prob.p.*|prob.p = ${P0} |" inputs.3d
sed -i "s|.*prob.T.*|prob.T = ${T0} |" inputs.3d
sed -i "s|.*prob.r0.*|prob.r0 = ${r0} |" inputs.3d
sed -i "s|.*prob.Erho.*|prob.Erho = ${Q} |" inputs.3d

# Get path and run
PELECHOME=/work2/07638/aduarteg/frontera/research/repositories/plasma/amrexclean/PeleC/Exec/pin-to-pin/examples/Tbubble


ip=0
for i in "${Qarr[@]}"
do

   # Modify the input file accordingly
   sed -i "s|.*prob.p.*|prob.p = ${P0} |" inputs.3d
   sed -i "s|.*prob.T.*|prob.T = ${T0} |" inputs.3d
   sed -i "s|.*prob.r0.*|prob.r0 = ${r0} |" inputs.3d
   sed -i "s|.*prob.Ld.*|prob.Ld = ${gaparr[${ip}]} |" inputs.3d
   sed -i "s|.*prob.Erho.*|prob.Erho = ${i} |" inputs.3d
   sed -i "s|.*prob.sigma0.*|prob.sigma0 = ${sigmaarr[${ip}]} |" inputs.3d   

   ./clean
#   ibrun $PELECHOME/PeleC3d.gnu.MPI.ex inputs.3d
   
   # Process 2-D files
   cd postprocess2D
   ./clean
   # process input filei
   cp inp_temp.yaml inp_merge.yaml
   sed -i "s|.*pressure/10.*|  formula: 'pressure/10*200e-6^2*${gap_d[$ip]}/${Qval[$ip]}'   |" inp_merge.yaml
      sed -i "s|.*density/density_0.*|  formula: 'density*1e3/$density_0'   |" inp_merge.yaml
      sed -i "s|.*x_velocity/100.*|  formula: 'x_velocity/100*200e-6^2/${gap_d[$ip]}*sqrt($density_0*${gap_d[$ip]}/${Qval[$ip]})'   |" inp_merge.yaml
      sed -i "s|.*y_velocity/100.*|  formula: 'y_velocity/100*200e-6^2/${gap_d[$ip]}*sqrt($density_0*${gap_d[$ip]}/${Qval[$ip]})'   |" inp_merge.yaml
      sed -i "s|.*z_velocity/100.*|  formula: 'z_velocity/100*200e-6^2/${gap_d[$ip]}*sqrt($density_0*${gap_d[$ip]}/${Qval[$ip]})'   |" inp_merge.yaml
      sed -i "s|.*magvel/100.*|  formula: 'magvel/100*200e-6^2/${gap_d[$ip]}*sqrt($density_0*${gap_d[$ip]}/${Qval[$ip]})'   |" inp_merge.yaml

    
#   ./run.sh
   cd ..
   
   cd postprocess1D
   ./clean
#   ./run.sh
   cd ..

   cp -r postprocess1D C_${ip}
   cp -r postprocess2D C_${ip}_2D
   ip=$(($ip+1))
done

