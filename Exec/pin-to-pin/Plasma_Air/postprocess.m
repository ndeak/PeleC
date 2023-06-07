clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Wrapping script for PeleC data line extraction
% Extracts lines of mass fractions, and other variables of interest
% Mass fractions are converted to number densities 
%
% Usage:
%   vars - String containing all fields to be extracted, space separated
%   dir - the extractionn direction (0=x, 1=y, 2=z)
%     (Note that extraction is only available in coord-aligned direction)
%   xc - coordinate of the x location where cut should be taken, units cm
%     (Assumed dir == 1 || dir == 2) 
%   yc - coordinate of the y location where cut should be taken, units cm
%     (Assumed dir == 0 || dir == 2) 
%   zc - coordinate of the z location where cut should be taken, units cm
%     (Assumed dir == 0 || dir == 1) 
%   tip1 - location of the lower pin tip (in the 'dir' coordinate)
%   tip2 - location of the upper pin tip (in the 'dir' coordinate)
%   datadir - directory that contains the unprocessed data
%   linedir - directory that contains the linecuts
%   moviedir - directory that contains the movie data
%   extractdir - directory that contains the fextract executable
%   ndens - total number density of the simulation, units 1/cm3
%     (Used to calculate reduced electric field)
%
%
% [ASSUMPTIONS AND USAGE NOTES]
%     - Movie script assumes Efieldx/y/z are last three vars, and uses these to create the reduced electric field
%     - phiV is assumed to preceed the electric field components
%     - Movie script also assumed vars preceding final 4 are various species number densities
%     - Currently, legend hard-codes the order of the species number densities - electron assumed first
%     - It is assumped that the fextract executable is compiled with gnu processor and MPI
%     - PeleC data files assumed to begin with 'plt' followed by 5 digit identifier (i.e. plt01000)
%     - Line extract performed in parallel on a single node 
%       (assumed on Stampede2 skylake node containing 48 procs)
%     - Works with PeleC/AMReX libraries up-to-date as of 08/26/2021
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Settings variables
vars = "'n(E) n(N2+) n(O2+) n(O2-) n(O-) n(O) n(OH) n(H) n(CO) n(CO2) n(H2O) n(C2H4) n(N2(v1)) n(N2(v2)) n(N2(v3)) n(N2(v4)) n(N2(v5)) n(N2(A3Sigma)) n(N2(B3Pi)) n(N2(C3Pi)) n(O2(a1Delta)) n(O2(b1Sigma)) n(O(1D)) pressure Temp rho_e phiV Efieldx Efieldy Efieldz'";
dir = 0;
xc = 1.0;
%yc = 0.325;
%yc = 0.265;
yc = 0.2;
zc = 1.0;
tip1 = 0.125;
tip2 = 0.375;
datadir = "/scratch1/04361/ndeak/pelec_data-7lev-300K-1atm-parabolicPinsFP-1p25mmh-50um-2p5mm-sigmoidPulse-13kVanode-2nsdt-2nsrt-1nspt-1MHz-PACMorrowIonNoCap-zeroGradBC-adiabatic/"
linedir = "/work2/04361/ndeak/frontera/forked_PeleC/PeleC/Exec/pin-to-pin/7lev-300K-1atm-parabolicPinsFP-1p25mmh-50um-2p5mm-sigmoidPulse-13kVanode-1MHz-2nsdt-2nsrt-1nspt-PACMorrowIonNoCap-zeroGradBC-adiabatic-impnEDiff/postproc-data/"
moviedir = "/work2/04361/ndeak/frontera/forked_PeleC/PeleC/Exec/pin-to-pin/7lev-300K-1atm-parabolicPinsFP-1p25mmh-50um-2p5mm-sigmoidPulse-13kVanode-1MHz-2nsdt-2nsrt-1nspt-PACMorrowIonNoCap-zeroGradBC-adiabatic-impnEDiff/postproc-movie/"
extractdir = "/work2/04361/ndeak/stampede2/amrex-combustion/amrex/Tools/Plotfile/"
ndens = 2.45e19;

% Call the extraction function
% May need to be called multiple times depending on size of data
pelecLineExtract(vars, dir, xc, yc, zc, datadir, linedir, extractdir);

% Call the movie creation function once all data has been extracted
% Removes blank lines from extract files, loads, and the plots data
% pelecCreateMovie(linedir, moviedir, ndens, tip1, tip2);
% pelecPointData(linedir, moviedir, ndens, tip1, tip2);
