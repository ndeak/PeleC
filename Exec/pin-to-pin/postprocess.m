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
vars = "'n(E) n(N2+) n(N4+) n(O2+) n(O4+) n(O2pN2) n(O2-) phiV Efieldx Efieldy Efieldz'";
dir = 1;
xc = 1.0;
yc = 1.0;
zc = 1.0;
tip1 = 0.75;
tip2 = 1.0;
datadir = "/scratch1/04361/ndeak/pelec_data-6lev-0p27cfl-343K-1p3bar-parabolicPinsFP-50um-2p5mm-trapPulse-15kVanode-10nst-10fwhm-1e3init-SRD1old-scaling-Egradtag100-kossyiUpd-0p01SEEC/"
linedir = "/work2/04361/ndeak/frontera/forked_PeleC/PeleC/Exec/pin-to-pin/6lev-0p27cfl-MOL-343K-1.3bar-parabolicPinsFP-50um-2p5mm-trapPulse-15kVanode-10nst-10fwhm-1e3init-SRD1old-scaling-Egradtag100-kossyiUpd-0p01SEEC/postproc-data/"
moviedir = "/work2/04361/ndeak/frontera/forked_PeleC/PeleC/Exec/pin-to-pin/6lev-0p27cfl-MOL-343K-1.3bar-parabolicPinsFP-50um-2p5mm-trapPulse-15kVanode-10nst-10fwhm-1e3init-SRD1old-scaling-Egradtag100-kossyiUpd-0p01SEEC/postproc-movie/"
extractdir = "/work2/04361/ndeak/stampede2/amrex-combustion/amrex/Tools/Plotfile/"
ndens = 2.45e19;

% Call the extraction function
% May need to be called multiple times depending on size of data
pelecLineExtract(vars, dir, xc, yc, zc, datadir, linedir, extractdir);

% Call the movie creation function once all data has been extracted
% Removes blank lines from extract files, loads, and the plots data
pelecCreateMovie(linedir, moviedir, ndens, tip1, tip2);
