#ifndef _PROB_PARM_H_
#define _PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

struct ProbParmDevice
{
  amrex::Real p = 1013250.0;
  amrex::Real T = 300.0;
  amrex::Real Tmax = 1000.0; // maximum of gaussian temperature sphere
  amrex::Real Trad = 0.04; // radius of gaussian temperature sphere
  amrex::Real rho = 0.0011694391;
  amrex::Real eint = 0.0;
  amrex::Real vx_in = 0.0;
  amrex::Real vy_in = 0.0;
  amrex::Real PhiV_top = 0.0;
  amrex::Real PhiV_bottom = 0.0;
  amrex::Real r0 = 250e-4; // cylinder radius        [cm]
  amrex::Real Ld = 0.25;   // cylinder length        [cm]
  amrex::Real y0 = 0.0;    // cylinder midpoint      [cm]
  amrex::Real sigma0 = 25e-4; // thickness  [cm]
  amrex::Real Erho = 1.2223e7;  // input energy density[erg/cm^3]
  int         Econst = 1; // constant energy density 
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {{1.0}};
}; 

struct ProbParmHost : amrex::Gpu::Managed
{
};

#endif
