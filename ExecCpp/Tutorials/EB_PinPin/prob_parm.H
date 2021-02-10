#ifndef _PROB_PARM_H_
#define _PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

struct ProbParmDevice : amrex::Gpu::Managed
{
  amrex::Real p = 1013250.0;
  amrex::Real T = 300.0;
  amrex::Real rho = 0.0;
  amrex::Real eint = 0.0;
  amrex::Real vx_in = 0.0;
  amrex::Real vy_in = 0.0;
  amrex::Real Re_L = 0.0;
  amrex::Real Pr = 0.0;
  amrex::Real PhiV_top = 0.0;
  amrex::Real PhiV_bottom = 0.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {1.0};
}; 

struct ProbParmHost : amrex::Gpu::Managed
{
};

#endif
