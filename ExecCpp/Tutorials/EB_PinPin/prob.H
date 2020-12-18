#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

#include "mechanism.h"

#include "IndexDefines.H"
#include "Constants.H"
#include "EOS.H"
#include "Tagging.H"
#include "ProblemDerive.H"
#include "prob_parm.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata)
{
  // Geometry
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();

  const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
  const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];

  // Initialize kernel of O4+ ions in air
  // TODO: fix error when factor >= 1.0e-2
  // TODO: Code to find center of domain for arbitrary size
  amrex::Real mfs[NUM_SPECIES];
  amrex::Real tempT, tempE;
  for (int n = 0; n < NUM_SPECIES; n++)
    mfs[n] = ProbParm::massfrac[n];
  mfs[6] = 5.0e-3 * exp(-pow((x - 0.5) / 0.1, 2)) * exp(-pow((y - 0.5) / 0.1, 2));
  mfs[2]  = ProbParm::massfrac[2] - mfs[6];
  // EOS::PYT2RE(ProbParm::p, mfs, ProbParm::T, tempR, tempE);
  EOS::RYP2E(ProbParm::rho, mfs, ProbParm::p, tempE);
  EOS::EY2T(tempE, mfs, tempT);

  state(i, j, k, URHO) = ProbParm::rho;
  state(i, j, k, UMX) = ProbParm::rho * ProbParm::vx_in;
  state(i, j, k, UMY) = ProbParm::rho * ProbParm::vy_in;
  state(i, j, k, UMZ) = 0.0;
  state(i, j, k, UEINT) = ProbParm::rho * tempE;
  state(i, j, k, UEDEN) =
    ProbParm::rho *
    (tempE + 0.5 * (ProbParm::vx_in * ProbParm::vx_in +
                             ProbParm::vy_in * ProbParm::vy_in));
  state(i, j, k, UTEMP) = tempT;
  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = ProbParm::rho * mfs[n];

  // state(i, j, k, URHO) = tempR;
  // state(i, j, k, UMX) = tempR * ProbParm::vx_in;
  // state(i, j, k, UMY) = tempR * ProbParm::vy_in;
  // state(i, j, k, UMZ) = 0.0;
  // state(i, j, k, UEINT) = tempR * tempE;
  // state(i, j, k, UEDEN) =
  //   tempR *
  //   (tempE + 0.5 * (ProbParm::vx_in * ProbParm::vx_in +
  //                            ProbParm::vy_in * ProbParm::vy_in));
  // state(i, j, k, UTEMP) = ProbParm::T;
  // for (int n = 0; n < NUM_SPECIES; n++)
  //   state(i, j, k, UFS + n) = tempR * mfs[n];

#ifdef PELEC_USE_PLASMA  
  state(i, j, k, UFX) = 1.0;     // UFX : first AUX -> PhiV
  state(i, j, k, UFX+1) = 0.0;   // nE
#endif
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NVAR],
  amrex::Real s_ext[NVAR],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata)
{
  s_ext[URHO] = ProbParm::rho;
  s_ext[UMX] = ProbParm::rho * ProbParm::vx_in;
  s_ext[UMY] = ProbParm::rho * ProbParm::vy_in;
  s_ext[UMZ] = 0.0;
  s_ext[UEINT] = ProbParm::rho * ProbParm::eint;
  s_ext[UEDEN] = ProbParm::rho *
                 (ProbParm::eint + 0.5 * (ProbParm::vx_in * ProbParm::vx_in +
                                          ProbParm::vy_in * ProbParm::vy_in));
  s_ext[UTEMP] = ProbParm::T;
  for (int n = 0; n < NUM_SPECIES; n++)
    s_ext[UFS + n] = ProbParm::rho * ProbParm::massfrac[n];

#ifdef PELEC_USE_PLASMA  
  // Handle phiV bc in y-direction
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  amrex::Real halfy = (prob_hi[1] + prob_lo[1]) * 0.5;
  if (x[1] >= halfy) {
      s_ext[UFX] = ProbParm::PhiV_top;
  } else if (x[1] < halfy) {
      s_ext[UFX] = ProbParm::PhiV_bottom;
  }
  // Zero dirichlet on nE if needed
  s_ext[UFX+1] = 0.0;
#endif
}

#ifdef PELEC_USE_PLASMA  
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ef_bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NUM_E],
  amrex::Real s_ext[NUM_E],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata)
{
  // Setting zero grad BCs fo rnow
  for(int i=0; i<NUM_E; i++) s_ext[i] = s_int[i]; 
}
#endif

void pc_prob_close();

using ProblemTags = EmptyProbTagStruct;
using ProblemDerives = EmptyProbDeriveStruct;

#endif
