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
  int useNL,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
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
  amrex::Real mwt[NUM_SPECIES] = {0.0};
  EOS::molecular_weight(mwt);
  amrex::Real mfs[NUM_SPECIES] = {0.0};
  amrex::Real tempT, tempE;

  // Air
  mfs[O2_ID] = 0.233;

  // Some electrons
  //mfs[E_ID] = 5.0e-16 * exp(-pow((y - 0.5) / 0.02, 2)) + 1e-20;

  amrex::Real Na = 6.022e23;
  // if using the non-linear solve move E from species to aux
  // and convert from massfrac -> nE = rho Y_e / mw_e * Na
  if ( useNL ) {
     state(i, j, k, UFX+1) = mfs[E_ID] * prob_parm.rho / mwt[E_ID] * Na;
     mfs[E_ID] = 0.0;
  }

  // N2
  mfs[N2_ID] = 1.0 - mfs[O2_ID] - mfs[E_ID];

  // mfs[9] = 5.0e-3 * exp(-pow((x - 0.5) / 0.1, 2)) * exp(-pow((y - 0.5) / 0.1, 2));
  // mfs[2]  = prob_parm.massfrac[2] - mfs[9];
  // EOS::PYT2RE(prob_parm.p, mfs, prob_parm.T, tempR, tempE);
  EOS::RYP2E(prob_parm.rho, mfs, prob_parm.p, tempE);
  EOS::EY2T(tempE, mfs, tempT);

  state(i, j, k, URHO) = prob_parm.rho;
  state(i, j, k, UMX) = prob_parm.rho * prob_parm.vx_in;
  state(i, j, k, UMY) = prob_parm.rho * prob_parm.vy_in;
  state(i, j, k, UMZ) = 0.0;
  state(i, j, k, UEINT) = prob_parm.rho * tempE;
  state(i, j, k, UEDEN) =
    prob_parm.rho *
    (tempE + 0.5 * (prob_parm.vx_in * prob_parm.vx_in +
                             prob_parm.vy_in * prob_parm.vy_in));
  state(i, j, k, UTEMP) = tempT;
  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = prob_parm.rho * mfs[n];

  // state(i, j, k, URHO) = tempR;
  // state(i, j, k, UMX) = tempR * prob_parm.vx_in;
  // state(i, j, k, UMY) = tempR * prob_parm.vy_in;
  // state(i, j, k, UMZ) = 0.0;
  // state(i, j, k, UEINT) = tempR * tempE;
  // state(i, j, k, UEDEN) =
  //   tempR *
  //   (tempE + 0.5 * (prob_parm.vx_in * prob_parm.vx_in +
  //                            prob_parm.vy_in * prob_parm.vy_in));
  // state(i, j, k, UTEMP) = prob_parm.T;
  // for (int n = 0; n < NUM_SPECIES; n++)
  //   state(i, j, k, UFS + n) = tempR * mfs[n];

#ifdef PELEC_USE_PLASMA  
  state(i, j, k, UFX) = 0.0;     // UFX : first AUX -> PhiV
  if (!useNL) {
     state(i, j, k, UFX+1) = 0.0;   // nE
  }
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
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{
  s_ext[URHO] = prob_parm.rho;
  s_ext[UMX] = prob_parm.rho * prob_parm.vx_in;
  s_ext[UMY] = prob_parm.rho * prob_parm.vy_in;
  s_ext[UMZ] = 0.0;
  s_ext[UEINT] = prob_parm.rho * prob_parm.eint;
  s_ext[UEDEN] = prob_parm.rho *
                 (prob_parm.eint + 0.5 * (prob_parm.vx_in * prob_parm.vx_in +
                                          prob_parm.vy_in * prob_parm.vy_in));
  s_ext[UTEMP] = prob_parm.T;
  for (int n = 0; n < NUM_SPECIES; n++)
    s_ext[UFS + n] = prob_parm.rho * prob_parm.massfrac[n];

#ifdef PELEC_USE_PLASMA  
  // Handle phiV bc in y-direction
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  amrex::Real halfy = (prob_hi[1] + prob_lo[1]) * 0.5;
  if (x[1] >= halfy) {
      s_ext[UFX] = prob_parm.PhiV_top;
  } else if (x[1] < halfy) {
      s_ext[UFX] = prob_parm.PhiV_bottom;
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
