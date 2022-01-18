#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

#include "EOS.H"
#include "prob_parm.H"

#include "PeleC.H"
#include "IndexDefines.H"
#include "PelePhysics.H"

#include "Constants.H"
#include "Tagging.H"
#include "ProblemDerive.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  int use_NL,
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
  amrex::Real mfs[NUM_SPECIES];
  amrex::Real tempT, tempE;
  for (int n = 0; n < NUM_SPECIES; n++)
    mfs[n] = 0.0;

  mfs[1] = 0.233;
  mfs[2] = 0.767;

  // Mass fractions corresponding to the A2-const ICs
  // mfs[0] = 1.402983e-28;    // E
  // mfs[1] = 2.329175e-01;    // O2
  // mfs[2] = 7.670825e-01;    // N2
  // mfs[3] = 0.0;             // O
  // mfs[4] = 0.0;             // O2+
  // mfs[5] = 0.0;             // N2+
  // mfs[6] = 1.597029e-18;    // O4+
  // mfs[7] = 0.0;             // N4+
  // mfs[8] = 0.0;             // O2+N2
  // mfs[9] = 7.450388e-19;    // O2-

  // Mass fractions corresponding to 1e3 1/cm3 ICs for electrons and O4+
  // mfs[0] = 7.7787881e-22;    // E
  // mfs[1] = 2.329175e-01;    // O2
  // mfs[2] = 7.670825e-01;    // N2
  // mfs[3] = 0.0;             // O
  // mfs[4] = 0.0;             // O2+
  // mfs[5] = 0.0;             // N2+
  // mfs[6] = 9.0678200e-17;    // O4+
  // mfs[7] = 0.0;             // N4+
  // mfs[8] = 0.0;             // O2+N2
  // mfs[9] = 0.0;    // O2-

  // Mass fractions corresponding to 1e3 1/cm3 ICs for electrons and O4+
  mfs[0] = 7.7787881e-16;    // E
  mfs[1] = 2.329175e-01;    // O2
  mfs[2] = 7.670825e-01;    // N2
  mfs[3] = 0.0;             // O
  mfs[4] = 0.0;             // O2+
  mfs[5] = 0.0;             // N2+
  mfs[6] = 9.0678200e-11;    // O4+
  mfs[7] = 0.0;             // N4+
  mfs[8] = 0.0;             // O2+N2
  mfs[9] = 0.0;    // O2-

  auto eos = pele::physics::PhysicsType::eos();
  eos.RYP2E(prob_parm.rho, mfs, prob_parm.p, tempE);
  eos.EY2T(tempE, mfs, tempT);

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
  state(i, j, k, UFX) = 1.0;     // UFX : first AUX -> PhiV
  state(i, j, k, UFX+1) = 1.0;   // nE
  state(i, j, k, UFX+2) = 1.0;   // nE
  state(i, j, k, UFX+3) = 1.0;   // nE
  state(i, j, k, UFX+4) = 1.0;   // nE
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
  // Handle phiV bc in y-direction (PIN PIN CASE)
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

  s_ext[UFX+2] = s_int[UFX+2];
  s_ext[UFX+3] = s_int[UFX+3];
  s_ext[UFX+4] = s_int[UFX+4];
  // Set PhiV on spherical electrodes (SPHERICAL DISCHARGE CASE)
  // const amrex::Real* prob_lo = geomdata.ProbLo();
  // const amrex::Real* prob_hi = geomdata.ProbHi();
  // const amrex::Real* dx      = geomdata.CellSize();
  // amrex::Real r = sqrt((x[0]-2.0)*(x[0]-2.0) + (x[1]-2.0)*(x[1]-2.0) + (x[2]-2.0)*(x[2]-2.0));
  // if (r >= 1.0) {
  //     s_ext[UFX] = prob_parm.PhiV_top;
  // } else {
  //     s_ext[UFX] = prob_parm.PhiV_bottom;
  // }
  // // Zero dirichlet on nE if needed
  // s_ext[UFX+1] = 0.0;
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
