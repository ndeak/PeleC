#ifndef PROB_H
#define PROB_H

#include <AMReX_Print.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_ParmParse.H>

#include "mechanism.H"

#include "PeleC.H"
#include "IndexDefines.H"
#include "PelePhysics.H"
#include "Tagging.H"
#include "ProblemDerive.H"
#include "prob_parm.H"
#include "Utilities.H"

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
  // const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  amrex::Real x[3] = {
    prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
    prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
    prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};

  // Fill in the velocities and energy.
  amrex::Real u[3] = {0.0};
  amrex::Real uinterp[3] = {0.0};

  // Interpolation factors
  amrex::Real mod[3] = {0.0};
  int idx[3] = {0};
  int idxp1[3] = {0};
  amrex::Real slp[3] = {0.0};
  for (int cnt = 0; cnt < 3; cnt++) {
    mod[cnt] = std::fmod(x[cnt], prob_parm.Linput);
    locate(prob_parm.d_xarray, prob_parm.inres, mod[cnt], idx[cnt]);
    idxp1[cnt] = (idx[cnt] + 1) % prob_parm.inres;
    slp[cnt] =
      (mod[cnt] - prob_parm.d_xarray[idx[cnt]]) / prob_parm.d_xdiff[idx[cnt]];
  }

  const amrex::Real f0 = (1 - slp[0]) * (1 - slp[1]) * (1 - slp[2]);
  const amrex::Real f1 = slp[0] * (1 - slp[1]) * (1 - slp[2]);
  const amrex::Real f2 = (1 - slp[0]) * slp[1] * (1 - slp[2]);
  const amrex::Real f3 = (1 - slp[0]) * (1 - slp[1]) * slp[2];
  const amrex::Real f4 = slp[0] * (1 - slp[1]) * slp[2];
  const amrex::Real f5 = (1 - slp[0]) * slp[1] * slp[2];
  const amrex::Real f6 = slp[0] * slp[1] * (1 - slp[2]);
  const amrex::Real f7 = slp[0] * slp[1] * slp[2];

  uinterp[0] =
    prob_parm.d_uinput
        [idx[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idx[2])] *
      f0 +
    prob_parm.d_uinput
        [idxp1[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idx[2])] *
      f1 +
    prob_parm.d_uinput
        [idx[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idx[2])] *
      f2 +
    prob_parm.d_uinput
        [idx[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idxp1[2])] *
      f3 +
    prob_parm.d_uinput
        [idxp1[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idxp1[2])] *
      f4 +
    prob_parm.d_uinput
        [idx[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idxp1[2])] *
      f5 +
    prob_parm.d_uinput
        [idxp1[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idx[2])] *
      f6 +
    prob_parm.d_uinput
        [idxp1[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idxp1[2])] *
      f7;
  uinterp[1] =
    prob_parm.d_vinput
        [idx[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idx[2])] *
      f0 +
    prob_parm.d_vinput
        [idxp1[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idx[2])] *
      f1 +
    prob_parm.d_vinput
        [idx[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idx[2])] *
      f2 +
    prob_parm.d_vinput
        [idx[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idxp1[2])] *
      f3 +
    prob_parm.d_vinput
        [idxp1[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idxp1[2])] *
      f4 +
    prob_parm.d_vinput
        [idx[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idxp1[2])] *
      f5 +
    prob_parm.d_vinput
        [idxp1[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idx[2])] *
      f6 +
    prob_parm.d_vinput
        [idxp1[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idxp1[2])] *
      f7;
  uinterp[2] =
    prob_parm.d_winput
        [idx[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idx[2])] *
      f0 +
    prob_parm.d_winput
        [idxp1[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idx[2])] *
      f1 +
    prob_parm.d_winput
        [idx[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idx[2])] *
      f2 +
    prob_parm.d_winput
        [idx[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idxp1[2])] *
      f3 +
    prob_parm.d_winput
        [idxp1[0] + prob_parm.inres * (idx[1] + prob_parm.inres * idxp1[2])] *
      f4 +
    prob_parm.d_winput
        [idx[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idxp1[2])] *
      f5 +
    prob_parm.d_winput
        [idxp1[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idx[2])] *
      f6 +
    prob_parm.d_winput
        [idxp1[0] + prob_parm.inres * (idxp1[1] + prob_parm.inres * idxp1[2])] *
      f7;

  u[0] = uinterp[0];
  u[1] = uinterp[1];
  u[2] = uinterp[2];

  // Assume quiescent conditions
  state(i, j, k, UMX) = 0.0;
  state(i, j, k, UMY) = 0.0;
  state(i, j, k, UMZ) = 0.0;

  // Uniform pressure and mass fractions
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  massfrac[0] = 7.7787881e-22;    // E
  massfrac[1] = 2.329175e-01;    // O2
  massfrac[2] = 7.670825e-01;    // N2
  massfrac[3] = 0.0;             // O
  massfrac[4] = 0.0;             // O2+
  massfrac[5] = 0.0;             // N2+
  massfrac[6] = 9.0678200e-17;    // O4+
  massfrac[7] = 0.0;             // N4+
  massfrac[8] = 0.0;             // O2+N2
  massfrac[9] = 0.0;    // O2-

  amrex::Real tempR = 0.0, tempE = 0.0; amrex::Real tempT = 0.0;

  // Fluctuating temperature field
  tempT = u[0] * 50.0 + 500.0;
  state(i, j, k, UTEMP) = tempT;

  // Calculate fluctuations in density and energy as well
  auto eos = pele::physics::PhysicsType::eos();
  eos.PYT2RE(prob_parm.p, massfrac, tempT, tempR, tempE);
  state(i, j, k, URHO) = tempR;
  state(i, j, k, UEINT) = tempR * tempE;
  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = tempR * massfrac[n];

  state(i, j, k, UEDEN) = state(i, j, k, UEINT);

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
  s_ext[URHO] = s_int[URHO];
  s_ext[UMX] = s_int[UMX];
  s_ext[UMY] = s_int[UMY];
  s_ext[UMZ] = s_int[UMZ];
  s_ext[UEINT] = s_int[UEINT];
  s_ext[UEDEN] = s_int[UEDEN];
  s_ext[UTEMP] = s_int[UTEMP];
  for (int n = 0; n < NUM_SPECIES; n++)
    // s_ext[UFS + n] = prob_parm.rho * prob_parm.massfrac[n];
    s_ext[UFS + n] = s_int[UFS + n];

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
#ifdef PELEC_USE_TWO_TEMP
  s_ext[UFX+5] = s_int[UFX+5];
  s_ext[UFX+6] = s_int[UFX+6];
#endif
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
