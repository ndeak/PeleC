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
#ifdef PELEC_USE_PLASMA  
  int use_NL,
#endif
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
  amrex::Real rhonew = 0.0;
  amrex::Real Enew = 0.0;
  amrex::Real yoLd = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++)
    mfs[n] = 0.0;

  mfs[O2_ID] = 0.21;
  mfs[N2_ID] = 0.79;

  // Set background conditions
  amrex::Real r = pow(pow(x , 2) + pow(z,2), 0.5);

  state(i, j, k, URHO) = prob_parm.rho;
  state(i, j, k, UMX) = prob_parm.rho * prob_parm.vx_in;
  state(i, j, k, UMY) = prob_parm.rho * prob_parm.vy_in;
  state(i, j, k, UMZ) = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = prob_parm.rho * mfs[n];
#ifdef PELEC_USE_TWO_TEMP
  amrex::Real kB = 1.380649e-16; // erg/K
  amrex::Real me_g = 9.10938356e-28;         // electron mass (g)
  amrex::Real ne = state(i, j, k, UFS + 0) * (1.0/me_g);
  state(i, j, k, UFX+6) = 300.0;  // Te
  // Calculate mean electron energy as 3/2 * kB * Te * ne
  state(i, j, k, UFX+5) = 1.5 * kB * state(i, j, k, UFX+6) * ne;
  amrex::Real Pe = kB * state(i, j, k, UFX+6) * ne;

  auto eos = pele::physics::PhysicsType::eos();
  eos.RYP2E(prob_parm.rho, mfs, prob_parm.p, Pe, tempE);
  eos.RYP2T(prob_parm.rho, mfs, prob_parm.p, Pe, tempT);
#else
  auto eos = pele::physics::PhysicsType::eos();
  eos.PYT2RE(prob_parm.p, mfs, prob_parm.T ,rhonew , tempE);
#endif

  // Add energy from function
  yoLd = y/prob_parm.Ld;
  amrex::Real aparam = 100.0/2.3342;
  amrex::Real bparam = 1.0/2.3342;
  amrex::Real cparam = 1.0/2.3342;
  amrex::Real Efac = 0.0;

  Efac = aparam*pow(yoLd,4)+bparam*pow(yoLd,2)+cparam;
  

  Enew = prob_parm.Erho/(3.1415926535*prob_parm.r0*prob_parm.r0*prob_parm.Ld);

  // Use empirical function for energy deposition
  if ( prob_parm.Econst != 1){
     Enew = Efac*Enew;
  }
  
   tempE = tempE+(1/rhonew)*Enew*(1.0/(1.0+exp((r-prob_parm.r0)/prob_parm.sigma0)))*(1.0/(1+exp((abs(y-prob_parm.y0)-0.5*0.95*prob_parm.Ld)/prob_parm.sigma0)));
  eos.EY2T(tempE, mfs, tempT);
  state(i, j, k, UEINT) = rhonew * tempE;
  state(i, j, k, UEDEN) =
    rhonew *
    (tempE + 0.5 * (prob_parm.vx_in * prob_parm.vx_in +
                             prob_parm.vy_in * prob_parm.vy_in));

  state(i, j, k, UTEMP) = tempT;

  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = state(i, j, k, URHO) * mfs[n];


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
