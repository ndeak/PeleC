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
  amrex::Real tempE,rhonew,tempT,mfsum;
  for (int n = 0; n < NUM_SPECIES; n++)
    mfs[n] = 0.0;


  // Mass fractions corresponding to 1e3 1/cm3 ICs for electrons and O2+, in stoichiometric ethylene air
  mfs[0] = 7.7787881e-22;    // E
  mfs[O2_ID] = 0.218157027610499;
  mfs[N2_ID] = 0.78184297239;
  mfs[O2p_ID] = 9.0678200e-17;    // O4+


  //mfs[40] = 2.869547e-02;//N2vib 1 
  //mfs[41] = 8.485209e-04;//N2vib 2
  //mfs[42] = 2.509064e-05;//N2vib 3
  //mfs[43] = 7.419264e-07;//N2vib 4
  //mfs[44] = 2.193865e-08;//N2vib 5

  mfsum = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++)
    mfsum = mfsum + mfs[n];

  for (int n = 0; n < NUM_SPECIES; n++)
    mfs[n] = mfs[n]/mfsum;

//  mfs[21] = 0.063742756504943;
 // mfs[51] = 1.0e-15;    // O4+

  auto eos = pele::physics::PhysicsType::eos();
  //eos.RYP2E(prob_parm.rho, mfs, prob_parm.p, tempE);
  //eos.EY2T(tempE, mfs, tempT);
  //eos.RYP2T(prob_parm.rho, mfs, prob_parm.p, tempT);
  eos.PYT2RE(prob_parm.p, mfs, prob_parm.T ,rhonew , tempE);

  state(i, j, k, URHO) = rhonew;
  state(i, j, k, UMX) = rhonew * prob_parm.vx_in;
  state(i, j, k, UMY) = rhonew * prob_parm.vy_in;
  state(i, j, k, UMZ) = 0.0;
  state(i, j, k, UEINT) = rhonew * tempE;
  state(i, j, k, UEDEN) =
    rhonew *
    (tempE + 0.5 * (prob_parm.vx_in * prob_parm.vx_in +
                             prob_parm.vy_in * prob_parm.vy_in));
  state(i, j, k, UTEMP) = prob_parm.T;
  for (int n = 0; n < NUM_SPECIES; n++)
    state(i, j, k, UFS + n) = rhonew * mfs[n];

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
#ifdef PELEC_USE_TWO_TEMP
  amrex::Real kB = 1.380649e-16; // erg/K
  amrex::Real me_g = 9.10938356e-28;         // electron mass (g)
  amrex::Real ne = state(i, j, k, UFS + 0) * (1.0/me_g);
  state(i, j, k, UFX+6) = prob_parm.T;  // Te
  // Calculate mean electron energy as 3/2 * kB * Te * ne
  state(i, j, k, UFX+5) = 1.5 * kB * state(i, j, k, UFX+6) * ne;
  amrex::Real Pe = kB * state(i, j, k, UFX+6) * ne;
  eos.RYP2E(rhonew, mfs, prob_parm.p, Pe, tempE);
  eos.RYP2T(rhonew, mfs, prob_parm.p, Pe, tempT);

#endif
#endif
  state(i, j, k, UEINT) = rhonew * tempE;
  state(i, j, k, UEDEN) =
    rhonew *
    (tempE + 0.5 * (prob_parm.vx_in * prob_parm.vx_in +
                             prob_parm.vy_in * prob_parm.vy_in));
  state(i, j, k, UTEMP) = prob_parm.T;

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
  //amrex::Print() << "e_int BC  = " << prob_parm.eint << " \n ";
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

  #ifdef PELEC_USE_TWO_TEMP
     s_ext[UFX+5] = s_int[UFX+5];
     s_ext[UFX+6] = s_int[UFX+6];
  #endif

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
