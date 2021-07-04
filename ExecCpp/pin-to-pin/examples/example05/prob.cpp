#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

#include "mechanism.h"

#include "EOS.H"
#include "prob_parm.H"
#include "prob.H"
#include "Transport.H"

void
pc_prob_close()
{
}

extern "C" {
void
amrex_probinit(
  const int* init,
  const int* name,
  const int* namelen,
  const amrex_real* problo,
  const amrex_real* probhi)
{
  // Parse params
  amrex::ParmParse pp("prob");
  pp.query("p", PeleC::prob_parm_device->p);
  pp.query("rho", PeleC::prob_parm_device->rho);
  pp.query("vx_in", PeleC::prob_parm_device->vx_in);
  pp.query("vy_in", PeleC::prob_parm_device->vy_in);
  pp.query("Re_L", PeleC::prob_parm_device->Re_L);
  pp.query("Pr", PeleC::prob_parm_device->Pr);
  pp.query("phiV_top", PeleC::prob_parm_device->PhiV_top);
  pp.query("phiV_bottom", PeleC::prob_parm_device->PhiV_bottom);

  amrex::Real L = (probhi[0] - problo[0]) * 0.2;

  amrex::Real cp = 0.0;
  // Air mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) PeleC::prob_parm_device->massfrac[n] = 0.0;
  PeleC::prob_parm_device->massfrac[1] = 0.233;
  PeleC::prob_parm_device->massfrac[2] = 0.767;

  // Test mass fractions (equal mole fraction values)
  // PeleC::prob_parm_device->massfrac[0] = 1.55946879584e-6;
  // PeleC::prob_parm_device->massfrac[1] = 0.090894590354;
  // PeleC::prob_parm_device->massfrac[2] = 0.0795738;
  // PeleC::prob_parm_device->massfrac[3] = 0.04544576356920416;
  // PeleC::prob_parm_device->massfrac[4] = 0.090894590354;
  // PeleC::prob_parm_device->massfrac[5] = 0.0795738;
  // PeleC::prob_parm_device->massfrac[6] = 0.18178762;
  // PeleC::prob_parm_device->massfrac[7] = 0.1591460659;
  // PeleC::prob_parm_device->massfrac[8] = 0.18178762;
  // PeleC::prob_parm_device->massfrac[9] = 0.090894590354;

  EOS::RYP2E(
    PeleC::prob_parm_device->rho, PeleC::prob_parm_device->massfrac.begin(), PeleC::prob_parm_device->p, PeleC::prob_parm_device->eint);
  EOS::EY2T(PeleC::prob_parm_device->eint, PeleC::prob_parm_device->massfrac.begin(), PeleC::prob_parm_device->T);
  EOS::TY2Cp(PeleC::prob_parm_device->T, PeleC::prob_parm_device->massfrac.begin(), cp);

 // transport_params::const_bulk_viscosity = 0.0;
 // transport_params::const_diffusivity = 0.0;
 // transport_params::const_viscosity =
 //   PeleC::prob_parm_device->rho * PeleC::prob_parm_device->vx_in * L / PeleC::prob_parm_device->Re_L;
 // transport_params::const_conductivity =
 //   transport_params::const_viscosity * cp / PeleC::prob_parm_device->Pr;
}
}

void
PeleC::problem_post_timestep()
{
}

void
PeleC::problem_post_init()
{
}

void
PeleC::problem_post_restart()
{
}
