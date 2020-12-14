#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

#include "mechanism.h"

#include "EOS.H"
#include "prob_parm.H"
#include "prob.H"
#include "Transport.H"

namespace ProbParm {
AMREX_GPU_DEVICE_MANAGED amrex::Real p = 1013250.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real T = 0.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real rho = 0.00116;
AMREX_GPU_DEVICE_MANAGED amrex::Real eint = 0.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real vx_in = 9000.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real vy_in = 0.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real Re_L = 2500.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real Pr = 0.7;
AMREX_GPU_DEVICE_MANAGED amrex::Real PhiV_top = 0.0;
AMREX_GPU_DEVICE_MANAGED amrex::Real PhiV_bottom = 0.0;
AMREX_GPU_DEVICE_MANAGED amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {
  1.0};
} // namespace ProbParm

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
  pp.query("p", ProbParm::p);
  pp.query("rho", ProbParm::rho);
  pp.query("vx_in", ProbParm::vx_in);
  pp.query("vy_in", ProbParm::vy_in);
  pp.query("Re_L", ProbParm::Re_L);
  pp.query("Pr", ProbParm::Pr);
  pp.query("phiV_top", ProbParm::PhiV_top);
  pp.query("phiV_bottom", ProbParm::PhiV_bottom);

  amrex::Real L = (probhi[0] - problo[0]) * 0.2;

  amrex::Real cp = 0.0;
  // Air mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) ProbParm::massfrac[n] = 0.0;
  ProbParm::massfrac[1] = 0.233;
  ProbParm::massfrac[2] = 0.767;

  // Test mass fractions (equal mole fraction values)
  // ProbParm::massfrac[0] = 1.55946879584e-6;
  // ProbParm::massfrac[1] = 0.090894590354;
  // ProbParm::massfrac[2] = 0.0795738;
  // ProbParm::massfrac[3] = 0.04544576356920416;
  // ProbParm::massfrac[4] = 0.090894590354;
  // ProbParm::massfrac[5] = 0.0795738;
  // ProbParm::massfrac[6] = 0.18178762;
  // ProbParm::massfrac[7] = 0.1591460659;
  // ProbParm::massfrac[8] = 0.18178762;
  // ProbParm::massfrac[9] = 0.090894590354;

  EOS::RYP2E(
    ProbParm::rho, ProbParm::massfrac.begin(), ProbParm::p, ProbParm::eint);
  EOS::EY2T(ProbParm::eint, ProbParm::massfrac.begin(), ProbParm::T);
  EOS::TY2Cp(ProbParm::T, ProbParm::massfrac.begin(), cp);

 // transport_params::const_bulk_viscosity = 0.0;
 // transport_params::const_diffusivity = 0.0;
 // transport_params::const_viscosity =
 //   ProbParm::rho * ProbParm::vx_in * L / ProbParm::Re_L;
 // transport_params::const_conductivity =
 //   transport_params::const_viscosity * cp / ProbParm::Pr;
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
