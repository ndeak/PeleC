#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

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
  pp.query("p", PeleC::h_prob_parm_device->p);
  //pp.query("rho", PeleC::h_prob_parm_device->rho);
  pp.query("T", PeleC::h_prob_parm_device->T);
  //pp.query("eint", PeleC::h_prob_parm_device->eint);
  pp.query("vx_in", PeleC::h_prob_parm_device->vx_in);
  pp.query("vy_in", PeleC::h_prob_parm_device->vy_in);
  pp.query("Re_L", PeleC::h_prob_parm_device->Re_L);
  pp.query("Pr", PeleC::h_prob_parm_device->Pr);
  pp.query("phiV_top", PeleC::h_prob_parm_device->PhiV_top);
  pp.query("phiV_bottom", PeleC::h_prob_parm_device->PhiV_bottom);

  amrex::Real L = (probhi[0] - problo[0]) * 0.2;

  amrex::Real cp = 0.0;
  // Air mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) PeleC::h_prob_parm_device->massfrac[n] = 0.0;
  PeleC::h_prob_parm_device->massfrac[O2_ID] = 0.2;
  PeleC::h_prob_parm_device->massfrac[N2_ID] = 0.8;
//  PeleC::h_prob_parm_device->massfrac[40] = 2.869547e-02;//N2vib 1 
//  PeleC::h_prob_parm_device->massfrac[41] = 8.485209e-04;//N2vib 2
//  PeleC::h_prob_parm_device->massfrac[42] = 2.509064e-05;//N2vib 3
//  PeleC::h_prob_parm_device->massfrac[43] = 7.419264e-07;//N2vib 4
//  PeleC::h_prob_parm_device->massfrac[44] = 2.193865e-08;//N2vib 5

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

  auto eos = pele::physics::PhysicsType::eos();
  //eos.RYP2E(
  //  PeleC::h_prob_parm_device->rho, PeleC::h_prob_parm_device->massfrac.begin(), PeleC::h_prob_parm_device->p, PeleC::h_prob_parm_device->eint);
  //eos.EY2T(PeleC::h_prob_parm_device->eint, PeleC::h_prob_parm_device->massfrac.begin(), PeleC::h_prob_parm_device->T);
  //eos.TY2Cp(PeleC::h_prob_parm_device->T, PeleC::h_prob_parm_device->massfrac.begin(), cp);
  eos.PYT2RE(PeleC::h_prob_parm_device->p,PeleC::h_prob_parm_device->massfrac.begin() , PeleC::h_prob_parm_device->T ,PeleC::h_prob_parm_device->rho , PeleC::h_prob_parm_device->eint);

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
