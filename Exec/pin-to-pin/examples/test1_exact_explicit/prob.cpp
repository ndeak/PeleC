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
  pp.query("rho", PeleC::h_prob_parm_device->rho);
  pp.query("T", PeleC::h_prob_parm_device->T);
  pp.query("vx_in", PeleC::h_prob_parm_device->vx_in);
  pp.query("vy_in", PeleC::h_prob_parm_device->vy_in);
  pp.query("Re_L", PeleC::h_prob_parm_device->Re_L);
  pp.query("Pr", PeleC::h_prob_parm_device->Pr);
  pp.query("n0", PeleC::h_prob_parm_device->n0);
  pp.query("phiV_top", PeleC::h_prob_parm_device->PhiV_top);
  pp.query("phiV_bottom", PeleC::h_prob_parm_device->PhiV_bottom);

  amrex::Real L = (probhi[0] - problo[0]) * 0.2;

  amrex::Real cp = 0.0;
  // Air mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) PeleC::h_prob_parm_device->massfrac[n] = 0.0;
  PeleC::h_prob_parm_device->massfrac[1] = 0.233;
  PeleC::h_prob_parm_device->massfrac[2] = 0.767;

  auto eos = pele::physics::PhysicsType::eos();
  eos.PYT2RE(PeleC::h_prob_parm_device->p, PeleC::h_prob_parm_device->massfrac.begin(), PeleC::h_prob_parm_device->T, PeleC::h_prob_parm_device->rho, PeleC::h_prob_parm_device->eint);
  eos.TY2Cp(PeleC::h_prob_parm_device->T, PeleC::h_prob_parm_device->massfrac.begin(), cp);
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
