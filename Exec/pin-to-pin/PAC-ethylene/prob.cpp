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
  pp.query("phifuel", PeleC::h_prob_parm_device->phifuel);
  pp.query("Re_L", PeleC::h_prob_parm_device->Re_L);
  pp.query("Pr", PeleC::h_prob_parm_device->Pr);
  pp.query("phiV_top", PeleC::h_prob_parm_device->PhiV_top);
  pp.query("phiV_bottom", PeleC::h_prob_parm_device->PhiV_bottom);

  amrex::Real L = (probhi[0] - problo[0]) * 0.2;

  amrex::Real cp = 0.0;
  // Air mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) PeleC::h_prob_parm_device->massfrac[n] = 0.0;

  // Calculate mole fracs from phifuel
  amrex::Real mO2 = 32.0; //  [g/mol]
  amrex::Real mN2 = 32.0; //  [g/mol]
  amrex::Real mfuel = 28.05; // [g/mol]
  
  amrex::Real Xfuel,Xair,XO2,XN2,mtot; 
 
  Xfuel = 1.0/(3.0/(0.21*PeleC::h_prob_parm_device->phifuel)+1.0);
  Xair =  1.0-Xfuel;
  XN2 = 0.79*Xair;
  XO2 = 0.21*Xair;
  mtot = mfuel*Xfuel+mO2*XO2+mN2*XN2;


  PeleC::h_prob_parm_device->massfrac[O2_ID] = mO2*XO2/mtot;
  PeleC::h_prob_parm_device->massfrac[N2_ID] = mN2*XN2/mtot;
  PeleC::h_prob_parm_device->massfrac[C2H4_ID] = mfuel*Xfuel/mtot;

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
