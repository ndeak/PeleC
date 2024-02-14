#ifndef _PROB_PARM_H_
#define _PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice
{
  bool binfmt = true;
  bool restart = false;
  int inres = 0;
  amrex::Real uin_norm = 1.0;
  amrex::Real L_x = 0.0;
  amrex::Real L_y = 0.0;
  amrex::Real L_z = 0.0;
  amrex::Real Linput = 0.0;
  amrex::Real rho = 0.0024006256882;
  amrex::Real p = 2773334.09306732; // [erg cm^-3]
  amrex::Real T = 400.0;
  amrex::Real eint = 0.0;
  amrex::Real* d_xinput = nullptr;
  amrex::Real* d_uinput = nullptr;
  amrex::Real* d_xarray = nullptr;
  amrex::Real* d_xdiff = nullptr;
  amrex::Real PhiV_top = 0.0;
  amrex::Real PhiV_bottom = 0.0;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {{1.0}};
};

struct ProbParmHost
{
  std::string iname;
  amrex::Vector<amrex::Real> h_xinput;
  amrex::Vector<amrex::Real> h_uinput;
  amrex::Vector<amrex::Real> h_xarray;
  amrex::Vector<amrex::Real> h_xdiff;
  amrex::Gpu::DeviceVector<amrex::Real> xinput;
  amrex::Gpu::DeviceVector<amrex::Real> uinput;
  amrex::Gpu::DeviceVector<amrex::Real> xarray;
  amrex::Gpu::DeviceVector<amrex::Real> xdiff;
  ProbParmHost()
    : xinput(0), uinput(0), xarray(0), xdiff()
  {
  }
};

#endif
