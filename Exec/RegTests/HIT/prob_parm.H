#ifndef PROB_PARM_H
#define PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice
{
  bool binfmt = false;
  bool restart = false;
  int inres = 0;
  amrex::Real uin_norm = 1.0;
  amrex::Real L_x = 0.0;
  amrex::Real L_y = 0.0;
  amrex::Real L_z = 0.0;
  amrex::Real Linput = 0.0;
  amrex::Real rho0 = 0.0;
  amrex::Real p = 1.013e6; // [erg cm^-3]
  amrex::Real T0 = 300.0;
  amrex::Real eint0 = 0.0;
  amrex::Real* d_xinput = nullptr;
  amrex::Real* d_uinput = nullptr;
  amrex::Real* d_vinput = nullptr;
  amrex::Real* d_winput = nullptr;
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
  amrex::Vector<amrex::Real> h_vinput;
  amrex::Vector<amrex::Real> h_winput;
  amrex::Vector<amrex::Real> h_xarray;
  amrex::Vector<amrex::Real> h_xdiff;
  amrex::Gpu::DeviceVector<amrex::Real> xinput;
  amrex::Gpu::DeviceVector<amrex::Real> uinput;
  amrex::Gpu::DeviceVector<amrex::Real> vinput;
  amrex::Gpu::DeviceVector<amrex::Real> winput;
  amrex::Gpu::DeviceVector<amrex::Real> xarray;
  amrex::Gpu::DeviceVector<amrex::Real> xdiff;
  ProbParmHost()
    : xinput(0), uinput(0), vinput(0), winput(0), xarray(0), xdiff(0)
  {
  }
};

#endif
