#include <AMReX_PhysBCFunct.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <IndexDefines.H>
#include "prob.H"

using namespace amrex;

struct PhiVFill
{
    AMREX_GPU_DEVICE
    void operator() (const amrex::IntVect& iv, amrex::Array4<amrex::Real> const& dest,
                     const int dcomp, const int numcomp,
                     amrex::GeometryData const& geom, const amrex::Real time,
                     const amrex::BCRec* bcr, const int bcomp,
                     const int orig_comp) const
    {
       const int* domlo = geom.Domain().loVect();
       const int* domhi = geom.Domain().hiVect();
       const amrex::Real* dx = geom.CellSize();
       const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
         domlo[0] + (iv[0] + 0.5) * dx[0], domlo[1] + (iv[1] + 0.5) * dx[1],
         domlo[2] + (iv[2] + 0.5) * dx[2])};
       const int* bc = bcr->data();

       amrex::Real s_int[NVAR] = {0.0};
       amrex::Real s_ext[NVAR] = {0.0};

       for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
          if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
             bcnormal(x, s_int, s_ext, idir, +1, time, geom);
             dest(iv, dcomp) = s_ext[UFX];
          }
          if ((bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and (iv[idir] > domhi[idir])) {
             bcnormal(x, s_int, s_ext, idir, -1, time, geom);
             dest(iv, dcomp) = s_ext[UFX];
          }
       }
    }
};

struct nEFill
{
    AMREX_GPU_DEVICE
    void operator() (const amrex::IntVect& iv, amrex::Array4<amrex::Real> const& dest,
                     const int dcomp, const int numcomp,
                     amrex::GeometryData const& geom, const amrex::Real time,
                     const amrex::BCRec* bcr, const int bcomp,
                     const int orig_comp) const
    {
       const int* domlo = geom.Domain().loVect();
       const int* domhi = geom.Domain().hiVect();
       const amrex::Real* dx = geom.CellSize();
       const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
         domlo[0] + (iv[0] + 0.5) * dx[0], domlo[1] + (iv[1] + 0.5) * dx[1],
         domlo[2] + (iv[2] + 0.5) * dx[2])};
       const int* bc = bcr->data();

       amrex::Real s_int[NVAR] = {0.0};
       amrex::Real s_ext[NVAR] = {0.0};

       for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
          if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
             bcnormal(x, s_int, s_ext, idir, +1, time, geom);
             dest(iv, dcomp) = s_ext[UFX+1];
          }
          if ((bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and (iv[idir] > domhi[idir])) {
             bcnormal(x, s_int, s_ext, idir, -1, time, geom);
             dest(iv, dcomp) = s_ext[UFX+1];
          }
       }
    }
};

struct EfieldFillExtDir
{
  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& dest,
    const int dcomp,
    const int numcomp,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int bcomp,
    const int orig_comp) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();
    const amrex::Real* prob_lo = geom.ProbLo();
    const amrex::Real* prob_hi = geom.ProbHi();
    const amrex::Real* dx = geom.CellSize();
    const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
      prob_lo[0] + (iv[0] + 0.5) * dx[0], prob_lo[1] + (iv[1] + 0.5) * dx[1],
      prob_lo[2] + (iv[2] + 0.5) * dx[2])};

    const int* bc = bcr->data();

    amrex::Real s_int[NUM_E] = {0.0};
    amrex::Real s_ext[NUM_E] = {0.0};

    // xlo and xhi
    int idir = 0;
    if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NUM_E; n++) {
        s_int[n] = dest(loc, n);
      }
      ef_bcnormal(x, s_int, s_ext, idir, +1, time, geom);
      for (int n = 0; n < NUM_E; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NUM_E; n++) {
        s_int[n] = dest(loc, n);
      }
      ef_bcnormal(x, s_int, s_ext, idir, -1, time, geom);
      for (int n = 0; n < NUM_E; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NUM_E; n++) {
        s_int[n] = dest(loc, n);
      }
      ef_bcnormal(x, s_int, s_ext, idir, +1, time, geom);
      for (int n = 0; n < NUM_E; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NUM_E; n++) {
        s_int[n] = dest(loc, n);
      }
      ef_bcnormal(x, s_int, s_ext, idir, -1, time, geom);
      for (int n = 0; n < NUM_E; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      for (int n = 0; n < NUM_E; n++) {
        s_int[n] = dest(iv[0], iv[1], domlo[idir], n);
      }
      ef_bcnormal(x, s_int, s_ext, idir, +1, time, geom);
      for (int n = 0; n < NUM_E; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < NUM_E; n++) {
        s_int[n] = dest(iv[0], iv[1], domhi[idir], n);
      }
      ef_bcnormal(x, s_int, s_ext, idir, -1, time, geom);
      for (int n = 0; n < NUM_E; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
#endif
#endif
  }
};
