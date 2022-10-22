#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include "PeleC.H"
#include "prob.H"

struct PCHypFillExtDir
{
  ProbParmDevice const* lprobparm;
  bool m_do_turb_inflow{false};

  AMREX_GPU_HOST
  constexpr explicit PCHypFillExtDir(
    const ProbParmDevice* d_prob_parm, const bool do_turb_inflow)
    : lprobparm(d_prob_parm), m_do_turb_inflow(do_turb_inflow)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& dest,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();
    const amrex::Real* prob_lo = geom.ProbLo();
    // const amrex::Real* prob_hi = geom.ProbHi();
    const amrex::Real* dx = geom.CellSize();
    const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
      prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
      prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
      prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};

    const int* bc = bcr->data();

    amrex::Real s_int[NVAR] = {0.0};
    amrex::Real s_ext[NVAR] = {0.0};

    // xlo and xhi
    int idir = 0;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      const amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      if (m_do_turb_inflow && (iv[idir] == domlo[idir] - 1)) {
        for (int n = 0; n < NVAR; n++) {
          s_ext[n] = dest(iv, n);
        }
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      const amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      if (m_do_turb_inflow && (iv[idir] == domlo[idir] - 1)) {
        for (int n = 0; n < NVAR; n++) {
          s_ext[n] = dest(iv, n);
        }
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      const amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      if (m_do_turb_inflow && (iv[idir] == domlo[idir] - 1)) {
        for (int n = 0; n < NVAR; n++) {
          s_ext[n] = dest(iv, n);
        }
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      const amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      if (m_do_turb_inflow && (iv[idir] == domhi[idir] + 1)) {
        for (int n = 0; n < NVAR; n++) {
          s_ext[n] = dest(iv, n);
        }
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      const amrex::IntVect loc(AMREX_D_DECL(iv[0], iv[1], domlo[idir]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      if (m_do_turb_inflow && (iv[idir] == domhi[idir] + 1)) {
        for (int n = 0; n < NVAR; n++) {
          s_ext[n] = dest(iv, n);
        }
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      const amrex::IntVect loc(AMREX_D_DECL(iv[0], iv[1], domhi[idir]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      if (m_do_turb_inflow && (iv[idir] == domhi[idir] + 1)) {
        for (int n = 0; n < NVAR; n++) {
          s_ext[n] = dest(iv, n);
        }
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
#endif
#endif

#ifdef PELEC_USE_PLASMA
    {
    // Try to catch the PhiV BCs.
    const int* bcp = bcr[UFX].data();
    // xlo and xhi
    idir = 0;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX) = s_ext[UFX];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX) = s_ext[UFX];
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX) = s_ext[UFX];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX) = s_ext[UFX];
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domlo[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX) = s_ext[UFX];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domhi[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX) = s_ext[UFX];
    }
#endif
#endif
    }

    {
    // Try to catch the nE BCs.
    const int* bcp = bcr[UFX+1].data();
    // xlo and xhi
    idir = 0;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+1) = s_ext[UFX+1];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+1) = s_ext[UFX+1];
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+1) = s_ext[UFX+1];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+1) = s_ext[UFX+1];
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domlo[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+1) = s_ext[UFX+1];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domhi[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+1) = s_ext[UFX+1];
    }
#endif
#endif
    }

#ifdef PELEC_USE_TWO_TEMP
    {
    // Try to catch the Uele BCs.
    const int* bcp = bcr[UFX+5].data();
    // xlo and xhi
    idir = 0;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+5) = s_ext[UFX+5];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+5) = s_ext[UFX+5];
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+5) = s_ext[UFX+5];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+5) = s_ext[UFX+5];
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domlo[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+5) = s_ext[UFX+5];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domhi[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+5) = s_ext[UFX+5];
    }
#endif
#endif
    }

    {
    // Try to catch the Tele BCs.
    const int* bcp = bcr[UFX+6].data();
    // xlo and xhi
    idir = 0;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+6) = s_ext[UFX+6];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+6) = s_ext[UFX+6];
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+6) = s_ext[UFX+6];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+6) = s_ext[UFX+6];
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bcp[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domlo[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom, *lprobparm);
      dest(iv, UFX+6) = s_ext[UFX+6];
    } else if (
      (bcp[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domhi[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom, *lprobparm);
      dest(iv, UFX+6) = s_ext[UFX+6];
    }
#endif
#endif
    }

#endif

#endif
  }
};

struct PCReactFillExtDir
{
  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& /*iv*/,
    amrex::Array4<amrex::Real> const& /*dest*/,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/,
    const amrex::BCRec* /*bcr*/,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
  }
};

void
pc_bcfill_hyp(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp)
{

  if (PeleC::turb_inflow.is_initialized()) {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      auto bndryBoxLO = amrex::Box(amrex::adjCellLo(geom.Domain(), dir) & bx);
      if (bcr[1].lo()[dir] == EXT_DIR && bndryBoxLO.ok()) {
        // Create box with ghost cells and set them to zero
        amrex::IntVect growVect(PeleC::numGrow());
        growVect[dir] = 0;
        const amrex::Box modDom = amrex::grow(geom.Domain(), growVect);
        const auto bndryBoxLO_ghost =
          amrex::Box(amrex::adjCellLo(modDom, dir) & bx);
        data.setVal<amrex::RunOn::Device>(
          0.0, bndryBoxLO_ghost, UMX, AMREX_SPACEDIM);

        PeleC::turb_inflow.add_turb(
          bndryBoxLO, data, UMX, geom, time, dir, amrex::Orientation::low);
      }

      auto bndryBoxHI = amrex::Box(amrex::adjCellHi(geom.Domain(), dir) & bx);
      if (bcr[1].hi()[dir] == EXT_DIR && bndryBoxHI.ok()) {
        // Create box with ghost cells and set them to zero
        amrex::IntVect growVect(PeleC::numGrow());
        growVect[dir] = 0;
        const amrex::Box modDom = amrex::grow(geom.Domain(), growVect);
        const auto bndryBoxHI_ghost =
          amrex::Box(amrex::adjCellHi(modDom, dir) & bx);
        data.setVal<amrex::RunOn::Device>(
          0.0, bndryBoxHI_ghost, UMX, AMREX_SPACEDIM);

        PeleC::turb_inflow.add_turb(
          bndryBoxHI, data, UMX, geom, time, dir, amrex::Orientation::high);
      }
    }
  }

  const ProbParmDevice* lprobparm = PeleC::d_prob_parm_device;
  amrex::GpuBndryFuncFab<PCHypFillExtDir> hyp_bndry_func(
    PCHypFillExtDir{lprobparm, PeleC::turb_inflow.is_initialized()});
  hyp_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}

void
pc_reactfill_hyp(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp)
{
  amrex::GpuBndryFuncFab<PCReactFillExtDir> react_bndry_func(
    PCReactFillExtDir{});
  react_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}

void
pc_nullfill(
  amrex::Box const& /*bx*/,
  amrex::FArrayBox& /*data*/,
  const int /*dcomp*/,
  const int /*numcomp*/,
  amrex::Geometry const& /*geom*/,
  const amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcr*/,
  const int /*bcomp*/,
  const int /*scomp*/)
{
}
