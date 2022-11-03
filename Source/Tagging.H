#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_FArrayBox.H>
#include <AMReX_TagBox.H>
#include "prob_parm.H"

struct TaggingParm
{
  amrex::Real denerr = 1.0e10;
  amrex::Real dengrad = 1.0e10;
  amrex::Real denratio = 1.0e10;
  amrex::Real presserr = 1.0e10;
  amrex::Real pressgrad = 1.0e10;
  amrex::Real velerr = 1.0e10;
  amrex::Real velgrad = 1.0e10;
  amrex::Real vorterr = 1.0e10;
  amrex::Real temperr = 1.0e10;
  amrex::Real lotemperr = -1.0e10;
  amrex::Real tempgrad = 1.0e10;
  amrex::Real ftracerr = 1.0e10;
  amrex::Real ftracgrad = 1.0e10;
  amrex::Real vfracerr = 1.0e10;
  amrex::Real detag_eb_factor = 3.0;

  std::string eb_refine_type = "static";

  int max_denerr_lev = 10;
  int max_dengrad_lev = 10;
  int max_denratio_lev = 10;
  int max_presserr_lev = 10;
  int max_pressgrad_lev = 10;
  int max_velerr_lev = 10;
  int max_velgrad_lev = 10;
  int max_vorterr_lev = 10;
  int max_temperr_lev = 10;
  int max_lotemperr_lev = 10;
  int max_tempgrad_lev = 10;
  int max_ftracerr_lev = 10;
  int max_ftracgrad_lev = 10;
  int max_vfracerr_lev = 10;

#ifdef PELEC_USE_PLASMA
  amrex::Real efielderr = 1.0e10;
  int max_efield_lev = 10;

  amrex::Real negraderr = 1.0e10;
  int max_negrad_lev = 10;

  amrex::Real neerr = 1.0e10;
  int max_ne_lev = 10;

  amrex::Real o4graderr = 1.0e10;
  int max_o4grad_lev = 10;

  amrex::Real efieldgraderr = 1.0e10;
  int max_efieldgrad_lev = 10;

  int plasma_channel_lev = 0;
  amrex::Real plasma_channel_top = 0.375;
  amrex::Real plasma_channel_bottom = 0.125;
  amrex::Real plasma_channel_r0 = 1.0;
  amrex::Real plasma_channel_rmax = 0.1;

  int pin_tip_lev = 0;
  amrex::Real pin_tip_rad = 0.05;
  amrex::Real pin_tip_h = 0.1;

  int vfrac_coarse = 10;
  amrex::Real vfrac_top = 1.0e10;
  amrex::Real vfrac_bottom = -1.0;

#ifdef PELEC_USE_TWO_TEMP
  amrex::Real Telegraderr = 1.0e10;
  int max_Telegrad_lev = 10;

  amrex::Real Teleerr = 1.0e10;
  int max_Tele_lev = 10;
#endif
#endif
  int max_eb_refine_lev = 10;
  int min_eb_refine_lev = -1;
  int adapt_eb_refined_lev = -1;
};

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_error(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fielderr,
  char tagval) noexcept
{
  // Tag on regions of high field values
  if (field(i, j, k) >= fielderr) {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_loerror(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fielderr,
  char tagval) noexcept
{
  // Tag on regions of low field values
  if (field(i, j, k) <= fielderr) {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_graderror(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fieldgrad,
  char tagval) noexcept
{
  // Tag on regions of high field gradient
  AMREX_D_TERM(
    amrex::Real ax = amrex::Math::abs(field(i + 1, j, k) - field(i, j, k));
    ax = amrex::max<amrex::Real>(
      ax, amrex::Math::abs(field(i, j, k) - field(i - 1, j, k)));
    , amrex::Real ay = amrex::Math::abs(field(i, j + 1, k) - field(i, j, k));
    ay = amrex::max<amrex::Real>(
      ay, amrex::Math::abs(field(i, j, k) - field(i, j - 1, k)));
    , amrex::Real az = amrex::Math::abs(field(i, j, k + 1) - field(i, j, k));
    az = amrex::max<amrex::Real>(
      az, amrex::Math::abs(field(i, j, k) - field(i, j, k - 1)));)
#if AMREX_SPACEDIM > 1
  if (amrex::max<amrex::Real>(AMREX_D_DECL(ax, ay, az)) >= fieldgrad)
#elif AMREX_SPACEDIM == 1
  if (ax >= fieldgrad)
#endif
  {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_ratioerror(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fieldratio,
  char tagval) noexcept
{
  // Tag on regions of high ratio of adacent cell values in field
  AMREX_D_TERM(
    amrex::Real axp1 = amrex::Math::abs(field(i + 1, j, k) / field(i, j, k));
    axp1 = amrex::max<amrex::Real>(axp1, 1.0 / axp1);
    amrex::Real axm1 = amrex::Math::abs(field(i - 1, j, k) / field(i, j, k));
    axm1 = amrex::max<amrex::Real>(axm1, 1.0 / axm1);
    amrex::Real ax = amrex::max<amrex::Real>(axm1, axp1);
    , amrex::Real ayp1 = amrex::Math::abs(field(i, j + 1, k) / field(i, j, k));
    ayp1 = amrex::max<amrex::Real>(ayp1, 1.0 / ayp1);
    amrex::Real aym1 = amrex::Math::abs(field(i, j - 1, k) / field(i, j, k));
    aym1 = amrex::max<amrex::Real>(aym1, 1.0 / aym1);
    amrex::Real ay = amrex::max<amrex::Real>(aym1, ayp1);
    , amrex::Real azp1 = amrex::Math::abs(field(i, j, k + 1) / field(i, j, k));
    azp1 = amrex::max<amrex::Real>(azp1, 1.0 / azp1);
    amrex::Real azm1 = amrex::Math::abs(field(i, j, k - 1) / field(i, j, k));
    azm1 = amrex::max<amrex::Real>(azm1, 1.0 / azm1);
    amrex::Real az = amrex::max<amrex::Real>(azm1, azp1);)
#if AMREX_SPACEDIM > 1
  if (amrex::max<amrex::Real>(AMREX_D_DECL(ax, ay, az)) >= fieldratio)
#elif AMREX_SPACEDIM == 1
  if (ax >= fieldratio)
#endif
  {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_abserror(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fielderr,
  char tagval) noexcept
{
  // Tag on regions of high field values
  if (amrex::Math::abs(field(i, j, k)) >= fielderr) {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_error_bounds(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real lbnd,
  const amrex::Real ubnd,
  char tagval) noexcept
{
  // Tag on regions inside bounds
  if ((lbnd < field(i, j, k)) && (field(i, j, k) < ubnd)) {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_plasma_channel(
  const int i,
  const int j,
  const int k,
  amrex::Real x,
  amrex::Real y,
  amrex::Real z, 
  amrex::Real channel_top,
  amrex::Real channel_bottom,
  amrex::Real channel_r0,
  amrex::Real channel_rmax,
  amrex::Array4<char> const& tag,
  char tagval) noexcept
{
  amrex::Real r = sqrt((x-channel_r0)*(x-channel_r0) + (z-channel_r0)*(z-channel_r0));
  if (r < channel_rmax && y > channel_bottom && y < channel_top)
  {
    tag(i, j, k) = tagval;
  }
  // amrex::Real r = sqrt((x-1.0)*(x-1.0) + (z-1.0)*(z-1.0));
  // if (r < 0.15 && y > 0.125 && y < 0.375)
  // {
  //   tag(i, j, k) = tagval;
  // }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_pin_tips(
  const int i,
  const int j,
  const int k,
  amrex::Real x,
  amrex::Real y,
  amrex::Real z, 
  amrex::Real channel_top,
  amrex::Real channel_bottom,
  amrex::Real channel_r0,
  amrex::Real pin_tip_rad,
  amrex::Real pin_tip_h,
  amrex::Array4<char> const& tag,
  char tagval) noexcept
{
  // Tag pin tip region (hard code for time being)
  amrex::Real r = sqrt((x-channel_r0)*(x-channel_r0) + (z-channel_r0)*(z-channel_r0));
  if (r < pin_tip_rad && ( (y > channel_bottom-0.005 && y < channel_bottom + pin_tip_h - 0.005) || (y > channel_top-pin_tip_h+0.005 && y < channel_top+0.005)) )
  {
    tag(i, j, k) = tagval;
  }
}

struct EmptyProbTagStruct
{
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void set_problem_tags(
    const int /*i*/,
    const int /*j*/,
    const int /*k*/,
    amrex::Array4<char> const& /*tag*/,
    amrex::Array4<amrex::Real const> const& /*field*/,
    char /*tagval*/,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*prob_lo*/,
    const amrex::Real /*time*/,
    const int /*level*/,
    ProbParmDevice const& /*d_prob_parm_device*/) noexcept
  {
  }
};

template <typename ProbTagStruct, typename... Args>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
set_problem_tags(Args&&... args)
{
  ProbTagStruct::set_problem_tags(std::forward<Args>(args)...);
}

#endif
