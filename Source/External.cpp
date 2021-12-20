#include "PeleC.H"
#include "IndexDefines.H"

void
PeleC::construct_old_ext_source(amrex::Real time, amrex::Real dt)
{
  const amrex::MultiFab& S_old = get_old_data(State_Type);

  int ng = 0; // None filled

  old_sources[ext_src]->setVal(0.0);

  if (!add_ext_src) {
    return;
  }

  fill_ext_source(time, dt, S_old, S_old, *old_sources[ext_src], ng);

  old_sources[ext_src]->FillBoundary(geom.periodicity());
}

void
PeleC::construct_new_ext_source(amrex::Real time, amrex::Real dt)
{
  const amrex::MultiFab& S_old = get_old_data(State_Type);
  const amrex::MultiFab& S_new = get_new_data(State_Type);

  int ng = 0;

  new_sources[ext_src]->setVal(0.0);

  if (!add_ext_src) {
    return;
  }

  fill_ext_source(time, dt, S_old, S_new, *new_sources[ext_src], ng);
}

void
PeleC::fill_ext_source(
  amrex::Real time,
  amrex::Real dt,
  const amrex::MultiFab& state_old,
  const amrex::MultiFab& state_new,
  amrex::MultiFab& ext_src,
  int ng)
{
  // const amrex::Real* dx = geom.CellSize();
  // const amrex::Real* prob_lo = geom.ProbLo();

  amrex::Real prev_time = state[State_Type].prevTime();
  amrex::Real elemChrg = 1.60217662e-19;     //Coulomb per charge
  amrex::Real me_g = 9.10938356e-28;         // electron mass (g)


#ifdef PELEC_USE_EB
  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(state_old.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();
#endif

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(ext_src, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.growntilebox(ng);

#ifdef PELEC_USE_EB
    const auto& flag_fab = flags[mfi];
    amrex::FabType typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
      continue;
    }
#endif

    // auto const& So = state_old.array(mfi);
    auto const& S_arr = (time == prev_time) ? state_old.array(mfi):state_new.array(mfi);
    auto const& Farr = ext_src.array(mfi);
    auto const& joule_src = joule_heating.array(mfi);
    auto const& E_cc = Efield.array(mfi);
    auto const& ve = spec_drift.array(mfi, NUM_E * E_ID);
    auto const& K_cc = KSpec_old.array(mfi);


    // Evaluate the external source
    // Calculating joule heating source term: S_joule = -e * ne * u_e \dot E    [erg/cm3-s]
    amrex::ParallelFor(
      bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        // joule_src(i,j,k) = -1.0*elemChrg * S_arr(i,j,k,UFS+E_ID) * (ve(i,j,k,0)*E_cc(i,j,k,0) + ve(i,j,k,1)*E_cc(i,j,k,1) + ve(i,j,k,2)*E_cc(i,j,k,2));
        joule_src(i,j,k) = -(1.0/me_g)*elemChrg * S_arr(i,j,k,UFS+E_ID) * K_cc(i,j,k,E_ID) * (E_cc(i,j,k,0)*E_cc(i,j,k,0) + E_cc(i,j,k,1)*E_cc(i,j,k,1) + E_cc(i,j,k,2)*E_cc(i,j,k,2));
        Farr(i, j, k, UEDEN) = joule_src(i,j,k);
        Farr(i, j, k, UEINT) = joule_src(i,j,k);
        if(i == 1 && j == 1 && k == 1) printf("JOULE HEATING SRC IS %.6e, Eden = %.6e, Eint = %.6e, lterm = %.6e, sterm = %.6e\n", Farr(i,j,k,Eden), S_arr(i,j,k,Eden), S_arr(i,j,k,Eint), K_cc(i,j,k,E_ID) * (E_cc(i,j,k,0)*E_cc(i,j,k,0) + E_cc(i,j,k,1)*E_cc(i,j,k,1) + E_cc(i,j,k,2)*E_cc(i,j,k,2)), elemChrg * S_arr(i,j,k,UFS+E_ID));
    });
  }
}
