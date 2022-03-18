#include <PeleC.H>
#include "IndexDefines.H"
#include <Plasma_K.H>
#include <Plasma.H>



using namespace amrex;


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

  // Joule heating of the gas due to the movement of ions
  // Option to include portion of electron heating as well to mimic fast heating
  // Source term calculated as S_{jh} = - e \sum_k z_k \Gamma_k . E
  // Ion flux \Gamma_k = z_k mu_k n_k E - D dn_k/dx + n_k u

  amrex::Real prev_time = state[State_Type].prevTime();
  amrex::Real elemChrg = 1.60217662e-19;     //Coulomb per charge
  amrex::Real me_g = 9.10938356e-28;         // electron mass (g)
  const amrex::Real* dx = geom.CellSize();
  const amrex::Box& domain = geom.Domain();
  const amrex::BCRec& bcrec = get_desc_lst()[State_Type].getBC(PhiV);
  amrex::Real fluxE_x, fluxE_y, fluxE_z;
  amrex::Real mwt[NUM_SPECIES];
  auto eos = pele::physics::PhysicsType::eos();
  eos.molecular_weight(mwt);   // CGS

#ifdef PELEC_USE_EB
  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(state_old.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();
#endif

  // First need cell-centered gradients for all ions
  for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
  {
    const auto bc_lo = bcrec.lo(dir);
    const auto bc_hi = bcrec.hi(dir);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(spec_2ndo_gradients,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& ebx = mfi.tilebox();
        const Box& gbx = mfi.growntilebox(1);
        auto const& spec_ar = (time == prev_time) ? state_old.array(mfi,UFS):state_new.array(mfi,UFS);
        const auto grad_ar = spec_2ndo_gradients.array(mfi);
        amrex::ParallelFor(ebx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
           int idx[3] = {i,j,k};
           bool extdir_or_ho_lo = ( bc_lo == amrex::BCType::ext_dir ) || ( bc_lo == amrex::BCType::hoextrap );
           bool extdir_or_ho_hi = ( bc_hi == amrex::BCType::ext_dir ) || ( bc_hi == amrex::BCType::hoextrap );
           for(int n = 0; n < NUM_SPECIES; n++){
              if(zk_num[n] != 0){
                if(dir == 0){
                  grad_ar(i,j,k,3*n + 0) = amrex_calc_xslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
                } else if (dir == 1){
                  grad_ar(i,j,k,3*n + 1) = amrex_calc_yslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
                } else {
                  grad_ar(i,j,k,3*n + 2)  = amrex_calc_zslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
                }
              }
           }
        });
    }
  }

  // Calculate the joule heating term based on ion transport
  for (amrex::MFIter mfi(joule_heating, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& tbox = mfi.tilebox();
      auto const& Farr = ext_src.array(mfi);
      auto const& joule_src = joule_heating.array(mfi);
      auto const& S_arr = (time == prev_time) ? state_old.array(mfi):state_new.array(mfi);
      auto const& E_cc = Efield.array(mfi);
      auto const& K_cc = KSpec_old.array(mfi);
      auto const& grad_ar = spec_2ndo_gradients.array(mfi);
      auto const& coe_rhoD = coeffs_old.array(mfi,dComp_rhoD);
      amrex::ParallelFor(
        tbox, [=,&fluxE_x, &fluxE_y, &fluxE_z] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Calculate the current due to charged species fluxes
          joule_src(i,j,k) = 0.0;
          for(int n = 0; n<NUM_SPECIES; n++){
            if(zk_num[n] != 0){
              // Cell-centered flux dot Efield components (g-erg/cm3-C-s)
              // Recall K_cc includes charge sign already
              fluxE_x = ( (E_cc(i,j,k,0) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMX)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n)
                        - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 0) ) * E_cc(i,j,k,0);
              fluxE_y = ( (E_cc(i,j,k,1) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMY)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n)
                        - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 1) ) * E_cc(i,j,k,1);
              fluxE_z = ( (E_cc(i,j,k,2) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMZ)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n)
                        - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 2) ) * E_cc(i,j,k,2);

              // Calculate the total flux contribution (erg/cm3-s)
              if(n == E_ID){
                joule_src(i,j,k) += ef_electron_heating_pct * zk_num[n] * EFConst::elemCharge * (fluxE_x + fluxE_x + fluxE_x) * ( EFConst::Na / mwt[n]);
              }
              else{
                joule_src(i,j,k) += zk_num[n] * EFConst::elemCharge * (fluxE_x + fluxE_x + fluxE_x) * ( EFConst::Na / mwt[n]);
              }
            }
          }
          Farr(i, j, k, UEDEN) = joule_src(i,j,k);
          Farr(i, j, k, UEINT) = joule_src(i,j,k);
        });
  }
}
