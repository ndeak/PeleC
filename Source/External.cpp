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

  BL_PROFILE("PeleC::get_JH()");

  // Joule heating of the gas due to the movement of ions
  // Option to include portion of electron heating as well to mimic fast heating
  // Source term calculated as S_{jh} = - e \sum_k z_k \Gamma_k . E
  // Ion flux \Gamma_k = z_k mu_k n_k E - D dn_k/dx + n_k u
  //
  // TODO: should be updated to account for use of ambipolar diffusion model
  // (not doing for now since JH is negligible during interpulse period)

#ifdef PELEC_USE_PLASMA
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

  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(state_old.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();

//   // First need cell-centered gradients for all ions
//   for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
//   {
//     const auto bc_lo = bcrec.lo(dir);
//     const auto bc_hi = bcrec.hi(dir);
// #ifdef _OPENMP
// #pragma omp parallel if (Gpu::notInLaunchRegion())
// #endif
//     for (MFIter mfi(spec_2ndo_gradients,TilingIfNotGPU()); mfi.isValid(); ++mfi)
//     {
//         const Box& ebx = mfi.tilebox();
//         const Box& gbx = mfi.growntilebox(1);
//         auto const& spec_ar = (time == prev_time) ? state_old.array(mfi,UFS):state_new.array(mfi,UFS);
//         const auto grad_ar = spec_2ndo_gradients.array(mfi);
//         amrex::ParallelFor(ebx,
//         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
//         {
//            int idx[3] = {i,j,k};
//            bool extdir_or_ho_lo = ( bc_lo == amrex::BCType::ext_dir ) || ( bc_lo == amrex::BCType::hoextrap );
//            bool extdir_or_ho_hi = ( bc_hi == amrex::BCType::ext_dir ) || ( bc_hi == amrex::BCType::hoextrap );
//            for(int n = 0; n < NUM_SPECIES; n++){
//               if(zk_num[n] != 0){
//                 if(dir == 0){
//                   grad_ar(i,j,k,3*n + 0) = amrex_calc_xslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
//                 } else if (dir == 1){
//                   grad_ar(i,j,k,3*n + 1) = amrex_calc_yslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
//                 } else {
//                   grad_ar(i,j,k,3*n + 2)  = amrex_calc_zslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
//                 }
//               }
//            }
//         });
//     }
//   }
// 
//   // Calculate the joule heating term based on ion transport
//   for (amrex::MFIter mfi(joule_heating, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
//       const amrex::Box& tbox = mfi.tilebox();
//       auto const& Farr = ext_src.array(mfi);
//       auto const& joule_src = joule_heating.array(mfi);
//       auto const& S_arr = (time == prev_time) ? state_old.array(mfi):state_new.array(mfi);
//       auto const& E_cc = Efield.array(mfi);
//       auto const& K_cc = KSpec_old.array(mfi);
//       auto const& grad_ar = spec_2ndo_gradients.array(mfi);
//       auto const& coe_rhoD = coeffs_old.array(mfi,dComp_rhoD);
//       amrex::ParallelFor(
//         tbox, [=,&fluxE_x, &fluxE_y, &fluxE_z] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//           // Calculate the current due to charged species fluxes
//           joule_src(i,j,k) = 0.0;
//           for(int n = 0; n<NUM_SPECIES; n++){
//             if(zk_num[n] != 0){
//               // Cell-centered flux dot Efield components (g-erg/cm3-C-s)
//               // Recall K_cc includes charge sign already
//               fluxE_x = ( (E_cc(i,j,k,0) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMX)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n)
//                         - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 0) ) * E_cc(i,j,k,0);
//               fluxE_y = ( (E_cc(i,j,k,1) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMY)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n)
//                         - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 1) ) * E_cc(i,j,k,1);
//               fluxE_z = ( (E_cc(i,j,k,2) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMZ)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n)
//                         - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 2) ) * E_cc(i,j,k,2);
// 
//               // Calculate the total flux contribution (erg/cm3-s)
//               if(n == E_ID){
//                 joule_src(i,j,k) += ef_electron_heating_pct * zk_num[n] * EFConst::elemCharge * (fluxE_x + fluxE_x + fluxE_x) * ( EFConst::Na / mwt[n]);
//               }
//               else{
//                 joule_src(i,j,k) += zk_num[n] * EFConst::elemCharge * (fluxE_x + fluxE_x + fluxE_x) * ( EFConst::Na / mwt[n]);
//               }
//             }
//           }
//           Farr(i, j, k, UEDEN) = joule_src(i,j,k);
//           Farr(i, j, k, UEINT) = joule_src(i,j,k);
//         });
//   }
  

  // Coefficients for radiative heat loss fit
  // Data fit from "Net emission coefficient of air thermal plasmas" (2002), see Fig. 13
  // Data is in SI units (W/m3-sr), and the fit is for log10(E_loss)
  // Coefficients for R_p = 0
  double radiation_coeffs[] = {-4.69393371944433e-33, 6.42484192766636e-28, -3.61449522007878e-23, 1.07401256128979e-18, -1.80422980655138e-14, 1.71848147786282e-10, -9.38484312102902e-07, 0.00380695502932053, -4.03169557840341 };
  // Coefficients for R_p = 0.1 mm
  // double radiation_coeffs[] = { -5.37734547038786e-33, 8.05470506138091e-28, -5.05170461809753e-23, 1.71463710673129e-18, -3.39808585583430e-14, 3.96509823029999e-10, -2.64746628950791e-06, 0.00992474394418707, -11.9984373861060};

  // OLD joule heating code...
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(ext_src, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.growntilebox(ng);
    const auto& flag_fab = flags[mfi];
    amrex::FabType typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
      continue;
    }
    auto const& S_arr = (time == prev_time) ? state_old.array(mfi):state_new.array(mfi);
    auto const& Farr = ext_src.array(mfi);
    auto const& joule_src = joule_heating.array(mfi);
    auto const& radiation_src = radiative_losses.array(mfi);
    auto const& E_cc = Efield.array(mfi);
    auto const& ve = spec_drift.array(mfi, NUM_E * E_ID);
    auto const& K_cc = KSpec_old.array(mfi);


    // Evaluate the external source
    // Calculating joule heating source term: S_joule = -e * ne * u_e \dot E    [erg/cm3-s]
    amrex::ParallelFor(
      bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        if(ef_use_joule_heating){
          joule_src(i,j,k) = -(1.0/me_g)*elemChrg * S_arr(i,j,k,UFS+E_ID) * K_cc(i,j,k,E_ID) * (E_cc(i,j,k,0)*E_cc(i,j,k,0) + E_cc(i,j,k,1)*E_cc(i,j,k,1) + E_cc(i,j,k,2)*E_cc(i,j,k,2));
        }
        if(ef_use_radiative_losses){
          amrex::Real polytemp = 0.0;
          for(int n = 0; n<9; n++) polytemp += radiation_coeffs[n] * pow(S_arr(i,j,k,UTEMP),8-n);
          radiation_src(i,j,k) = pow(10,polytemp) * 10.0 * 4.0 * constants::PI();   // Factor to convert W/m3-sr -> erg/cm3
        }
#ifdef PELEC_USE_TWO_TEMP
        Farr(i, j, k, UEDEN) = -radiation_src(i,j,k);
        Farr(i, j, k, UEINT) = -radiation_src(i,j,k);
        Farr(i, j, k, Uele) = joule_src(i,j,k);
#else
        Farr(i, j, k, UEDEN) = joule_src(i,j,k) - radiation_src(i,j,k);
        Farr(i, j, k, UEINT) = joule_src(i,j,k) - radiation_src(i,j,k);
#endif
        // if(i == 1 && j == 1 && k == 1) printf("JOULE HEATING SRC IS %.6e, Eden = %.6e, Eint = %.6e, lterm = %.6e, sterm = %.6e\n", Farr(i,j,k,Eden), S_arr(i,j,k,Eden), S_arr(i,j,k,Eint), K_cc(i,j,k,E_ID) * (E_cc(i,j,k,0)*E_cc(i,j,k,0) + E_cc(i,j,k,1)*E_cc(i,j,k,1) + E_cc(i,j,k,2)*E_cc(i,j,k,2)), elemChrg * S_arr(i,j,k,UFS+E_ID));
    });
  }
#endif  
  amrex::Gpu::synchronize();
}
