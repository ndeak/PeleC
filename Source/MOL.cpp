#include "MOL.H"
#ifdef PELEC_USE_PLASMA
#include "PeleC.H"
#include <Plasma.H>
#endif
#include "Godunov.H"

void
pc_compute_hyp_mol_flux(
  const amrex::Box& cbox,
  const amrex::Array4<const amrex::Real>& q,
  const amrex::Array4<const amrex::Real>& qaux,
  const amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> area,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const int plm_iorder,
  const bool use_laxf_flux
#ifdef PELEC_USE_PLASMA
  ,
  const amrex::Array4<const amrex::Real>& s_cc,
  const amrex::Array4<const amrex::Real>& K_cc,
  const amrex::Array4<const amrex::Real>& E_cc,
  const amrex::Array4<amrex::Real>& drift_cc,
  const amrex::Array4<amrex::Real>& eon,
  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> E_edge,
  std::array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> ionFlux_arr,
  const amrex::Array4<amrex::Real>& ionFlux_eb_arr,
  const int* bcr,
  const amrex::Geometry& geom,
  const int do_harmonic,
  const int ion_bc_type,
  const int zero_bc_flux,
  const int zero_bc_grad,
  const int use_NL,
  const amrex::Real secondary_em_coef,
  const amrex::Real electron_emit_const,
  const int ef_do_drift,
  const amrex::Array4<amrex::Real>& coe_cc
#endif
  ,
  const amrex::Array4<amrex::EBCellFlag const>& flags,
  const EBBndryGeom* ebg,
  const int /*Nebg*/,
  amrex::Real* ebflux,
  const int nebflux)
{
  const int R_RHO = 0;
  const int R_UN = 1;
  const int R_UT1 = 2;
  const int R_UT2 = 3;
  const int R_P = 4;
  const int R_ADV = 5;
  const int R_Y = R_ADV + NUM_ADV;
  const int R_AUX = R_Y + NUM_SPECIES;
  const int R_LIN = R_AUX + NUM_AUX;
  const int R_NUM = 5 + NUM_SPECIES + NUM_ADV + NUM_LIN + NUM_AUX;
  const int bc_test_val = 1;

#ifdef PELEC_USE_PLASMA
  const int* domlo = geom.Domain().loVect();
  const int* domhi = geom.Domain().hiVect();
  double Te;
#endif

  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    amrex::FArrayBox dq_fab(cbox, QVAR, amrex::The_Async_Arena());
    auto const& dq = dq_fab.array();
    setV(cbox, QVAR, dq, 0.0);

    // dimensional indexing
    const amrex::GpuArray<const int, 3> bdim{
      {static_cast<int>(dir == 0), static_cast<int>(dir == 1),
       static_cast<int>(dir == 2)}};
    const amrex::GpuArray<const int, 3> q_idx{
      {bdim[0] * QU + bdim[1] * QV + bdim[2] * QW,
       bdim[0] * QV + bdim[1] * QU + bdim[2] * QU,
       bdim[0] * QW + bdim[1] * QW + bdim[2] * QV}};
    const amrex::GpuArray<const int, 3> f_idx{
      {bdim[0] * UMX + bdim[1] * UMY + bdim[2] * UMZ,
       bdim[0] * UMY + bdim[1] * UMX + bdim[2] * UMX,
       bdim[0] * UMZ + bdim[1] * UMZ + bdim[2] * UMY}};

    if (plm_iorder != 1) {
      amrex::ParallelFor(
        cbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          mol_slope(i, j, k, dir, q_idx, q, qaux, dq, flags);
        });
    }
    // ndeak note - box is contracted in the dir direction so we don't index out
    // ebox by defauly has 3 ghost cells in all directions
    const amrex::Box tbox = amrex::grow(cbox, dir, -1);
    const amrex::Box ebox = amrex::surroundingNodes(tbox, dir);
    amrex::ParallelFor(
      ebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
        const amrex::IntVect ivm(iv - amrex::IntVect::TheDimensionVector(dir));

        amrex::Real qtempl[R_NUM] = {0.0};
        qtempl[R_UN] =
          q(ivm, q_idx[0]) + 0.5 * ((dq(ivm, 1) - dq(ivm, 0)) / q(ivm, QRHO));
        qtempl[R_P] =
          q(ivm, QPRES) + 0.5 * (dq(ivm, 0) + dq(ivm, 1)) * qaux(ivm, QC);
        qtempl[R_UT1] = q(ivm, q_idx[1]) + 0.5 * dq(ivm, 2);
        qtempl[R_UT2] =
          AMREX_D_PICK(0.0, 0.0, q(ivm, q_idx[2]) + 0.5 * dq(ivm, 3));
        qtempl[R_RHO] = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] =
            q(ivm, QFS + n) * q(ivm, QRHO) +
            0.5 * (dq(ivm, QFS + n) +
                   q(ivm, QFS + n) * (dq(ivm, 0) + dq(ivm, 1)) / qaux(ivm, QC));
          qtempl[R_RHO] += qtempl[R_Y + n];
        }

        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] = qtempl[R_Y + n] / qtempl[R_RHO];
        }
#ifdef PELEC_USE_TWO_TEMP
          // FIXME: Using first order Godunov until bug fix for Ue slope
          amrex::Real uelel = q(ivm, QFX+5) + 0.0 * dq(ivm, QFX+5);
#endif

        amrex::Real qtempr[R_NUM] = {0.0};
        qtempr[R_UN] =
          q(iv, q_idx[0]) - 0.5 * ((dq(iv, 1) - dq(iv, 0)) / q(iv, QRHO));
        qtempr[R_P] =
          q(iv, QPRES) - 0.5 * (dq(iv, 0) + dq(iv, 1)) * qaux(iv, QC);
        qtempr[R_UT1] = q(iv, q_idx[1]) - 0.5 * dq(iv, 2);
        qtempr[R_UT2] =
          AMREX_D_PICK(0.0, 0.0, q(iv, q_idx[2]) - 0.5 * dq(iv, 3));
        qtempr[R_RHO] = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempr[R_Y + n] =
            q(iv, QFS + n) * q(iv, QRHO) -
            0.5 * (dq(iv, QFS + n) +
                   q(iv, QFS + n) * (dq(iv, 0) + dq(iv, 1)) / qaux(iv, QC));
          qtempr[R_RHO] += qtempr[R_Y + n];
        }
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempr[R_Y + n] = qtempr[R_Y + n] / qtempr[R_RHO];
        }
#ifdef PELEC_USE_TWO_TEMP
          // FIXME: Using first order Godunov until bug fix for Ue slope
          amrex::Real ueler = q(iv, QFX+5) - 0.0 * dq(iv, QFX+5);
#endif

        for (int n = 0; n < NUM_ADV; n++) {
          qtempl[R_ADV + n] = q(ivm, QFA + n) + 0.5 * dq(ivm, QFA + n);
          qtempr[R_ADV + n] = q(iv, QFA + n) - 0.5 * dq(iv, QFA + n);
        }
        for (int n = 0; n < NUM_AUX; n++) {
          qtempl[R_AUX + n] = q(ivm, QFX + n) + 0.5 * dq(ivm, QFX + n);
          qtempr[R_AUX + n] = q(iv, QFX + n) - 0.5 * dq(iv, QFX + n);
        }
        for (int n = 0; n < NUM_LIN; n++) {
          qtempl[R_LIN + n] = q(ivm, QLIN + n) + 0.5 * dq(ivm, QLIN + n);
          qtempr[R_LIN + n] = q(iv, QLIN + n) - 0.5 * dq(iv, QLIN + n);
        }
        const amrex::Real cavg = 0.5 * (qaux(iv, QC) + qaux(ivm, QC));

        amrex::Real spl[NUM_SPECIES];
        for (int n = 0; n < NUM_SPECIES; n++) {
          spl[n] = qtempl[R_Y + n];
        }

        amrex::Real spr[NUM_SPECIES];
        for (int n = 0; n < NUM_SPECIES; n++) {
          spr[n] = qtempr[R_Y + n];
        }

        // ndeak TODO: After calculation of edge states, need cell-edge effective velocity
        // Step 1: Get cell-centered mobilities
        // Step 2: Calculate cell-centered effective velocities
        // Step 3: Extrapolate to get cell-edge values
        
        amrex::Real drift_tmp[NUM_SPECIES] = {0.0};
#ifdef PELEC_USE_PLASMA
        // ndeak note - because ebox is contracted in the dir direction,
        // we do not index out when we access i-1, j-1, etc. 
  

        // get cell-edge mobilities for each species (includes charge sign)
        amrex::Real c[NUM_SPECIES];
        for(int n=0; n<NUM_SPECIES; n++)
          c[n] = 0.5 * (K_cc(iv,n) + K_cc(ivm,n));
        
        if(ef_do_drift){
          // Calculate the cell-edge drift velocity
          for(int n=0; n<NUM_SPECIES; n++){
            drift_tmp[n] = c[n] * E_edge[dir](iv);
          }

          // Store cell-center drift velocity for time step estimation
          for(int n=0; n<NUM_SPECIES; n++){
            drift_cc(iv, NUM_E*n + 0) = amrex::Math::abs(K_cc(iv, n) * E_cc(iv, 0));
            drift_cc(iv, NUM_E*n + 1) = amrex::Math::abs(K_cc(iv, n) * E_cc(iv, 1));
            drift_cc(iv, NUM_E*n + 2) = amrex::Math::abs(K_cc(iv, n) * E_cc(iv, 2));
          }
        }
#endif
        amrex::Real flux_tmp[NVAR] = {0.0};
        amrex::Real ustar = 0.0;

        if (!use_laxf_flux) {
          amrex::Real qint_iu = 0.0, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0,
                      tmp4 = 0.0;
          riemann(
            qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
            qtempl[R_P], spl, qtempr[R_RHO], qtempr[R_UN], qtempr[R_UT1],
            qtempr[R_UT2], qtempr[R_P], spr, bc_test_val, cavg, drift_tmp, ustar,
            flux_tmp[URHO], &flux_tmp[UFS], flux_tmp[f_idx[0]],
            flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
            flux_tmp[UEINT], qint_iu, tmp1, tmp2, tmp3, tmp4);
          const amrex::Real flxrho = flux_tmp[URHO];
          for (int n = 0; n < NUM_ADV; n++) {
            pc_cmpflx_passive(
              ustar, flxrho, qtempl[R_ADV + n], qtempr[R_ADV + n],
              flux_tmp[UFA + n]);
          }
#ifndef PELEC_USE_PLASMA
          // Don't want to advect potential or efield components
          for (int n = 0; n < NUM_AUX; n++) {
            pc_cmpflx_passive(
              ustar, flxrho, qtempl[R_AUX + n], qtempr[R_AUX + n],
              flux_tmp[UFX + n]);
          }
#endif
          for (int n = 0; n < NUM_LIN; n++) {
            pc_cmpflx_passive(
              ustar, qint_iu, qtempl[R_LIN + n], qtempr[R_LIN + n],
              flux_tmp[ULIN + n]);
          }
#ifdef PELEC_USE_PLASMA
          // TODO account for electrohydrodynamic force

          
#ifdef PELEC_USE_TWO_TEMP
          // Find the electron temperature at the cell edge
          amrex::Real Te_edge = 0.5 * (q(iv,UFX+6) + q(ivm,UFX+6));

          // Calculate electron energy flux based on electron flux
          // Assuming flux_Ue = flux_nE * (3/2) * kB * Te   =>  * (5/3 factor)
          // TODO: Is it ok to do this?
          flux_tmp[UFX+6] = (flux_tmp[UFS+E_ID] / EFConst::me_cgs) * EFConst::kB * Te_edge * (5.0/2.0);
#endif
#endif
        } else {
          // TODO: update LF flux to incorporate drift velocity
          amrex::Real maxeigval = 0.0;
          laxfriedrich_flux(
            qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
            qtempl[R_P], spl, qtempr[R_RHO], qtempr[R_UN], qtempr[R_UT1],
            qtempr[R_UT2], qtempr[R_P], spr, bc_test_val, cavg, ustar,
            maxeigval, flux_tmp[URHO], &flux_tmp[UFS], flux_tmp[f_idx[0]],
            flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
            flux_tmp[UEINT]);
          const amrex::Real ul = qtempl[R_UN];
          const amrex::Real ur = qtempr[R_UN];
          const amrex::Real rl = qtempl[R_RHO];
          const amrex::Real rr = qtempr[R_RHO];
          for (int n = 0; n < NUM_ADV; n++) {
            pc_lax_cmpflx_passive(
              ul, ur, rl, rr, qtempl[R_ADV + n], qtempr[R_ADV + n], maxeigval,
              flux_tmp[UFA + n]);
          }
          for (int n = 0; n < NUM_AUX; n++) {
            pc_lax_cmpflx_passive(
              ul, ur, rl, rr, qtempl[R_AUX + n], qtempr[R_AUX + n], maxeigval,
              flux_tmp[UFX + n]);
          }
          for (int n = 0; n < NUM_LIN; n++) {
            pc_lax_cmpflx_passive(
              ul, ur, 1., 1., qtempl[R_LIN + n], qtempr[R_LIN + n], maxeigval,
              flux_tmp[ULIN + n]);
          }
        }
        for (int ivar = 0; ivar < NVAR; ivar++) {
          flx[dir](iv, ivar) += flux_tmp[ivar] * area[dir](iv);
        }

#ifdef PELEC_USE_PLASMA 
        // Overwrite species fluxes at the electrode boundaries
        // Calculate number density at the interior cell (0th order approx for now)
        // assumes Y_k at ghost cell is equal to interior value at ext_dir boundary,
        // so doesn't matter which species array we take from for now
        // TODO: Make sure other flux values are updated as well, if necessary

        amrex::Real ndens = 0.0;
        double EoN, Te;
        amrex::Real mwt[NUM_SPECIES];
        auto eos = pele::physics::PhysicsType::eos();
        eos.molecular_weight(mwt);
        amrex::Real ionFlux = 0.0;

        // Need to save down ion flux to boundary if using nonlinear coupled system solve
        if (use_NL) {
           ionFlux_arr[dir](iv) = 0.0;
        }

        // overwrite fluxes on all ext_dir boundaries
        if ((bcr[dir] == amrex::BCType::ext_dir) and (iv[dir] == domlo[dir])) {
          // Use EoN to get Te for electron flux at the boundary
          ExtrapTe(eon(iv, 0), &Te);
          if(!zero_bc_flux){
            flx[dir](iv, URHO) = 0.0;
            for(int n=0; n<NUM_SPECIES; n++){
                flx[dir](iv, UFS + n) = 0.0;
                if(zero_bc_grad == 1){
                  flx[dir](iv,UFS+n) = qtempr[R_RHO] * spr[n] * c[n] * E_edge[dir](iv) * area[dir](iv);
#ifdef PELEC_USE_TWO_TEMP
                  if(n == E_ID) flx[dir](iv,UFX+5) = (5.0/3.0) * ueler * c[n] * E_edge[dir](iv) * area[dir](iv);
#endif
                }
                else{
                  if(n == E_ID && !use_NL){
                    flx[dir](iv, UFS + n) = -0.5 * qtempr[R_RHO] * spr[n] * pow( (8.0*EFConst::kB*Te) / (EFConst::me_cgs * constants::PI()) ,0.5) * area[dir](iv);
                  }
                  if(n != E_ID && K_cc(i,j,k,n) != 0){
                    if(ion_bc_type == 0){
                      flx[dir](iv, UFS + n) = -0.5 * qtempr[R_RHO] * spr[n] * pow( (8.0*EFConst::kB*q(iv,QTEMP)) / ((mwt[n]/EFConst::Na) * constants::PI()) ,0.5) * area[dir](iv);
                    }
                    else if(ion_bc_type == 1){
                      if((K_cc(iv,n) < 0 && E_edge[dir](iv) > 0) || (K_cc(iv,n) > 0 && E_edge[dir](iv) < 0)){
                        flx[dir](iv, UFS + n) = qtempr[R_RHO] * spr[n] * c[n] * E_edge[dir](iv) * area[dir](iv);
                      }
                      else{
                        flx[dir](iv, UFS + n) = 0.0;
                      }
                    }
                    else{
                      printf("Ion BC type not supported!\n");
                      exit(1);
                    }
                    // Save ion flux for secondary electron emissions and convert to number density
                    if ( use_NL ) {
                       ionFlux_arr[dir](iv) += flx[dir](iv, UFS + n) / mwt[n] * EFConst::Na;
                    } else {
                       ionFlux += flx[dir](iv, UFS + n) / mwt[n] * EFConst::Na;
                    }
                  }
                }
                flx[dir](iv, URHO) += flx[dir](iv, UFS + n);
            }
            // Subtrat from source since ionFlux is negative and contribution should be positive
            if (!use_NL && !zero_bc_grad) flx[dir](iv, UFS + E_ID) -= 2.0 * secondary_em_coef * ionFlux * EFConst::me_cgs;
          }
        }
        if ((bcr[dir+AMREX_SPACEDIM] == amrex::BCType::ext_dir) and (iv[dir] == domhi[dir]+1)) {
          ExtrapTe(eon(ivm, 0), &Te);
          if(!zero_bc_flux){
            flx[dir](iv, URHO) = 0.0;
            for(int n=0; n<NUM_SPECIES; n++){
                flx[dir](iv, UFS + n) = 0.0;
                if(zero_bc_grad == 1){
                  flx[dir](iv,UFS+n) = qtempl[R_RHO] * spl[n] * c[n] * E_edge[dir](iv) * area[dir](iv);
#ifdef PELEC_USE_TWO_TEMP
                  if(n == E_ID) flx[dir](iv,UFX+5) = (5.0/3.0) * uelel * c[n] * E_edge[dir](iv) * area[dir](iv);
#endif
                }
                else{
                  if(zero_bc_grad == 2){    // Assume cathode is on domlo
                    flx[dir](iv,UFS+n) = qtempr[R_RHO] * spr[n] * c[n] * E_edge[dir](iv) * area[dir](iv);
                  }
                  else{
                    if(n == E_ID && !use_NL){
                      flx[dir](iv, UFS + n) = 0.5 * qtempl[R_RHO] * spl[n] * pow( (8.0*EFConst::kB*Te) / (EFConst::me_cgs * constants::PI()) ,0.5) * area[dir](iv);
                    }
                    if(n != E_ID && K_cc(iv,n) != 0){
                      if(ion_bc_type == 0){
                        flx[dir](iv, UFS + n) = 0.5 * qtempl[R_RHO] * spl[n] * pow( (8.0*EFConst::kB*q(ivm,QTEMP)) / ((mwt[n]/EFConst::Na) * constants::PI()) ,0.5) * area[dir](iv);
                      }
                      else if(ion_bc_type == 1){
                        if((K_cc(iv,n) < 0 && E_edge[dir](iv) < 0) || (K_cc(iv,n) > 0 && E_edge[dir](iv) > 0)){
                          flx[dir](iv, UFS + n) = qtempl[R_RHO] * spl[n] * c[n] * E_edge[dir](iv) * area[dir](iv);
                        }
                        else{
                          flx[dir](iv, UFS + n) = 0.0;
                        }
                      }
                      else{
                        printf("Ion BC type not supported!\n");
                        exit(1);
                      }
                      // Save ion flux for secondary electron emissions and convert to number density
                      if ( use_NL ) {
                         ionFlux_arr[dir](iv) += flx[dir](iv, UFS + n) / mwt[n] * EFConst::Na;
                      } else {
                         ionFlux += flx[dir](iv, UFS + n) / mwt[n] * EFConst::Na;
                      }
                    }
                  }
                }
                flx[dir](iv, URHO) += flx[dir](iv, UFS + n);
            }

            // Add on secondary electron emission based on ion fluxes
            // It is assumed that electrode boundary is an absolutely absorbing wall
            // Subtract from flux, since ion flux is positive (out of the domain), and a negative flux means a positive electron contribution
            if (!use_NL && !zero_bc_grad) flx[dir](iv, UFS + E_ID) -= 2.0 * secondary_em_coef * ionFlux * EFConst::me_cgs;

            // Imposed cathode flux (used to test space charge-induced electric field calculations) 
            // Subtracted from flux to ensure electrons move into the domain
            // electron_emit_const provided in [1/cm3]
            if (!use_NL && !zero_bc_grad) flx[dir](iv, UFS + E_ID) -= electron_emit_const * 0.5 * (pow( (8.0*EFConst::kB*Te) / (EFConst::me_cgs * constants::PI()) ,0.5)) * EFConst::me_cgs * area[dir](iv);
          }
        }
#endif

      });
  }

  // nextra was 3 for EB in PeleC but we are operating on a different
  // box here, so this should be zero.
  const int nextra = 0;

  const amrex::Real full_area = std::pow(del[0], AMREX_SPACEDIM - 1);
  const auto lo = amrex::lbound(cbox);
  const auto hi = amrex::ubound(cbox);
  const amrex::Real* dx      = geom.CellSize();
  const amrex::Real* problo  = geom.ProbLo();
  const amrex::Real* probhi  = geom.ProbHi();
  const amrex::Box bxg = amrex::grow(cbox, nextra - 1);

  amrex::ParallelFor(nebflux, [=] AMREX_GPU_DEVICE(int L) {
    const amrex::IntVect& iv = ebg[L].iv;
    AMREX_D_TERM(const int i = ebg[L].iv[0];, const int j = ebg[L].iv[1];
                 , const int k = ebg[L].iv[2];)
    if (bxg.contains(iv)) {
      amrex::Real ebnorm[AMREX_SPACEDIM] = {AMREX_D_DECL(
        ebg[L].eb_normal[0], ebg[L].eb_normal[1], ebg[L].eb_normal[2])};
      const amrex::Real ebnorm_mag = std::sqrt(AMREX_D_TERM(
        ebnorm[0] * ebnorm[0], +ebnorm[1] * ebnorm[1], +ebnorm[2] * ebnorm[2]));
      for (amrex::Real& dir : ebnorm) {
        dir /= ebnorm_mag;
      }
      amrex::Real y = problo[1] + (j + 0.5)*dx[1];


      amrex::Real flux_tmp[NVAR] = {0.0};
      AMREX_D_TERM(flux_tmp[UMX] = -q(iv, QPRES) * ebnorm[0];
                   , flux_tmp[UMY] = -q(iv, QPRES) * ebnorm[1];
                   , flux_tmp[UMZ] = -q(iv, QPRES) * ebnorm[2];)

#ifdef PELEC_USE_PLASMA
      // Overwrite species fluxes at the EB electrode boundaries
      // Currently, cell-centered values are used to approximate 
      // values at the EB face
      // TODO - may need to use better EB face approx
      // TODO figure out efield logic for strong ion BCs

      if (use_NL) {
         ionFlux_eb_arr(iv) = 0.0;
      }

      amrex::Real ndens = 0.0;
      double EoN, Te;
      amrex::Real mwt[NUM_SPECIES];
      auto eos = pele::physics::PhysicsType::eos();
      eos.molecular_weight(mwt);
      amrex::Real ionFlux = 0.0;

      // Calculate the electric field normal to the EB face (pointing into the fluid, negative value indicates into the surface)
      amrex::Real Enorm = (s_cc(iv, UFX+2) * ebnorm[0] + s_cc(iv, UFX+3) * ebnorm[1] + s_cc(iv, UFX+4) * ebnorm[2]);

      // overwrite fluxes on all ext_dir boundaries
      // Use EoN to get Te for electron flux at the boundary
      ExtrapTe(eon(iv, 0), &Te);
      if(!zero_bc_flux){
        flux_tmp[URHO] = 0.0;
        for(int n=0; n<NUM_SPECIES; n++){
            flux_tmp[UFS + n] = 0.0;
            if(zero_bc_grad == 1){
                flux_tmp[UFS + n] = q(iv,QRHO) * q(iv,QFS + n) * K_cc(iv,n) * Enorm;
#ifdef PELEC_USE_TWO_TEMP
                if(n == E_ID) flux_tmp[UFX + 5] = (5.0/3.0) * q(iv,QFX+5) * K_cc(iv,n) * Enorm;
#endif
            }
            else{
              if(zero_bc_grad == 2 && y >= probhi[1] / 2.0){
                  flux_tmp[UFS + n] = q(iv,QRHO) * q(iv,QFS + n) * K_cc(iv,n) * Enorm;
              }
              else{
                if(n == E_ID){
                  flux_tmp[UFS + n] = -0.5 * q(iv,QRHO) * q(iv,  QFS + n) * pow( (8.0*EFConst::kB*Te) / (EFConst::me_cgs * constants::PI()) ,0.5);
                }
                if(n != E_ID && K_cc(iv,n) != 0){
                  if(ion_bc_type == 0){
                    flux_tmp[UFS + n] = -0.5 * q(iv,QRHO) * q(iv,QFS + n) * pow( (8.0*EFConst::kB*q(iv,QTEMP)) / ((mwt[n]/EFConst::Na) * constants::PI()) ,0.5);
                  }
                  else if(ion_bc_type == 1){
                    if((K_cc(iv,n) < 0 && Enorm > 0) || (K_cc(iv,n) > 0 && Enorm < 0)){
                      flux_tmp[UFS + n] = q(iv,QRHO) * q(iv,QFS + n) * K_cc(iv,n) * Enorm;
                    }
                    else{
                      flux_tmp[UFS + n] = 0.0;
                    }
                  }
                  else{
                    printf("Ion BC type not supported!\n");
                    exit(1);
                  }
                  // Save ion flux for secondary electron emissions and convert to number density
                  if ( use_NL ) {
                    ionFlux_eb_arr(iv) += flux_tmp[UFS + n] / mwt[n] * EFConst::Na;
                  } else{
                    ionFlux += flux_tmp[UFS + n] / mwt[n] * EFConst::Na;
                  }
                }
              }
            }
            flux_tmp[URHO] += flux_tmp[UFS + n];
        }
        // Negative sign, since flux contribution should be opposite sign from the ionFlux is
        // Only worrying about SE if BC is not zero grad.
        if(zero_bc_grad == 0 || (zero_bc_grad == 2 && y <= probhi[1] / 2.0)) flux_tmp[UFS + E_ID] -= 2.0 * secondary_em_coef * ionFlux * EFConst::me_cgs;
        // flux_tmp is directed into the EB, so positive values imply electrode losses, and vice versa
        for(int n = 0; n<NUM_SPECIES; n++) flux_tmp[UFS + n] *= -1.0;
#ifdef PELEC_USE_TWO_TEMP
        flux_tmp[UFX+5] *= -1.0;
#endif
      }
#endif
  
      // Copy result into ebflux vector. Being a bit chicken here and only
      // copy values where ebg % iv is within box
      for (int n = 0; n < NVAR; n++) {
        ebflux[n * nebflux + L] += flux_tmp[n] * ebg[L].eb_area * full_area;
      }
    }
  });
}
