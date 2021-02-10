#include "MOL.H"
#ifdef PELEC_USE_PLASMA
#include "mechanism.h"
#endif

void
pc_compute_hyp_mol_flux(
  const amrex::Box& cbox,
  const amrex::Array4<const amrex::Real>& q,
  const amrex::Array4<const amrex::Real>& qaux,
  const amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>
    a,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
#ifdef PELEC_USE_EB
    del
#endif
  ,
  const int plm_iorder
#ifdef PELEC_USE_PLASMA
  ,
  const amrex::Array4<const amrex::Real>& K_cc,
  const amrex::Array4<const amrex::Real>& E_cc,
  const amrex::Array4<amrex::Real>& drift_cc,
  const amrex::Array4<amrex::Real>& eon,
  std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> E_edge,
  const int* bcr,
  const amrex::Geometry& geom,
  const int do_harmonic,
  const int ion_bc_type,
  const amrex::Real secondary_em_coef
#endif
#ifdef PELEC_USE_EB
  ,
  const amrex::Real eb_small_vfrac,
  const amrex::Array4<const amrex::Real>& vfrac,
  const amrex::Array4<amrex::EBCellFlag const>& flags,
  const EBBndryGeom* ebg,
  const int /*Nebg*/,
  amrex::Real* ebflux,
  const int nebflux
#endif
)
{
  const int R_RHO = 0;
  const int R_UN = 1;
  const int R_UT1 = 2;
  const int R_UT2 = 3;
  const int R_P = 4;
  const int R_Y = 5;
  const int bc_test_val = 1;

#ifdef PELEC_USE_PLASMA
  const int* domlo = geom.Domain().loVect();
  const int* domhi = geom.Domain().hiVect();
  double Te;
#endif

  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    amrex::FArrayBox dq_fab(cbox, QVAR);
    amrex::Elixir dq_fab_eli = dq_fab.elixir();
    auto const& dq = dq_fab.array();
    setV(cbox, QVAR, dq, 0.0);

    // dimensional indexing
    const amrex::GpuArray<const int, 3> bdim{{dir == 0, dir == 1, dir == 2}};
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
          mol_slope(
            i, j, k, bdim, q_idx, q, qaux, dq
#ifdef PELEC_USE_EB
            ,
            flags
#endif
          );
        });
    }
    // ndeak note - box is contracted in the dir direction so we don't index out
    // ebox by defauly has 3 ghost cells in all directions
    const amrex::Box tbox = amrex::grow(cbox, dir, -1);
    const amrex::Box ebox = amrex::surroundingNodes(tbox, dir);
    amrex::ParallelFor(
      ebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const int ii = i - bdim[0];
        const int jj = j - bdim[1];
        const int kk = k - bdim[2];

        // ndeak printing 
        // printf("Ex(%i, %i) = %.6e,\t Ey(%i, %i) = %.6e\n", i, j, E_cc(i,j,k,0), i, j, E_cc(i,j,k,1));
        // for(int n=0; n<NUM_SPECIES; n++) printf("mu(%i, %i, %i) = %.6e\n", i, j, n, E_cc(i,j,k,n));

        amrex::Real qtempl[5 + NUM_SPECIES] = {0.0};
        qtempl[R_UN] =
          q(ii, jj, kk, q_idx[0]) +
          0.5 * ((dq(ii, jj, kk, 1) - dq(ii, jj, kk, 0)) / q(ii, jj, kk, QRHO));
        qtempl[R_P] =
          q(ii, jj, kk, QPRES) +
          0.5 * (dq(ii, jj, kk, 0) + dq(ii, jj, kk, 1)) * qaux(ii, jj, kk, QC);
        qtempl[R_UT1] = q(ii, jj, kk, q_idx[1]) + 0.5 * dq(ii, jj, kk, 2);
        qtempl[R_UT2] = q(ii, jj, kk, q_idx[2]) + 0.5 * dq(ii, jj, kk, 3);
        qtempl[R_RHO] = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] = q(ii, jj, kk, QFS + n) * q(ii, jj, kk, QRHO) +
                            0.5 * (dq(ii, jj, kk, 4 + n) +
                                   q(ii, jj, kk, QFS + n) *
                                     (dq(ii, jj, kk, 0) + dq(ii, jj, kk, 1)) /
                                     qaux(ii, jj, kk, QC));
          qtempl[R_RHO] += qtempl[R_Y + n];
        }

        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] = qtempl[R_Y + n] / qtempl[R_RHO];
        }

        amrex::Real qtempr[5 + NUM_SPECIES] = {0.0};
        qtempr[R_UN] =
          q(i, j, k, q_idx[0]) -
          0.5 * ((dq(i, j, k, 1) - dq(i, j, k, 0)) / q(i, j, k, QRHO));
        qtempr[R_P] = q(i, j, k, QPRES) - 0.5 *
                                            (dq(i, j, k, 0) + dq(i, j, k, 1)) *
                                            qaux(i, j, k, QC);
        qtempr[R_UT1] = q(i, j, k, q_idx[1]) - 0.5 * dq(i, j, k, 2);
        qtempr[R_UT2] = q(i, j, k, q_idx[2]) - 0.5 * dq(i, j, k, 3);
        qtempr[R_RHO] = 0.0;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempr[R_Y + n] =
            q(i, j, k, QFS + n) * q(i, j, k, QRHO) -
            0.5 * (dq(i, j, k, 4 + n) + q(i, j, k, QFS + n) *
                                          (dq(i, j, k, 0) + dq(i, j, k, 1)) /
                                          qaux(i, j, k, QC));
          qtempr[R_RHO] += qtempr[R_Y + n];
        }
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempr[R_Y + n] = qtempr[R_Y + n] / qtempr[R_RHO];
        }

        const amrex::Real cavg =
          0.5 * (qaux(i, j, k, QC) + qaux(ii, jj, kk, QC));
        const amrex::Real csmall = amrex::min<amrex::Real>(
          qaux(i, j, k, QCSML), qaux(ii, jj, kk, QCSML));

        amrex::Real eos_state_rho;
        amrex::Real eos_state_p;
        amrex::Real eos_state_e;
        amrex::Real eos_state_cs;
        amrex::Real eos_state_gamma;
        amrex::Real eos_state_T;

        eos_state_rho = qtempl[R_RHO];
        eos_state_p = qtempl[R_P];
        amrex::Real spl[NUM_SPECIES];
        for (int n = 0; n < NUM_SPECIES; n++) {
          spl[n] = qtempl[R_Y + n];
        }
        EOS::RYP2T(eos_state_rho, spl, eos_state_p, eos_state_T);
        EOS::RYP2E(eos_state_rho, spl, eos_state_p, eos_state_e);
        EOS::TY2G(eos_state_T, spl, eos_state_gamma);
        EOS::RPY2Cs(eos_state_rho, eos_state_p, spl, eos_state_cs);
        const amrex::Real rhoe_l = eos_state_rho * eos_state_e;
        const amrex::Real gamc_l = eos_state_gamma;

        eos_state_rho = qtempr[R_RHO];
        eos_state_p = qtempr[R_P];
        amrex::Real spr[NUM_SPECIES];
        for (int n = 0; n < NUM_SPECIES; n++) {
          spr[n] = qtempr[R_Y + n];
        }
        EOS::RYP2T(eos_state_rho, spr, eos_state_p, eos_state_T);
        EOS::RYP2E(eos_state_rho, spr, eos_state_p, eos_state_e);
        EOS::TY2G(eos_state_T, spr, eos_state_gamma);
        EOS::RPY2Cs(eos_state_rho, eos_state_p, spr, eos_state_cs);
        const amrex::Real rhoe_r = eos_state_rho * eos_state_e;
        const amrex::Real gamc_r = eos_state_gamma;

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
          c[n] = 0.5 * (K_cc(i,j,k,n) + K_cc(ii,jj,kk,n));
        
        // Calculate the cell-edge drift velocity
        for(int n=0; n<NUM_SPECIES; n++){
          drift_tmp[n] = 0.0;
          drift_tmp[n] = c[n] * E_edge[dir](i,j,k);
        }

        // Store cell-center drift velocity for time step estimation
        for(int n=0; n<NUM_SPECIES; n++){
          drift_cc(i, j, k, NUM_E*n + 0) = K_cc(i, j, k, n) * E_cc(i, j, k, 0);
          drift_cc(i, j, k, NUM_E*n + 1) = K_cc(i, j, k, n) * E_cc(i, j, k, 1);
          drift_cc(i, j, k, NUM_E*n + 2) = K_cc(i, j, k, n) * E_cc(i, j, k, 2);
        }
#endif
        amrex::Real flux_tmp[NVAR] = {0.0};
        amrex::Real ustar = 0.0;

        amrex::Real tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
        riemann(
          qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2],
          qtempl[R_P], rhoe_l, spl, gamc_l, qtempr[R_RHO], qtempr[R_UN],
          qtempr[R_UT1], qtempr[R_UT2], qtempr[R_P], rhoe_r, spr, gamc_r,
          bc_test_val, csmall, cavg, ustar, flux_tmp[URHO], flux_tmp[f_idx[0]],
          flux_tmp[f_idx[1]], flux_tmp[f_idx[2]], flux_tmp[UEDEN],
          flux_tmp[UEINT], tmp0, tmp1, tmp2, tmp3, tmp4, tmp5);

        for (int n = 0; n < NUM_SPECIES; n++) {
          flux_tmp[UFS + n] = (ustar + drift_tmp[n] > 0.0) ? flux_tmp[URHO] * qtempl[R_Y + n]
                                            : flux_tmp[URHO] * qtempr[R_Y + n];
          flux_tmp[UFS + n] =
            (ustar + drift_tmp[n] == 0.0)
              ? flux_tmp[URHO] * 0.5 * (qtempl[R_Y + n] + qtempr[R_Y + n])
              : flux_tmp[UFS + n];
        }


        flux_tmp[UTEMP] = 0.0;
        for (int n = UFX; n < UFX + NUM_AUX; n++) {
          flux_tmp[n] = (NUM_AUX > 0) ? 0.0 : flux_tmp[n];
        }
        for (int n = UFA; n < UFA + NUM_ADV; n++) {
          flux_tmp[n] = (NUM_ADV > 0) ? 0.0 : flux_tmp[n];
        }

#ifdef PELEC_USE_PLASMA
        // Recalculate fluxes taking into account drift velocity
        // Riemann solver temporary values: tmp0 = idir velocity
        //                                  tmp1 = other velocity comp 1
        //                                  tmp2 = other velocity comp 2
        //                                  tmp3 = godunov state pressure
        //                                  tmp4 = 

        // Calculate new species and u momentum fluxes
        amrex::Real Xstar[NUM_SPECIES];
        amrex::Real ndens = 0.0;
        amrex::Real kB = 1.380649e-16; // erg/K
        amrex::Real NA = 6.0221409e23; // 1/mol
        amrex::Real Ttemp;
        double EoN, Te;
        amrex::Real mwt[NUM_SPECIES];
        EOS::molecular_weight(mwt);
        amrex::Real mfgd[NUM_SPECIES];
        amrex::Real uflux_tmp = 0.0;
        flux_tmp[f_idx[0]] = 0.0;
        for(int n = 0; n < NUM_SPECIES; n++){
          // Calculate species density flux using left or right state (based on effective velocity)
          flux_tmp[UFS + n] = (ustar + drift_tmp[n] > 0.0) ? tmp5 * (tmp0 + drift_tmp[n]) * qtempl[R_Y + n]
                                                           : tmp5 * (tmp0 + drift_tmp[n]) * qtempr[R_Y + n];
          
          // Calculate u momentum density flux using left or right state (based on effective velocity)
          uflux_tmp = (ustar + drift_tmp[n] > 0.0) ? tmp5 * pow((tmp0 + drift_tmp[n]), 2) * qtempl[R_Y + n]
                                                   : tmp5 * pow((tmp0 + drift_tmp[n]), 2) * qtempr[R_Y + n];

          // Correct flux value to account for a zero effective velocity
          flux_tmp[UFS + n] =
            (ustar + drift_tmp[n] == 0.0)
              ? tmp5 * (tmp0 + drift_tmp[n]) * 0.5 * (qtempl[R_Y + n] + qtempr[R_Y + n])
              : flux_tmp[UFS + n];

          flux_tmp[f_idx[0]] +=
            (ustar + drift_tmp[n] == 0.0)
              ? tmp5 * pow((tmp0 + drift_tmp[n]), 2) * 0.5 * (qtempl[R_Y + n] + qtempr[R_Y + n])
              : uflux_tmp;


          // Re-evaluate species mass fractions based on corrections          
          // mfgd[n] = flux_tmp[UFS + n] / (tmp5 * (tmp0 + drift_tmp[n]));
        }
      
        // TODO: should the Godunov states (density pressure velocity) be re-evaluated as well? 

        // Use species fluxes to correct density flux and density values
        flux_tmp[URHO] = 0.0;  
        for(int n = 0; n < NUM_SPECIES; n++) flux_tmp[URHO] += flux_tmp[UFS + n];
        // rgd = flux_tmp[URHO] / tmp0;

        // Use updated density flux to calculate new momentum fluxes
        // TODO: do the velocity components that are not orthogonal to the cell 
        // face also need to be updated with appropriate drift components?
        // flux_tmp[f_idx[0]] +=  flux_tmp[URHO] * tmp0 + tmp3;
        flux_tmp[f_idx[0]] +=  tmp3;
        flux_tmp[f_idx[1]] = flux_tmp[URHO] * tmp1;
        flux_tmp[f_idx[2]] = flux_tmp[URHO] * tmp2;

        // Re-evaluate other quantities to obtain new energy fluxes
        // amrex::Real egd;
        // EOS::RYP2E(tmp5, mfgd, tmp3, egd);      
        // amrex::Real regd = tmp5 * egd;
        // amrex::Real rhoetot = regd + 0.5 * tmp5 * (tmp0 * tmp0 + tmp1 * tmp1 + tmp2 * tmp2);
        // flux_tmp[UEDEN] = tmp0 * (rhoetot + tmp3); 
        // flux_tmp[UEINT] = tmp0 * regd;
#endif
        for (int ivar = 0; ivar < NVAR; ivar++) {
          flx[dir](i, j, k, ivar) += flux_tmp[ivar] * a[dir](i, j, k);
        }

#ifdef PELEC_USE_PLASMA 
        // Overwrite species fluxes at the electrode boundaries
        // Calculate number density at the interior cell (0th order approx for now)
        // assumes Y_k at ghost cell is equal to interior value at ext_dir boundary,
        // so doesn't matter which species array we take from for now
        // TODO: make sure calculation of EoN is in units of Td
        // TODO: Make sure other flux values are updated as well, if necessary

        int iv[3] = {i,j,k};
        amrex::Real ionFlux = 0.0;

        // overwrite fluxes on all ext_dir boundaries
        if ((bcr[dir] == amrex::BCType::ext_dir) and (iv[dir] == domlo[dir])) {
          // Use EoN to get Te for electron flux at the boundary
          ExtrapTe(eon(i, j, k, 0), &Te);
          flx[dir](i, j, k, URHO) = 0.0;
          for(int n=0; n<NUM_SPECIES; n++){
            flx[dir](i, j, k, UFS + n) = 0.0;
            if(n == E_ID){
              flx[dir](i, j, k, UFS + n) = -0.5 * qtempr[R_RHO] * spr[n] * pow( (8.0*kB*Te) / ((mwt[n]/NA) * constants::PI()) ,0.5) * a[dir](i, j, k);
            }
            if(n != E_ID && K_cc(i,j,k,n) != 0){
              if(ion_bc_type == 0){
                flx[dir](i, j, k, UFS + n) = -0.5 * qtempr[R_RHO] * spr[n] * pow( (8.0*kB*Ttemp) / ((mwt[n]/NA) * constants::PI()) ,0.5) * a[dir](i, j, k);
              }
              else if(ion_bc_type == 1){
                if((K_cc(i,j,k,n) < 0 && E_edge[dir](i,j,k) > 0) || (K_cc(i,j,k,n) > 0 && E_edge[dir](i,j,k) < 0)){
                  flx[dir](i, j, k, UFS + n) = qtempr[R_RHO] * spr[n] * c[n] * E_edge[dir](i,j,k) * a[dir](i, j, k);
                }
                else{
                  flx[dir](i, j, k, UFS + n) = 0.0;
                }
              }
              else{
                printf("Ion BC type not supported!\n");
                exit(1);
              }
              // Save ion flux for secondary electron emissions and convert to number density
              ionFlux += flx[dir](i, j, k, UFS + n) / mwt[n] * NA;
            }
            flx[dir](i, j, k, URHO) += flx[dir](i, j, k, UFS + n);
          }
          flx[dir](i, j, k, UFS + E_ID) += 2.0 * secondary_em_coef * ionFlux * mwt[E_ID] / NA;
        }
        if ((bcr[dir+AMREX_SPACEDIM] == amrex::BCType::ext_dir) and (iv[dir] == domhi[dir]+1)) {
          ExtrapTe(eon(ii, jj, kk, 0), &Te);
          flx[dir](i, j, k, URHO) = 0.0;
          for(int n=0; n<NUM_SPECIES; n++){
            flx[dir](i, j, k, UFS + n) = 0.0;
            if(n == E_ID){
              flx[dir](i, j, k, UFS + n) = 0.5 * qtempl[R_RHO] * spl[n] * pow( (8.0*kB*Te) / ((mwt[n]/NA) * constants::PI()) ,0.5) * a[dir](i, j, k);
            }
            if(n != E_ID && K_cc(i,j,k,n) != 0){
              if(ion_bc_type == 0){
                flx[dir](i, j, k, UFS + n) = 0.5 * qtempl[R_RHO] * spl[n] * pow( (8.0*kB*Ttemp) / ((mwt[n]/NA) * constants::PI()) ,0.5) * a[dir](i, j, k);
              }
              else if(ion_bc_type == 1){
                if((K_cc(i,j,k,n) < 0 && E_edge[dir](i,j,k) < 0) || (K_cc(i,j,k,n) > 0 && E_edge[dir](i,j,k) > 0)){
                  flx[dir](i, j, k, UFS + n) = qtempl[R_RHO] * spl[n] * c[n] * E_edge[dir](i,j,k) * a[dir](i, j, k);
                }
                else{
                  flx[dir](i, j, k, UFS + n) = 0.0;
                }
              }
              else{
                printf("Ion BC type not supported!\n");
                exit(1);
              }
              // Save ion flux for secondary electron emissions and convert to number density
              ionFlux += flx[dir](i, j, k, UFS + n) / mwt[n] * NA;
            }
            flx[dir](i, j, k, URHO) += flx[dir](i, j, k, UFS + n);
          }

          // Add on secondary electron emission based on ion fluxes
          // It is assumed that electrode boundary is an absolutely absorbing wall
          flx[dir](i, j, k, UFS + E_ID) -= 2.0 * secondary_em_coef * ionFlux * mwt[E_ID] / NA;
        }
#endif

      });
  }

#ifdef PELEC_USE_EB
  // nextra was 3 for EB in PeleC but we are operating on a different
  // box here, so this should be zero.
  const int nextra = 0;

  const amrex::Real full_area = std::pow(del[0], AMREX_SPACEDIM - 1);
  const auto lo = amrex::lbound(cbox);
  const auto hi = amrex::ubound(cbox);

  const amrex::Real captured_eb_small_vfrac = eb_small_vfrac;
  amrex::ParallelFor(nebflux, [=] AMREX_GPU_DEVICE(int L) {
    const int i = ebg[L].iv[0];
    const int j = ebg[L].iv[1];
    const int k = ebg[L].iv[2];
    amrex::Real qtempl[5 + NUM_SPECIES] = {0.0};
    amrex::Real qtempr[5 + NUM_SPECIES] = {0.0};
    amrex::Real cavg = 0.0;
    amrex::Real csmall = 0.0;
    amrex::Real /*cspeed = 0.0,*/ rhoe_l = 0.0;
    amrex::Real gamc_l = 0.0;
    amrex::Real spl[NUM_SPECIES] = {0.0};
    amrex::Real flux_tmp[NVAR] = {0.0};
    amrex::Real ebnorm[AMREX_SPACEDIM] = {AMREX_D_DECL(
      ebg[L].eb_normal[0], ebg[L].eb_normal[1], ebg[L].eb_normal[2])};
    const amrex::Real ebnorm_mag = std::sqrt(
      ebnorm[0] * ebnorm[0] + ebnorm[1] * ebnorm[1] + ebnorm[2] * ebnorm[2]);
    for (amrex::Real& dir : ebnorm) {
      dir /= ebnorm_mag;
    }
    if (is_inside(i, j, k, lo, hi, nextra)) {
      if (vfrac(i, j, k) < captured_eb_small_vfrac) {
        amrex::Real sum_kappa = 0.0;
        amrex::Real sum_nbrs_qc = 0.0;
        amrex::Real sum_nbrs_qcsmall = 0.0;
        amrex::Real sum_nbrs_qu = 0.0;
        amrex::Real sum_nbrs_qv = 0.0;
        amrex::Real sum_nbrs_qw = 0.0;
        amrex::Real sum_nbrs_qp = 0.0;
        amrex::Real sum_nbrs_qr = 0.0;
        amrex::Real sum_nbrs_sp[NUM_SPECIES] = {0.0};
        for (int ii = -1; ii <= 1; ii++) {
#if AMREX_SPACEDIM > 1
          for (int jj = -1; jj <= 1; jj++) {
#if AMREX_SPACEDIM == 3
            for (int kk = -1; kk <= 1; kk++) {
#endif
#endif
              int nbr = flags(i, j, k).isConnected(ii, jj, kk);
              if ((ii == 0) && (jj == 0) && (kk == 0)) {
                nbr = 0;
              }
              sum_kappa += nbr * vfrac(i + ii, j + jj, k + kk);
              sum_nbrs_qc += nbr * vfrac(i + ii, j + jj, k + kk) *
                             qaux(i + ii, j + jj, k + kk, QC);
              sum_nbrs_qcsmall += nbr * vfrac(i + ii, j + jj, k + kk) *
                                  qaux(i + ii, j + jj, k + kk, QCSML);
              sum_nbrs_qu += nbr * vfrac(i + ii, j + jj, k + kk) *
                             q(i + ii, j + jj, k + kk, QU);
              sum_nbrs_qv += nbr * vfrac(i + ii, j + jj, k + kk) *
                             q(i + ii, j + jj, k + kk, QV);
              sum_nbrs_qw += nbr * vfrac(i + ii, j + jj, k + kk) *
                             q(i + ii, j + jj, k + kk, QW);
              sum_nbrs_qp += nbr * vfrac(i + ii, j + jj, k + kk) *
                             q(i + ii, j + jj, k + kk, QPRES);
              sum_nbrs_qr += nbr * vfrac(i + ii, j + jj, k + kk) *
                             q(i + ii, j + jj, k + kk, QRHO);
              for (int n = 0; n < NUM_SPECIES; n++) {
                sum_nbrs_sp[n] += nbr * vfrac(i + ii, j + jj, k + kk) *
                                  q(i + ii, j + jj, k + kk, QFS + n);
              }
#if AMREX_SPACEDIM > 1
#if AMREX_SPACEDIM == 3
            }
#endif
          }
#endif
        }
        qtempl[R_UN] = 0.0;
        qtempl[R_UN] -= (sum_nbrs_qu * ebnorm[0] + sum_nbrs_qv * ebnorm[1] +
                         sum_nbrs_qw * ebnorm[2]) /
                        sum_kappa;
        qtempl[R_UT1] = 0.0;
        qtempl[R_UT2] = 0.0;
        qtempl[R_P] = sum_nbrs_qp / sum_kappa;
        qtempl[R_RHO] = sum_nbrs_qr / sum_kappa;
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] = sum_nbrs_sp[n] / sum_kappa;
        }
        cavg = sum_nbrs_qc / sum_kappa;
        csmall = sum_nbrs_qcsmall / sum_kappa;
        // cspeed = cavg;

        // Flip the velocity about the normal for the right state - will use
        // left state for remainder of right state
        qtempr[R_UN] = -1.0 * qtempl[R_UN];

      } else {
        // Assume left state is the cell centered state - normal velocity
        qtempl[R_UN] =
          -(q(i, j, k, QU) * ebnorm[0] + q(i, j, k, QV) * ebnorm[1] +
            q(i, j, k, QW) * ebnorm[2]);
        qtempl[R_UT1] = 0.0;
        qtempl[R_UT2] = 0.0;
        qtempl[R_P] = q(i, j, k, QPRES);
        qtempl[R_RHO] = q(i, j, k, QRHO);
        for (int n = 0; n < NUM_SPECIES; n++) {
          qtempl[R_Y + n] = q(i, j, k, QFS + n);
        }
        cavg = qaux(i, j, k, QC);
        csmall = qaux(i, j, k, QCSML);
        // cspeed = qaux(i, j, k, QC);

        // Flip the velocity about the normal for the right state - will use
        // left  state for remainder of right state
        qtempr[R_UN] = -1.0 * qtempl[R_UN];
      }

      amrex::Real eos_state_rho = qtempl[R_RHO];
      amrex::Real eos_state_p = qtempl[R_P];
      for (int n = 0; n < NUM_SPECIES; n++) {
        spl[n] = qtempl[R_Y + n];
      }
      amrex::Real eos_state_e;
      EOS::RYP2E(eos_state_rho, spl, eos_state_p, eos_state_e);
      rhoe_l = eos_state_rho * eos_state_e;
      amrex::Real eos_state_T;
      EOS::RYP2T(eos_state_rho, spl, eos_state_p, eos_state_T);
      EOS::TY2G(eos_state_T, spl, gamc_l);
    }

    if (is_inside(i, j, k, lo, hi, nextra - 1)) {
      amrex::Real tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, ustar = 0.0;
      riemann(
        qtempl[R_RHO], qtempl[R_UN], qtempl[R_UT1], qtempl[R_UT2], qtempl[R_P],
        rhoe_l, spl, gamc_l, qtempl[R_RHO], qtempr[R_UN], qtempl[R_UT1],
        qtempl[R_UT2], qtempl[R_P], rhoe_l, spl, gamc_l, bc_test_val, csmall,
        cavg, ustar, flux_tmp[URHO], flux_tmp[UMX], flux_tmp[UMY],
        flux_tmp[UMZ], flux_tmp[UEDEN], flux_tmp[UEINT], tmp0, tmp1, tmp2, tmp3,
        tmp4, tmp5);

      flux_tmp[UMY] = -flux_tmp[UMX] * ebnorm[1];
      flux_tmp[UMZ] = -flux_tmp[UMX] * ebnorm[2];
      flux_tmp[UMX] = -flux_tmp[UMX] * ebnorm[0];

      // Compute species flux like passive scalar from intermediate state
      for (int n = 0; n < NUM_SPECIES; n++) {
        flux_tmp[UFS + n] = flux_tmp[URHO] * qtempl[R_Y + n];
      }

#ifdef PELEC_USE_PLASMA
      // /EB/AMReX_EBMultiFabUtil_2D_C.H eb_interp_centroid2facecent_x
#endif
  
      // Copy result into ebflux vector. Being a bit chicken here and only
      // copy values where ebg % iv is within box
      for (int n = 0; n < NVAR; n++) {
        ebflux[n * nebflux + L] += flux_tmp[n] * ebg[L].eb_area * full_area;
      }
    }
  });
#endif
}
