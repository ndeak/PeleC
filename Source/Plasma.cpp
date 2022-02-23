#include <PeleC.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLPoisson.H>
#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#endif
#include <AMReX_ParmParse.H>
#include <Plasma_K.H>
#include <PlasmaBCFill.H>
#include <Plasma.H>

using namespace amrex;

namespace EFConst
{
   amrex::Real eps0 = 8.854187817e-12;                //Free space permittivity (C2/(V.m))
   amrex::Real eps0_cgs = 8.854187817e-12 * 1.0e-9;   //Free space permittivity (C2/(erg.cm))
   amrex::Real epsr = 1.0;
   amrex::Real elemCharge = 1.60217662e-19;     //Coulomb per charge
   amrex::Real Na = 6.0221409e23;                   //Avogadro's number
   amrex::Real kB = 1.380649e-16;
   amrex::Real PP_RU_MKS = 8.31446261815324;    //Universal gas constant (J/mol-K)
   amrex::Real PP_RU_CGS = 83144626.1815324;    // (erg/mol-K)
   amrex::Real me_cgs = 9.10938356e-28;         // electron mass (g)
}

void 
PeleC::plasma_init() 
{
    amrex::Print() << " Init PLASMA solve options \n";

    // Params defaults
    ef_verbose = 0;
    ef_debug = 0;
    def_harm_avg_cen2edge  = false;

    // User input parser to query efield inputs
    amrex::ParmParse pp("ef");
    pp.query("verbose",ef_verbose);
    pp.query("debug",ef_debug);
    pp.query("def_harm_avg_cen2edge",def_harm_avg_cen2edge);
    pp.query("use_nonLinearSolve",ef_use_NLsolve);

    pp.query("Poisson_tol",ef_PoissonTol);
    pp.query("Poisson_verbose",ef_PoissonVerbose);
    pp.query("noSpaceCharge",ef_noSpaceCharge);
    pp.query("constVoltage",ef_constVoltage);
    pp.query("triangle_pulse",ef_triangle_pulse);
    pp.query("trapezoidal_pulse",ef_trapezoidal_pulse);
    pp.query("sigmoid_pulse",ef_sigmoid_pulse);
    pp.query("pulse_delay",ef_pulse_delay);
    pp.query("pulse_rise",ef_pulse_rise);
    pp.query("pulse_plateau",ef_pulse_plateau);
    pp.query("do_drift",ef_do_drift);
    pp.query("do_photoionization", ef_do_photoionization);
    pp.query("star_update", ef_star_update);

    pp.query("JFNK_newtonTol",ef_newtonTol);
    pp.query("JFNK_maxNewton",ef_maxNewtonIter);
    pp.query("JFNK_difftype",ef_diffT_jfnk);
    pp.query("JFNK_lambda",ef_lambda_jfnk);
    pp.query("GMRES_restart_size",ef_GMRES_size);
    pp.query("GMRES_rel_tol",ef_GMRES_reltol);
    pp.query("GMRES_max_restart",ef_GMRES_maxRst);
    pp.query("GMRES_verbose",ef_GMRES_verbose);
    pp.query("JFNK_time_order", ef_time_order);
    pp.query("JFNK_space_order", ef_space_order);
    pp.query("JFNK_pred_charge", ef_pred_charge);

    pp.query("Precond_MG_tol",ef_PC_MG_Tol);
    pp.query("Precond_fixedIter",ef_PC_fixedIter);
    pp.query("Precond_SchurApprox",ef_PC_approx);

    pp.query("pac_mechanism", pac_mechanism);
    pp.query("constEleTransport", ef_constEleTransport);
    pp.query("eleMobility", ef_eleMobility);
    pp.query("eleDiffusivity", ef_eleDiffusivity);

    pp.query("plot_numdens", plot_numdens);

    pp.query("circuit_model", ef_circuit_model);
    pp.query("circuit_time_delay", ef_circuit_time_delay);
    pp.query("circuit_impedance", ef_circuit_impedance);
    pp.query("circuit_upstream_resistance", ef_circuit_upstream_resistance);
    pp.query("circuit_capacitance", ef_circuit_capacitance);
    pp.query("circuit_load_data", ef_circuit_load_data);

    // get charge per unit mass (C/g) CGS
    Real zk_temp[NUM_SPECIES] = {0.0};
    int zk_num_temp[NUM_SPECIES] = {0};
    auto eos = pele::physics::PhysicsType::eos();
    eos.charge_mass(zk_temp);
    CKCHRG(zk_num_temp);
    for (int k = 0; k < NUM_SPECIES; k++) {
       zk[k] = zk_temp[k];
       zk_num[k] = zk_num_temp[k];
    }

   // Allocate time series arrays for circuit modeling
   if(ef_circuit_model){
      // Assuming electrode voltage at time t=0 is 0
      eleVoltage_ts[0] = 0.0;
   }
}

void PeleC::plasma_define_data() {

   // Define vector of PI source MFs so coarse BC data is available
   // if(level == 0){
   //    const int max_level = parent->maxLevel();
   //    const int nlevs = max_level + 1;

   //    PI_source.resize(nlevs);
   // }

   // TODO Solve Poisson problem for potential, and fill in E components and redE after creating
   Efield.define(grids, dmap, NUM_E, numGrow(), amrex::MFInfo(), Factory()); Efield.setVal(0.0);
   Efield_L_p2.define(grids, dmap, NUM_E, numGrow(), amrex::MFInfo(), Factory()); Efield_L_p2.setVal(0.0);
   old_Efield.define(grids, dmap, NUM_E, numGrow(), amrex::MFInfo(), Factory()); Efield.setVal(0.0);
   redEfield.define(grids, dmap, 1, numGrow(), amrex::MFInfo(), Factory()); redEfield.setVal(0.0);
   KSpec_old.define(grids,dmap,NUM_SPECIES, numGrow()); KSpec_old.setVal(0.0);
   KSpec_new.define(grids,dmap,NUM_SPECIES, numGrow()); KSpec_new.setVal(0.0);
   spec_drift.define(grids,dmap,NUM_E*NUM_SPECIES,numGrow()); spec_drift.setVal(0.0);
   coeffs_old.define(grids,dmap,NUM_SPECIES+3, numGrow()); coeffs_old.setVal(0.0);
   Q_ext.define(grids,dmap,NQ,numGrow()); Q_ext.setVal(0.0);
   Qaux_ext.define(grids,dmap,NQAUX,numGrow()); Qaux_ext.setVal(0.0);
   ionFlx_eb.define(grids,dmap,1,numGrow()); ionFlx_eb.setVal(0.0);      // EB ion fluxes - a bit inefficient to store as full MF
   PI_source.define(grids, dmap, 4, 1, amrex::MFInfo(), Factory()); PI_source.setVal(0.0);
   dielectric_ts.define(grids, dmap, 1, numGrow(), amrex::MFInfo(), Factory()); dielectric_ts.setVal(1.0);
   disp_current_mf.define(grids, dmap, 1, 1, amrex::MFInfo(), Factory()); disp_current_mf.setVal(0.0);

   // Intermediate MFs used in transport coef. calculations for NL system
   Ke_cc_mf.define(grids,dmap,1,numGrow(), amrex::MFInfo(), Factory()); Ke_cc_mf.setVal(0.0);
   De_cc_mf.define(grids,dmap,1,numGrow(), amrex::MFInfo(), Factory()); De_cc_mf.setVal(0.0);

   if(ef_circuit_model) {
      spec_2ndo_gradients.define(grids,dmap,3*NUM_SPECIES,2); spec_2ndo_gradients.setVal(0.0);
   }

   if (ef_use_NLsolve) {
      nl_state.define(grids,dmap,2,2);
      nl_resid.define(grids,dmap,2,2);
      nl_F_old.define(grids,dmap,2,2);
      bg_charge.define(grids,dmap,1,1);
      ef_state_old.define(grids,dmap,2,2);
      gasN_cc.define(grids,dmap,1,1);
      tmp_nE_forcing.define(grids,dmap,1,2,MFInfo(),Factory()); tmp_nE_forcing.setVal(0.0);
      nl_nE_2ndo_slopes.define(grids, dmap, 3, 2); nl_nE_2ndo_slopes.setVal(0.0);

      // FIXME: Valgrind complained about unitialized values here, but turning off leads to leak
//       if (elec_Ueff != 0) delete [] elec_Ueff;

      // elec_Ueff = new MultiFab[AMREX_SPACEDIM];
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
         const BoxArray& edgeba = getEdgeBoxArray(d);
         elec_Ueff[d].define(edgeba, dmap, 1, 1,MFInfo(),Factory());
      }

      // Transport coefficients
      diff_e.define(this);
      De_ec = diff_e.get();
      De_ec[0]->setVal(0.0);
      De_ec[1]->setVal(0.0);
      De_ec[2]->setVal(0.0);
      mob_e.define(this);
      Ke_ec = mob_e.get();
      gasN_fb.define(this);
      gasN_ec = gasN_fb.get();
      // ionFlx_fb.define(this,1,3);
      ionFlx_fb.define(this,1,4);
      ionFlx = ionFlx_fb.get();
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
         ionFlx[d]->setVal(0.0);
      }
   }
}

void PeleC::ef_calcGradPhiV(const Real&    time_lcl,
                                  MultiFab &a_phiv,
                                  MultiFab *grad_phiV[AMREX_SPACEDIM]) {

   // Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
   info.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
   const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
   MLEBABecLap poisson({geom}, {grids}, {dmap}, info, {ebf});
#else
   MLABecLaplacian poisson({geom}, {grids}, {dmap}, info);
#endif
   // MLPoisson poisson({geom}, {grids}, {dmap}, info);

   poisson.setMaxOrder(ef_PoissonMaxOrder);

   // BCs
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
   ef_set_PoissonBC(mlmg_lobc, mlmg_hibc);
   poisson.setDomainBC(mlmg_lobc, mlmg_hibc);

   MultiFab phiV_crse;
   if (level > 0) {
      auto& crselev = getLevel(level-1);
      phiV_crse.define(crselev.grids, crselev.dmap, 1, 0, MFInfo(), crselev.Factory());
      MultiFab state_c(crselev.grids, crselev.dmap, NVAR, 0);
      FillPatch(crselev,state_c,0,time_lcl, State_Type, 0, NVAR, 0);
      MultiFab::Copy(phiV_crse,state_c,PhiV,0,1,0);
      poisson.setCoarseFineBC(&phiV_crse, crse_ratio[0]);
   }
   poisson.setLevelBC(0, &a_phiv);
  
   grad_phiV[0]->setVal(0.0);
   grad_phiV[1]->setVal(0.0);
#if AMREX_SPACEDIM == 3
   grad_phiV[2]->setVal(0.0);
#endif
   // Linear solver
   MLMG mlmg(poisson);
   std::array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(grad_phiV[0],grad_phiV[1],grad_phiV[2])};
   mlmg.getFluxes({fp},{&a_phiv});

   if ( ef_debug ) {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
         VisMF::Write(*grad_phiV[d],"GradPhiV_Dir"+std::to_string(d)+"_lvl"+std::to_string(level));
      }
   }
}

void PeleC::ef_calc_transport(const amrex::MultiFab& S, const amrex::Real &time) {
  BL_PROFILE("PeleC::ef_calc_transport()");
 
  // ndeak note - since only MOL is being used for now, it is assumed all data MFs are at time t=n

  if ( ef_verbose ) amrex::Print() << " Compute EF transport prop.\n";

  const TimeLevel whichTime = which_time(State_Type, time);

  // Get the cc transport coeffs. These are temporary.
  MultiFab Ke_cc(grids,dmap,1,S.nGrow());
  MultiFab De_cc(grids,dmap,1,S.nGrow());

  // ndeak add - get BCs for species (used in center->edge extrap)
  const amrex::BCRec& bcspec = get_desc_lst()[State_Type].getBC(UFS);
  amrex::Real mwt[NUM_SPECIES];
  auto eos = pele::physics::PhysicsType::eos();
  eos.molecular_weight(mwt);   // CGS

  // Calculate a constant electron mobility given N*mu, and assuming atmospheric conditions
  // Handling of De is the same
  // N*mu and N*De supplied should be consistent with a given E/N value
  // TODO: this is performed each time step since xport coefs would be overwritten otherwise, but
  // this can be done more efficiently in the future. 
  // Note: rhoDe needs to be recalculated for each cell anyways since rho may differ across the domain
  amrex::Real eleMobility;
  amrex::Real eleDiffusivity; 
  if (ef_constEleTransport == 1){
     eleMobility = -1.0*ef_eleMobility * 1.0e-9 / (2.45e19);     // Converting to cm2-C/erg-s, with charge sign
     eleDiffusivity = ef_eleDiffusivity * 1.0e-2 / (2.45e19); 
  }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
     const amrex::Box& gbox = mfi.growntilebox();
     auto const& rho_ar = S.array(mfi,0);
     auto const& rhoY = S.array(mfi,UFS);
     auto const& T    = Q_ext.array(mfi,QTEMP);
     auto const& rhoD = coeffs_old.array(mfi,dComp_rhoD);
     auto const& Ke   = Ke_cc.array(mfi);
     auto const& De   = De_cc.array(mfi);
     auto const& Ks   = KSpec_old.array(mfi);
     auto const& redEfab = redEfield.array(mfi);
     Real factor = EFConst::PP_RU_CGS / ( EFConst::Na * EFConst::elemCharge );
     int useNL   = ef_use_NLsolve;
     amrex::ParallelFor(gbox, [rhoY, T, factor, Ks, rho_ar, rhoD, Ke, De, useNL, redEfab, mwt, eleMobility, eleDiffusivity]
     AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        if(ef_constEleTransport == 1){
          if (useNL){
            Ke(i,j,k) = eleMobility;
            De(i,j,k) = eleDiffusivity;
          }
          else{
            Ks(i,j,k,E_ID) = eleMobility;
            rhoD(i,j,k,E_ID) = eleDiffusivity * rho_ar(i,j,k);
          }
        }
        else{
          if (useNL) {
             getKappaE(i,j,k,0,Ke,redEfab,rhoY,mwt);
             getDiffE(i,j,k,0,useNL,factor,rhoY,De,redEfab,mwt);
          } else {
             getKappaE(i,j,k,E_ID,Ks,redEfab,rhoY,mwt);
             getDiffE(i,j,k,E_ID,useNL,factor,rhoY,rhoD,redEfab,mwt);
          }
        }
     });
     Real mwt[NUM_SPECIES];
     eos.molecular_weight(mwt);  // Return mwt in CGS
     amrex::ParallelFor(gbox, [rhoY, rhoD, T, Ks, mwt]
     AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        getKappaSp(i,j,k,zk_num, Ks);
     });
  }
  // Copy NL Ke results back into normal array for CFL calculation later
  if(ef_use_NLsolve) MultiFab::Copy(KSpec_old, Ke_cc, 0, E_ID, 1, 0);
  if ( ef_debug ) {
     std::string timetag = (whichTime == AmrOldTime) ? "old" : "new";
     VisMF::Write(KSpec_old,"KappaSpec"+timetag+"_Lvl"+std::to_string(level));
  }

  if ( ef_use_NLsolve ) {
     // CC -> EC transport coeffs. These are PeleC class object used in the non-linear residual.
     // ndeak TODO: check to make sure we are checking all the necessary BCTypes for on_lo/hi
     // TODO: does cen2edg_cpp need to be modified to take into account EBs?
     const Box& domain = geom.Domain();
     bool use_harmonic_avg = def_harm_avg_cen2edge ? true : false;
     const BCRec& bcrec = get_desc_lst()[State_Type].getBC(nE);
 #ifdef _OPENMP
 #pragma omp parallel if (Gpu::notInLaunchRegion())
 #endif
      for (MFIter mfi(De_cc,TilingIfNotGPU()); mfi.isValid();++mfi)
      {
         for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
         {
            const Box ebx = mfi.nodaltilebox(dir);
            const Box& edomain = amrex::surroundingNodes(domain,dir);
            const auto& diff_c  = De_cc.array(mfi);
            const auto& diff_ed = De_ec[dir]->array(mfi);
            const auto& kappa_c  = Ke_cc.array(mfi);
            const auto& kappa_ed = Ke_ec[dir]->array(mfi);
            const auto bc_lo = bcrec.lo(dir);
            const auto bc_hi = bcrec.hi(dir);
            amrex::ParallelFor(ebx, [dir, bc_lo, bc_hi, use_harmonic_avg, diff_c, diff_ed,
                                     kappa_c, kappa_ed, edomain]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && idx[dir] <= edomain.smallEnd(dir) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && idx[dir] >= edomain.bigEnd(dir) );
               cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, diff_c, diff_ed);
               cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, kappa_c, kappa_ed);
            });
         }
      }
      if ( ef_debug ) {
         VisMF::Write(*De_ec[0],"DeEcX_Lvl"+std::to_string(level));
         VisMF::Write(*De_ec[1],"DeEcY_Lvl"+std::to_string(level));
         VisMF::Write(*Ke_ec[0],"KeEcX_Lvl"+std::to_string(level));
         VisMF::Write(*Ke_ec[1],"KeEcY_Lvl"+std::to_string(level));
      }
  }
}

// TODO: finish working on version of ef transport calc that can be folded more cleanly into Diffusion.cpp
void PeleC::ef_calc_transport(amrex::Box const& bx,
                              amrex::Array4<const amrex::Real> const& rhoY_in,
                              amrex::Array4<const amrex::Real> const& EoN_in,
                              amrex::Array4<amrex::Real> const& Ke_out,
                              amrex::Array4<amrex::Real> const& rhoDe_out,
                              amrex::Array4<amrex::Real> const& K_out
) {
  BL_PROFILE("PeleC::ef_calc_transport()");
 
  // ndeak note - since only MOL is being used for now, it is assumed all data MFs are at time t=n

  if ( ef_verbose ) amrex::Print() << " Compute EF transport prop.\n";

  // ndeak add - get BCs for species (used in center->edge extrap)
  amrex::Real mwt[NUM_SPECIES];
  auto eos = pele::physics::PhysicsType::eos();
  eos.molecular_weight(mwt);   // CGS

  Real factor = EFConst::PP_RU_CGS / ( EFConst::Na * EFConst::elemCharge );
  int useNL   = ef_use_NLsolve;
  amrex::ParallelFor(bx, [=]
  AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
     getKappaE(i,j,k,E_ID,Ke_out,EoN_in,rhoY_in,mwt);
     getDiffE(i,j,k,E_ID,useNL,factor,rhoY_in,rhoDe_out,EoN_in,mwt);
     getKappaSp(i,j,k, zk_num, K_out);
  });

//   if ( ef_use_NLsolve ) {
//      // CC -> EC transport coeffs. These are PeleC class object used in the non-linear residual.
//      // ndeak TODO: check to make sure we are checking all the necessary BCTypes for on_lo/hi
//      // TODO: does cen2edg_cpp need to be modified to take into account EBs?
//      const Box& domain = geom.Domain();
//      bool use_harmonic_avg = def_harm_avg_cen2edge ? true : false;
//      const BCRec& bcrec = get_desc_lst()[State_Type].getBC(nE);
//  #ifdef _OPENMP
//  #pragma omp parallel if (Gpu::notInLaunchRegion())
//  #endif
//       for (MFIter mfi(De_cc,TilingIfNotGPU()); mfi.isValid();++mfi)
//       {
//          for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
//          {
//             const Box ebx = mfi.nodaltilebox(dir);
//             const Box& edomain = amrex::surroundingNodes(domain,dir);
//             const auto& diff_c  = De_cc.array(mfi);
//             const auto& diff_ed = De_ec[dir]->array(mfi);
//             const auto& kappa_c  = Ke_cc.array(mfi);
//             const auto& kappa_ed = Ke_ec[dir]->array(mfi);
//             const auto bc_lo = bcrec.lo(dir);
//             const auto bc_hi = bcrec.hi(dir);
//             amrex::ParallelFor(ebx, [dir, bc_lo, bc_hi, use_harmonic_avg, diff_c, diff_ed,
//                                      kappa_c, kappa_ed, edomain]
//             AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//             {
//                int idx[3] = {i,j,k};
//                bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && idx[dir] <= edomain.smallEnd(dir) );
//                bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && idx[dir] >= edomain.bigEnd(dir) );
//                cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, diff_c, diff_ed);
//                cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, kappa_c, kappa_ed);
//             });
//          }
//       }
//       if ( ef_debug ) {
//          VisMF::Write(*De_ec[0],"DeEcX_Lvl"+std::to_string(level));
//          VisMF::Write(*De_ec[1],"DeEcY_Lvl"+std::to_string(level));
//          VisMF::Write(*Ke_ec[0],"KeEcX_Lvl"+std::to_string(level));
//          VisMF::Write(*Ke_ec[1],"KeEcY_Lvl"+std::to_string(level));
//       }
//   }
}

// Setup BC conditions for linear Poisson solve on PhiV. Directly copied from the diffusion one ...
void PeleC::ef_set_PoissonBC(std::array<LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                             std::array<LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc) {

    const BCRec& bc = get_desc_lst()[State_Type].getBC(PhiV);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (Geom().isPeriodic(idim))
        {
            mlmg_lobc[idim] = mlmg_hibc[idim] = LinOpBCType::Periodic;
        }
        else
        {
            int pbc = bc.lo(idim);
            if (pbc == EXT_DIR)
            {
                mlmg_lobc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      ||
                     pbc == REFLECT_EVEN)
            {
                mlmg_lobc[idim] = LinOpBCType::Neumann;
            }
            else if (pbc == REFLECT_ODD)
            {
                mlmg_lobc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                mlmg_lobc[idim] = LinOpBCType::bogus;
            }

            pbc = bc.hi(idim);
            if (pbc == EXT_DIR)
            {
                mlmg_hibc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      ||
                     pbc == REFLECT_EVEN)
            {
                mlmg_hibc[idim] = LinOpBCType::Neumann;
            }
            else if (pbc == REFLECT_ODD)
            {
                mlmg_hibc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                mlmg_hibc[idim] = LinOpBCType::bogus;
            }
        }
    }
}

// Setup BC conditions for linear Poisson solve on PhiV. Directly copied from the diffusion one ...
void PeleC::ef_set_neBC(std::array<LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                        std::array<LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc) {

    const BCRec& bc = get_desc_lst()[State_Type].getBC(nE);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (Geom().isPeriodic(idim))
        {
            mlmg_lobc[idim] = mlmg_hibc[idim] = LinOpBCType::Periodic;
        }
        else
        {
            int pbc = bc.lo(idim);
            if (pbc == EXT_DIR)
            {
                mlmg_lobc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      ||
                     pbc == REFLECT_EVEN)
            {
                mlmg_lobc[idim] = LinOpBCType::Neumann;
            }
            else if (pbc == REFLECT_ODD)
            {
                mlmg_lobc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                mlmg_lobc[idim] = LinOpBCType::bogus;
            }

            pbc = bc.hi(idim);
            if (pbc == EXT_DIR)
            {
                mlmg_hibc[idim] = LinOpBCType::Dirichlet;
            }
            else if (pbc == FOEXTRAP      ||
                     pbc == HOEXTRAP      ||
                     pbc == REFLECT_EVEN)
            {
                mlmg_hibc[idim] = LinOpBCType::Neumann;
            }
            else if (pbc == REFLECT_ODD)
            {
                mlmg_hibc[idim] = LinOpBCType::reflect_odd;
            }
            else
            {
                mlmg_hibc[idim] = LinOpBCType::bogus;
            }
        }
    }
}

// Setup BC conditions for linear Poisson solve on PhiV. Directly copied from the diffusion one ...
void PeleC::setBCPhiV(std::array<LinOpBCType,AMREX_SPACEDIM> &linOp_bc_lo,
                      std::array<LinOpBCType,AMREX_SPACEDIM> &linOp_bc_hi) {

   const BCRec& bc = get_desc_lst()[State_Type].getBC(PhiV);

   for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
   {
      if (Geom().isPeriodic(idim))
      {    
         linOp_bc_lo[idim] = linOp_bc_hi[idim] = LinOpBCType::Periodic;
      }    
      else 
      {
         int pbc = bc.lo(idim);  
         if (pbc == EXT_DIR)
         {    
            linOp_bc_lo[idim] = LinOpBCType::Dirichlet;
         } 
         else if (pbc == FOEXTRAP    ||
                  pbc == REFLECT_EVEN )
         {   
            linOp_bc_lo[idim] = LinOpBCType::Neumann;
         }   
         else
         {   
            linOp_bc_lo[idim] = LinOpBCType::bogus;
         }   
         
         pbc = bc.hi(idim);  
         if (pbc == EXT_DIR)
         {    
            linOp_bc_hi[idim] = LinOpBCType::Dirichlet;
         } 
         else if (pbc == FOEXTRAP    ||
                  pbc == REFLECT_EVEN )
         {   
            linOp_bc_hi[idim] = LinOpBCType::Neumann;
         }   
         else
         {   
            linOp_bc_hi[idim] = LinOpBCType::bogus;
         }   
      }
   }
}

// Setup BC conditions for linear Poisson solve on PhiV. Directly copied from the diffusion one ...
void PeleC::setBCPI(std::array<LinOpBCType,AMREX_SPACEDIM> &linOp_bc_lo,
                      std::array<LinOpBCType,AMREX_SPACEDIM> &linOp_bc_hi) {

   const BCRec& bc = get_desc_lst()[State_Type].getBC(PhiV);

   for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
   {
      if (Geom().isPeriodic(idim))
      {    
         linOp_bc_lo[idim] = linOp_bc_hi[idim] = LinOpBCType::Periodic;
      }    
      else 
      {
         int pbc = bc.lo(idim);  
         if (pbc == EXT_DIR)
         {    
            linOp_bc_lo[idim] = LinOpBCType::Dirichlet;
         } 
         else if (pbc == FOEXTRAP    ||
                  pbc == REFLECT_EVEN )
         {   
            linOp_bc_lo[idim] = LinOpBCType::Neumann;
         }   
         else
         {   
            linOp_bc_lo[idim] = LinOpBCType::bogus;
         }   
         
         pbc = bc.hi(idim);  
         if (pbc == EXT_DIR)
         {    
            linOp_bc_hi[idim] = LinOpBCType::Dirichlet;
         } 
         else if (pbc == FOEXTRAP    ||
                  pbc == REFLECT_EVEN )
         {   
            linOp_bc_hi[idim] = LinOpBCType::Neumann;
         }   
         else
         {   
            linOp_bc_hi[idim] = LinOpBCType::bogus;
         }   
      }
   }
}

// Set the voltage at a given time
void PeleC::setCurrVoltage(Real time) {
  amrex::Real pulse_sigma = pulse_fwhm / (2.0 * sqrt(2.0*log(2.0)));     // Pulse sigma
  amrex::Real pulse_time_tmp;

  curr_voltage = 0.0;
  if(ef_triangle_pulse == 1){
    // Triangular pulse assumes a rise time equal to the pulse fwhm
    for(int i=0; i<pulse_num; i++){
      pulse_time_tmp = pulse_timing  + (i)*(1.0/pulse_freq);
      curr_voltage += (amrex::Math::abs(time - pulse_time_tmp) < pulse_fwhm) ? (1.0 - amrex::Math::abs(time - pulse_time_tmp)/pulse_fwhm)*pulse_peak :0.0;
    }
  }
  else if(ef_trapezoidal_pulse == 1){
    for(int i=0; i<pulse_num; i++){
      pulse_time_tmp = pulse_timing  + (i)*(1.0/pulse_freq);
    
      if(pulse_time_tmp - time > pulse_fwhm/2.0 && pulse_time_tmp - time < pulse_fwhm){
        curr_voltage += (1.0 - amrex::Math::abs( (pulse_time_tmp - (pulse_fwhm/2.0) - time) / (pulse_fwhm/2.0))) * pulse_peak;
      }
      else if(time - pulse_time_tmp > pulse_fwhm/2.0 && time - pulse_time_tmp < pulse_fwhm){
        curr_voltage += (1.0 - amrex::Math::abs( (pulse_time_tmp + (pulse_fwhm/2.0) - time) / (pulse_fwhm/2.0))) * pulse_peak;
      }
      else if( amrex::Math::abs(pulse_time_tmp - time) < pulse_fwhm/2.0){
        curr_voltage += pulse_peak;
      }
      else{
        curr_voltage += 0.0;
      }
    }
  }
  else if(ef_sigmoid_pulse == 1){
    amrex::Real pulse_lambda = 8.0 / ef_pulse_rise;
    amrex::Real pulse_t1 = time - ef_pulse_delay;
    amrex::Real pulse_t2 = time - ef_pulse_delay - ef_pulse_plateau - ef_pulse_rise;
    curr_voltage = pulse_peak* ( (1.0 / (1.0 + exp(-pulse_lambda*pulse_t1) )) + (1.0 / (1.0 + exp(pulse_lambda*pulse_t2) )) - 1.0);
  }
  else{
    for(int i=0; i<pulse_num; i++){
      pulse_time_tmp = pulse_timing  + (i)*(1.0/pulse_freq);
      curr_voltage += pulse_peak * exp(-0.5 * pow( (time - pulse_time_tmp) / pulse_sigma, 2) );
    }
  }

  if(ef_constVoltage == 1) curr_voltage = pulse_peak;
  amrex::Print() << "APPLIED VOLTAGE IS " << curr_voltage/1.0e10 << " kV\n";

  ProbParmDevice* lprobparm = d_prob_parm_device;
  lprobparm->PhiV_top = curr_voltage;
  lprobparm->PhiV_bottom = 0.0;
}

// Return the current applied voltage only
amrex::Real PeleC::getCurrVoltage(Real time) {
  amrex::Real pulse_sigma = pulse_fwhm / (2.0 * sqrt(2.0*log(2.0)));     // Pulse sigma
  amrex::Real pulse_time_tmp;

  curr_voltage = 0.0;
  if(ef_triangle_pulse == 1){
    // Triangular pulse assumes a rise time equal to the pulse fwhm
    for(int i=0; i<pulse_num; i++){
      pulse_time_tmp = pulse_timing  + (i)*(1.0/pulse_freq);
      curr_voltage += (amrex::Math::abs(time - pulse_time_tmp) < pulse_fwhm) ? (1.0 - amrex::Math::abs(time - pulse_time_tmp)/pulse_fwhm)*pulse_peak :0.0;
    }
  }
  else if(ef_trapezoidal_pulse == 1){
    for(int i=0; i<pulse_num; i++){
      pulse_time_tmp = pulse_timing  + (i)*(1.0/pulse_freq);
    
      if(pulse_time_tmp - time > pulse_fwhm/2.0 && pulse_time_tmp - time < pulse_fwhm){
        curr_voltage += (1.0 - amrex::Math::abs( (pulse_time_tmp - (pulse_fwhm/2.0) - time) / (pulse_fwhm/2.0))) * pulse_peak;
      }
      else if(time - pulse_time_tmp > pulse_fwhm/2.0 && time - pulse_time_tmp < pulse_fwhm){
        curr_voltage += (1.0 - amrex::Math::abs( (pulse_time_tmp + (pulse_fwhm/2.0) - time) / (pulse_fwhm/2.0))) * pulse_peak;
      }
      else if( amrex::Math::abs(pulse_time_tmp - time) < pulse_fwhm/2.0){
        curr_voltage += pulse_peak;
      }
      else{
        curr_voltage += 0.0;
      }
    }
  }
  else if(ef_sigmoid_pulse == 1){
    amrex::Real pulse_lambda = 8.0 / ef_pulse_rise;
    amrex::Real pulse_t1 = time - ef_pulse_delay;
    amrex::Real pulse_t2 = time - ef_pulse_delay - ef_pulse_plateau - ef_pulse_rise; 
    curr_voltage = pulse_peak* ( (1.0 / (1.0 + exp(-pulse_lambda*pulse_t1) )) + (1.0 / (1.0 + exp(pulse_lambda*pulse_t2) )) - 1.0);
  }
  else{
    for(int i=0; i<pulse_num; i++){
      pulse_time_tmp = pulse_timing  + (i)*(1.0/pulse_freq);
      curr_voltage += pulse_peak * exp(-0.5 * pow( (time - pulse_time_tmp) / pulse_sigma, 2) );
    }
  }

  if(ef_constVoltage == 1) curr_voltage = pulse_peak;
  amrex::Print() << "APPLIED VOLTAGE IS " << curr_voltage/1.0e10 << " kV\n";

  return curr_voltage;
}

// Note: this function should only be called on the finest level
void PeleC::ef_circuitModel(amrex::Real time, amrex::Real dt){
   
  // Calculate the wave voltage source
  ef_waveVoltageSources(time);

  // Calculate voltages and currents (at source and gap)
  ef_voltagesCurrents(time, dt);

  // Propagate results down to coarser levels
  // Is this needed?
  int lidx = 0;
  int step_num = parent->levelSteps(0) + ef_circuit_load_num - 1;
  while(lidx < parent->finestLevel()) {
    auto& crselev = getLevel(lidx);
    crselev.setCircuitValues(step_num+1, time_ts[step_num], sourceVoltage_ts[step_num], 
                             eleVoltage_ts[step_num + 1], sourceCurrent_ts[step_num], 
                             eleCurrent_ts[step_num], incidentWave_ts[step_num], 
                             reflectedWave_ts[step_num]);
    lidx++;
  }
}

void PeleC::ef_waveVoltageSources(amrex::Real time){

  // Get the current step number
  int step_num = parent->levelSteps(0) + ef_circuit_load_num - 1;

  // Fill in the time array
  time_ts[step_num] = time;

  // Prior to the time delay, there are no wave voltage sources
  if(time < ef_circuit_time_delay){
    incidentWave_ts[step_num] = 0.0;
    reflectedWave_ts[step_num] = 0.0;
  }
  else{     // Otherwise, pull in closest data at (or just after) time - tau
    int idx = 0;
    while(time_ts[idx] < time - ef_circuit_time_delay) idx++;
    incidentWave_ts[step_num] = 2.0 * eleVoltage_ts[idx] - reflectedWave_ts[idx];
    reflectedWave_ts[step_num] = 2.0 * sourceVoltage_ts[idx] - incidentWave_ts[idx];
  }
}

void PeleC::ef_dispCurrent(const amrex::MultiFab &state_curr,
                      const amrex::MultiFab &mu_curr,
                      const amrex::MultiFab &E_curr,
                      const amrex::MultiFab &D_curr){

  amrex::Real mwt[NUM_SPECIES];
  auto eos = pele::physics::PhysicsType::eos();
  eos.molecular_weight(mwt);   // CGS
  amrex::Real fluxE_x, fluxE_y, fluxE_z;
  amrex::Real flux_component;
  const Real* dx = geom.CellSize();
  const Box& domain = geom.Domain();
  const BCRec& bcrec = get_desc_lst()[State_Type].getBC(PhiV);

  // First, loop over all interior cells + 1 ghost cell to fill second order derivative values
  // Assuming that state array passed in has already been FillPatch'd
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
        const auto spec_ar = state_curr.const_array(mfi,UFS);
        const auto grad_ar = spec_2ndo_gradients.array(mfi,dir);
        amrex::ParallelFor(ebx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
           int idx[3] = {i,j,k};
           bool extdir_or_ho_lo = ( bc_lo == amrex::BCType::ext_dir ) || ( bc_lo == amrex::BCType::hoextrap );
           bool extdir_or_ho_hi = ( bc_hi == amrex::BCType::ext_dir ) || ( bc_hi == amrex::BCType::hoextrap );
           for(int n = 0; n < NUM_SPECIES; n++){
              if(dir == 0){
                grad_ar(i,j,k,3*n) = amrex_calc_xslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
              } else if (dir == 1){
                grad_ar(i,j,k,3*n) = amrex_calc_yslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
              } else {
                grad_ar(i,j,k,3*n) = amrex_calc_zslope_extdir(i,j,k,n,2,spec_ar,extdir_or_ho_lo,extdir_or_ho_hi,domain.smallEnd(dir),domain.bigEnd(dir)) / dx[dir];
              }
           }
        });
     }
  }

  for (amrex::MFIter mfi(disp_current_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& tbox = mfi.tilebox();
      const auto curr_arr = disp_current_mf.array(mfi);
      auto const& S_arr = state_curr.array(mfi);
      auto const& E_cc = E_curr.array(mfi);
      auto const& K_cc = KSpec_old.array(mfi);
      auto const& grad_ar = spec_2ndo_gradients.array(mfi);
      auto const& coe_rhoD = coeffs_old.array(mfi,dComp_rhoD);
      amrex::ParallelFor(
        tbox, [=,&fluxE_x, &fluxE_y, &fluxE_z, &flux_component] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Calculate the current due to charged species fluxes
          flux_component = 0.0;
          for(int n = 0; n<NUM_SPECIES; n++){
            if(zk_num[n] != 0){
              // Flux dot Efield components (g-erg/cm3-C-s)
              fluxE_x = zk_num[n] * ( (E_cc(i,j,k,0) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMX)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n) 
                        - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 0) ) * E_cc(i,j,k,0);
              fluxE_y = zk_num[n] * ( (E_cc(i,j,k,1) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMY)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n) 
                        - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n + 1) ) * E_cc(i,j,k,1);
              fluxE_z = zk_num[n] * ( (E_cc(i,j,k,2) * K_cc(i,j,k,n) + (S_arr(i,j,k,UMZ)/S_arr(i,j,k,URHO)))*S_arr(i,j,k,UFS+n) 
                        - (coe_rhoD(i,j,k,n)/S_arr(i,j,k,URHO))*grad_ar(i,j,k,3*n+2) ) * E_cc(i,j,k,2);
      
              // Calculate the total flux contribution (erg/cm3-s)
              flux_component += EFConst::elemCharge * (fluxE_x + fluxE_x + fluxE_x) * ( EFConst::Na / mwt[n]);
            }
          }

          // Calculate the displacement current density (erg/cm3-s)
          curr_arr(i,j,k) = flux_component;
        });
  }

  // Perform volume weighted sum (integral) over the whole domain to obtain displacement current (erg/s)
  // Note: finemask is set to true, so that the integral at each level is calculated only with cells that aren't covered by a finer level
  // By summing the component from each level, we recover the full volume integral
  disp_current = volWgtSumMF(disp_current_mf, 0, false, true);

  // If we are at the finest level, we need to calculate displacement current as the sum from each previous level
  if(level == parent->finestLevel()){
    int lidx = 0;
    while(lidx < parent->finestLevel()) {
      auto& crselev = getLevel(lidx);
      disp_current += crselev.getDispCurrent();
      lidx++;
    }

    // Now need to set the correct displace current at each coarse level
    lidx = 0;
    while(lidx < parent->finestLevel()) {
      auto& crselev = getLevel(lidx);
      crselev.setDispCurrent(disp_current);
      lidx++;
    }
  }

  amrex::Print() << "AT LEVEL " << level << " DISP CURRENT IS " << disp_current << "\n";
}

void PeleC::ef_voltagesCurrents(amrex::Real time, amrex::Real dt){

  // Get the current step number
  int step_num = parent->levelSteps(0) + ef_circuit_load_num - 1;

  // Calculate the gap current I_e^n at the current time
  eleCurrent_ts[step_num] = (eleVoltage_ts[step_num] - reflectedWave_ts[step_num]) / ef_circuit_impedance;

  // Calculate electrode voltage to be used at next time step V_e^n+1
  eleVoltage_ts[step_num + 1] = (eleVoltage_ts[step_num] == 0) ? eleVoltage_ts[step_num] - (dt/ef_circuit_capacitance) * (eleCurrent_ts[step_num]):eleVoltage_ts[step_num] - (dt/ef_circuit_capacitance) * (eleCurrent_ts[step_num] - disp_current / eleVoltage_ts[step_num]); 

  // Calculate source current I_s^n
  // amrex::Real applied_voltage = (time < 2.0*ef_circuit_time_delay) ? getCurrVoltage(time): -1.0*incidentWave_ts[step_num];
  amrex::Real applied_voltage = getCurrVoltage(time);
  sourceCurrent_ts[step_num] = (1.0 / (1.0 + ef_circuit_upstream_resistance / ef_circuit_impedance)) * (applied_voltage - incidentWave_ts[step_num]) / ef_circuit_impedance;

  // Evaluate the source voltage V_s^n
  sourceVoltage_ts[step_num] = applied_voltage - sourceCurrent_ts[step_num] * ef_circuit_upstream_resistance; 
}

void PeleC::ef_loadCircuitData (amrex::Real time){

  // NOTE: function assumes that circuit.dat contains correct set of data
  // Time provided should either be the specified start time (if not restarting), or the cumulative restart time
  // Function is only called once, on level 0. Circuit data arrays at high levels still need to be filled in
  // FIXME: exit conditions somehow being met when restarting from checkpoint, although no error message given...
  
  amrex::Print() << "Beginning circuit.dat file loading process!\n";

  // Check that start time > 0 (or that we are restarting)
  if(time == 0.0){
    amrex::Print() << "WARNING: Start time should be greater than zero when loading in circuit data!\n";
    // exit(1);
  }  

  // Load file and check to make sure circuit.dat exists and has at least one valid line
  std::string circuit_file = "circuit.dat";
  std::ifstream circuitfile(circuit_file.c_str());
  if(!circuitfile.good()) {
    amrex::Print() << "INPUT ERROR : unable to open circuit.dat file!\n";
    exit(1);
  }

  // Get the number of lines in the file
  int circuit_len = std::count(std::istreambuf_iterator<char>(circuitfile), std::istreambuf_iterator<char>(), '\n') - 1;  // Assumes 1 line for header file
  // Quick checks on file length
  if(circuit_len < 1){
    amrex::Print() << "WARNING: circuit.dat file must have at least 1 entries!\n";
    // exit(1);
  }

  // If we are restarting from checkpoint, make sure file has at least as many entries as prev. time steps taken
  // if( parent->levelSteps(0) - 1 > circuit_len ){
  //   amrex::Print() << "WARNING: circuit file has fewer entries than previous time steps taken!\n";
  //   // exit(1); 
  // }
  
  // Load in data line by line until file time >= strt_time
  amrex::Real tstep, ttime, tsvol, tevol, tevoln, tscur, tecur, tiwave, trwave, tfluxcur, tgapcap;
  std::ifstream Loadfile(circuit_file.c_str());
  std::string line;
  std::getline(Loadfile, line);   // Pull off header file
  for(int i=0; i<circuit_len; i++){
    std::getline(Loadfile, line);   // Get data row
    std::istringstream iss(line);
    iss >> tstep >> ttime >> tsvol >> tevol >> tevoln >> tscur >> tecur >> tiwave >> trwave >> tfluxcur >> tgapcap;

    // Make sure first line step number is 1
    if(i == 0 && tstep != 1){
      amrex::Print() << "WARNING: first circuit file line should correspond with time step 1!\n";
      // exit(1); 
    }

    if(amrex::Math::abs(ttime - time) <= 1.0e-20) {
      tstep--;
      break;
    }

    time_ts[tstep - 1] = ttime;
    sourceVoltage_ts[tstep - 1] = tsvol * 1.0e10;  
    eleVoltage_ts[tstep - 1] = tevol * 1.0e10;  
    eleVoltage_ts[tstep] = tevoln * 1.0e10;  
    sourceCurrent_ts[tstep - 1] = tscur;  
    eleCurrent_ts[tstep - 1] = tecur;  
    incidentWave_ts[tstep - 1] = tiwave * 1.0e10;
    reflectedWave_ts[tstep - 1] = trwave * 1.0e10;   
    disp_current = tfluxcur;
  }


  // ef_circuit_load_num used to ensure correct indexing, in event that simulation is started from nonzero start time
  ef_circuit_load_num = tstep - parent->levelSteps(0);

  if( ef_circuit_load_num < 0){
    amrex::Print() << "WARNING: circuit load number should not be negative, but value is " << ef_circuit_load_num << "!\n";
    // exit(1);
  }

  if( tstep < 1){
    amrex::Print() << "WARNING: No lines were loaded in from circuit file!!\n";
    // exit(1);
  }

  // Output final file time loaded in along with simulation start time
  amrex::Print() << "FINISHED: Loading in circuit file data, at level " << level << "\n";
  amrex::Print() << "   latest file time = " << time_ts[tstep - 1] << ", current time = " << time << "\n";

  amrex::Print() << "Loaded " << tstep << " lines, circuit load number is " << ef_circuit_load_num << "\n"; 

  // Rewrite circuit file with only data loaded in
  amrex::Print() << "Rewriting circuit file!\n";
  circuitFileSetup();
  for(int i=0; i<tstep; i++ ) writeCircuitFile(time_ts[i], i);

  amrex::Print() << "NEXT INDEX WRITTEN SHOULD BE " << tstep << "\n";

  // Write to circuit data arrays at higher levels
  // TODO: needed? or are these arrays already shared?
  // int lidx = 1;
  // while(lidx <= parent->finestLevel()) {
  //   auto& crselev = getLevel(lidx);
  //   
  //   lidx++;
  // }
}
