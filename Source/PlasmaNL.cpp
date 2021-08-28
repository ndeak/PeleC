#include <PeleC.H>
#include <GMRES.H>
#include <AMReX_Extrapolater.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLABecCecLaplacian.H>
#include <AMReX_MLPoisson.H>
#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBABecCecLap.H>
#endif
#include <Plasma_K.H>
#include <PlasmaBCFill.H>
#include <Plasma.H>

using namespace amrex;

void PeleC::jtimesv(const MultiFab &v,
                          MultiFab &Jv)
{
    Real vNorm;
    ef_normMF(v,vNorm);
 
    // x is zero, Ax is zero and return
    if ( vNorm == 0.0 ) {
       Jv.setVal(0.0);
       return;
    }
 
    Real delta_pert = ef_lambda_jfnk * ( ef_lambda_jfnk + nl_stateNorm / vNorm );
 
    if ( ef_diffT_jfnk == 1 ) {
       // Create perturbed state
       MultiFab S_pert(grids,dmap,2,2);
       MultiFab::Copy(S_pert, nl_state, 0, 0, 2, 2);
       MultiFab::Saxpy(S_pert,delta_pert,v, 0, 0, 2 ,0);
 
       // Get perturbed residual
       MultiFab res_pert(grids,dmap,2,1);
       ef_nlResidual(nl_dt,S_pert,res_pert);
       res_pert.mult(-1.0);
 
       // Get Ax by finite differece
       MultiFab::LinComb(Jv,1.0,res_pert,0,-1.0,nl_resid,0,0,2,0);
       Jv.mult(-1.0/delta_pert);
    } else if ( ef_diffT_jfnk == 2 ) {
       // Create perturbed states
       MultiFab S_pertm(grids,dmap,2,2);
       MultiFab S_pertp(grids,dmap,2,2);
       MultiFab::Copy(S_pertp, nl_state, 0, 0, 2, 2);
       MultiFab::Copy(S_pertm, nl_state, 0, 0, 2, 2);
       MultiFab::Saxpy(S_pertp,delta_pert,v, 0, 0, 2 ,0);
       MultiFab::Saxpy(S_pertm,-delta_pert,v, 0, 0, 2 ,0);
 
       // Get perturbed residuals
       MultiFab res_pertp(grids,dmap,2,1);
       MultiFab res_pertm(grids,dmap,2,1);
       ef_nlResidual(nl_dt,S_pertp,res_pertp);
       ef_nlResidual(nl_dt,S_pertm,res_pertm);
       res_pertm.mult(-1.0);
       res_pertp.mult(-1.0);
 
       // Get Ax by finite differece
       MultiFab::LinComb(Jv,1.0,res_pertp,0,-1.0,res_pertm,0,0,2,0);
       Jv.mult(-0.5/delta_pert);
    } else {
       Abort(" Unrecognized ef_diffT_jfnk. Should be either 1 (one-sided) or 2 (centered)");
    }
}

void PeleC::ef_solve_NL(const Real     &dt,
                        const Real     &time,
                        const MultiFab &state_in,      
                        const MultiFab &forcing_state,
                        const MultiFab &I_R_in,
                              MultiFab &forcing_nE)
{
   BL_PROFILE("PC_EF::ef_solve_NL()");

   const Real strt_time = ParallelDescriptor::second();

   // Substepping of non-linear solve: DEACTIVATE for now
   nl_dt = dt/1.0;

   // Copy into nl_state (assume state has been FillPatched already)
   MultiFab::Copy(nl_state, state_in, PhiV, 0, 2, nl_state.nGrow());
   // and save the 'old' state
   MultiFab::Copy(ef_state_old, state_in, PhiV, 0, 2, nl_state.nGrow());

   // GMRES
   GMRESSolver gmres;
   int GMRES_tot_count = 0; 
   if ( !ef_use_PETSC_direct ) {
      gmres.define(this,ef_GMRES_size,2,1);        // 2 component in GMRES, 1 GC (needed ?)
      JtimesVFunc jtv = &PeleC::jtimesv;
      gmres.setJtimesV(jtv);
      NormFunc normF = &PeleC::ef_normMF;          // Right now, same norm func as default in GMRES.
      gmres.setNorm(normF);
      PrecondFunc prec = &PeleC::ef_applyPrecond;
      gmres.setPrecond(prec);
      gmres.setVerbose(ef_GMRES_verbose);
      gmres.setMaxRestart(ef_GMRES_maxRst);
   }

   // Need to create the preconditioner LinOp
   PCLinOp_needUpdate = 1;
   PCMLMG_needUpdate = 1;
   int NK_tot_count = 0;

   // -------------------------------------------
   // Pre Newton   
   // Set up the NL state scaling
   nE_scale = (nl_state.norm0(1) > 1.0e-12) ? nl_state.norm0(1) : 1.0;
   phiV_scale = (nl_state.norm0(0) > 1.0e-6 ) ? nl_state.norm0(0) : 1.0;
   nl_state.mult(1.0/nE_scale,1,1,2);
   nl_state.mult(1.0/phiV_scale,0,1,2);
   if ( ef_verbose ) {
      amrex::Print() << " ne scaling: " << nE_scale << "\n";
      amrex::Print() << " phiV scaling: " << phiV_scale << "\n";
   }
   if ( ef_debug ) VisMF::Write(nl_state,"NLScaledStateInit_Lvl"+std::to_string(level));

   // Compute the background charge distribution
   compute_bg_charge(nl_dt, state_in, forcing_state, I_R_in);
   if ( ef_debug ) VisMF::Write(bg_charge,"NLBgCharge_Lvl"+std::to_string(level));

   // Compute face-centered gas number density
   compute_gasN(nl_dt, state_in, forcing_state, I_R_in);

   // Newton initial guess
   ef_normMF(nl_state,nl_stateNorm);

   // Initial NL residual: update residual scaling and preconditioner
   ef_nlResidual( nl_dt, nl_state, nl_resid, true, true );
   nl_resid.mult(-1.0,0,2,2);
   ef_normMF(nl_resid,nl_residNorm);
   if ( ef_debug ) VisMF::Write(nl_resid,"NLResInit_Lvl"+std::to_string(level));

   // Check for direct convergence
   Real max_nlres = std::max(nl_resid.norm0(0),nl_resid.norm0(1));
   if ( max_nlres <= ef_newtonTol ) {
      if ( ef_verbose ) {
         amrex::Print() << "No Newton iteration needed, exiting. \n";
      }
      return;
   }

   // -------------------------------------------
   // Newton   
   int exit_newton = 0;
   int NK_ite = 0;
   do {
      NK_ite += 1;

      // Verbose
      if ( ef_verbose ) {
         amrex::Print() << " Newton it: " << NK_ite << " L2**2 residual: " << 0.5*nl_residNorm*nl_residNorm
                                                    << ". Linf residual: " << max_nlres << "\n";
      }

      // Solve for Newton direction
      MultiFab newtonDir(grids,dmap,2,1);
      newtonDir.setVal(0.0,0,2,1);
      if ( !ef_use_PETSC_direct ) {
         const Real S_tol     = ef_GMRES_reltol;
         const Real S_tol_abs = max_nlres * ef_GMRES_reltol;
         GMRES_tot_count += gmres.solve(newtonDir,nl_resid,S_tol_abs,S_tol);
         if ( ef_debug ) VisMF::Write(newtonDir,"NLDir_NewtIte"+std::to_string(NK_ite)+"_Lvl"+std::to_string(level));
      } else {
         amrex::Print() << "PETSC direct solve in Newton not implemented \n";
      }

      // Linesearch & update state: TODO
      nl_state.plus(newtonDir,0,2,0);
      ef_normMF(nl_state,nl_stateNorm);
      ef_nlResidual( nl_dt, nl_state, nl_resid, false, true );
      nl_resid.mult(-1.0,0,2,2);
      if ( ef_debug ) VisMF::Write(nl_resid,"NLRes_NewtIte"+std::to_string(NK_ite)+"_Lvl"+std::to_string(level));
      if ( ef_debug ) VisMF::Write(nl_state,"NLState_NewtIte"+std::to_string(NK_ite)+"_Lvl"+std::to_string(level));
      ef_normMF(nl_resid,nl_residNorm);
      max_nlres = std::max(nl_resid.norm0(0),nl_resid.norm0(1));

      // Exit condition
      exit_newton = testExitNewton(nl_resid,NK_ite);

   } while( !exit_newton );
   NK_tot_count += NK_ite;

   // -------------------------------------------
   // Post Newton   
   
   // Unscale nl_state
   nl_state.mult(nE_scale,1,1);
   nl_state.mult(phiV_scale,0,1);

   // Compute forcing term on nE
#ifdef _OPENMP   
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(forcing_nE,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
      const Box& bx = mfi.tilebox();
      auto const& old_nE   = ef_state_old.const_array(mfi,1);
      auto const& new_nE   = nl_state.const_array(mfi,1);
      auto const& I_R_nE   = I_R_in.const_array(mfi,NUM_SPECIES+1);
      auto const& force    = forcing_nE.array(mfi);
      Real dtinv           = 1.0 / nl_dt;
      amrex::ParallelFor(bx, [old_nE, new_nE, I_R_nE, force, dtinv, do_react]
      AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
         force(i,j,k) = (new_nE(i,j,k) - old_nE(i,j,k)) * dtinv;
         if (do_react) force(i,j,k) -= I_R_nE(i,j,k);
      });
   }
   if ( ef_debug ) VisMF::Write(forcing_nE,"NL_ForcingnE_Lvl"+std::to_string(level));

   if ( ef_verbose )
   {
      const int IOProc = ParallelDescriptor::IOProcessorNumber();

      Real mx = ParallelDescriptor::second() - strt_time, mn = mx;

      ParallelDescriptor::ReduceRealMin(mn,IOProc);
      ParallelDescriptor::ReduceRealMax(mx,IOProc);

      if ( !ef_use_PETSC_direct ) {
         Real avgGMRES = (float)GMRES_tot_count/(float)NK_tot_count;
         amrex::Print() << " dt: " << dt << " - Avg GMRES/Newton: " << avgGMRES << "\n";
      }
      amrex::Print() << "PeleLM_EF::ef_solve_PNP(): lev: " << level << ", time: ["
                     << mn << " ... " << mx << "]\n";
   }
}

int PeleC::testExitNewton(const MultiFab  &res,
                                int       newtonIter){

   int exit = 0; 
   Real max_res = std::max(res.norm0(0),res.norm0(1));
   if ( max_res <= ef_newtonTol ) {
      exit = 1; 
      if ( ef_verbose ) {
         amrex::Print() << " Newton iterations converged: \n";
         amrex::Print() << " Final Newton L2**2 res norm : " << 0.5*nl_residNorm*nl_residNorm << "\n";
         amrex::Print() << " Final Newton Linf res norm : " << max_res << "\n";
      }    
   }

   if ( newtonIter >= ef_maxNewtonIter && exit == 0 ) {
      exit = 1; 
      amrex::Print() << " Max Newton iteration reached without convergence !!! \n";
   }

   return exit;
}

void PeleC::ef_nlResidual(const Real      &dt_lcl,
                          const MultiFab  &a_nl_state,
                                MultiFab  &a_nl_resid,
                                int       update_res_scaling,
                                int       update_precond){
   BL_PROFILE("PC_EF::ef_nlResidual()");

   // Get the unscaled non-linear state
   MultiFab nl_state_usc(grids,dmap,2,2);
   MultiFab::Copy(nl_state_usc, a_nl_state, 0, 0, 2, 2);
   nl_state_usc.mult(nE_scale,1,1,1);
   nl_state_usc.mult(phiV_scale,0,1,1);

   // Get aliases to make it easier
   MultiFab nE_a(nl_state_usc,amrex::make_alias,1,1);
   MultiFab phi_a(nl_state_usc,amrex::make_alias,0,1);

   // Lap(PhiV) and grad(PhiV)
   FluxBoxes gphi_fb(this, 1, 0);
   MultiFab** gphiV = gphi_fb.get();
   MultiFab laplacian_term(grids, dmap, 1, 0);
   const ProbParmDevice* lprobparm = d_prob_parm_device;
   compPhiVLap(phi_a,laplacian_term,gphiV, *lprobparm);
   if ( ef_debug ) VisMF::Write(laplacian_term,"NLRes_phiVLap_"+std::to_string(level));

   // Diffusion term nE
   MultiFab diffnE(grids, dmap, 1, 0);
   compElecDiffusion(nE_a,diffnE);
   if ( ef_debug ) VisMF::Write(diffnE,"NLRes_ElecDiff_"+std::to_string(level));

   // Advective term nE
   MultiFab advnE(grids, dmap, 1, 0);
   compElecAdvection(nE_a,phi_a,gphiV,advnE);
   if ( ef_debug ) VisMF::Write(advnE,"NLRes_ElecAdv_"+std::to_string(level));

   // Assemble the non-linear residual
   // res(ne(:)) = dt * ( diff(:) + conv(:) + I_R(:) ) - ( ne(:) - ne_old(:) )
   // res(phiv(:)) = \Sum z_k * \tilde Y_k / q_e - ne + Lapl_PhiV
   a_nl_resid.setVal(0.0);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
#ifdef PELEC_USE_EB
  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(Sborder.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();
#endif
   for (MFIter mfi(a_nl_resid,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
      const Box& bx = mfi.tilebox();
      auto const& I_R_nE   = get_new_data(Reactions_Type).const_array(mfi,NUM_SPECIES+1);
      auto const& lapPhiV  = laplacian_term.const_array(mfi);
      auto const& ne_diff  = diffnE.const_array(mfi);
      auto const& ne_adv   = advnE.const_array(mfi);
      auto const& ne_curr  = nE_a.const_array(mfi);
      auto const& ne_old   = ef_state_old.const_array(mfi,1);
      auto const& charge   = bg_charge.const_array(mfi);
      auto const& res_nE   = a_nl_resid.array(mfi,1);
      auto const& res_phiV = a_nl_resid.array(mfi,0);
#ifdef PELEC_USE_EB
      auto flag_arr = flags.const_array(mfi);
#endif
      Real scalLap         = EFConst::eps0_cgs * EFConst::epsr / EFConst::elemCharge;
      amrex::ParallelFor(bx, [ne_curr,ne_old,lapPhiV,I_R_nE,ne_diff,ne_adv,charge,res_nE,res_phiV,
                              dt_lcl,scalLap,do_react
#ifdef PELEC_USE_EB
                              , flag_arr
#endif
                                                      ]
      AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {    
         res_nE(i,j,k) = ne_old(i,j,k) - ne_curr(i,j,k) + dt_lcl * ( ne_diff(i,j,k) + ne_adv(i,j,k) );
         if (do_react) res_nE(i,j,k) += dt_lcl * I_R_nE(i,j,k);
         res_phiV(i,j,k) = lapPhiV(i,j,k) * scalLap - ne_curr(i,j,k) + charge(i,j,k);
#ifdef PELEC_USE_EB
         if(flag_arr(i,j,k).isCovered()){
            res_nE(i,j,k) = 0.0;
            res_phiV(i,j,k) = 0.0;
         }
#endif
      });  
   }

   // Deal with scaling
   if ( update_res_scaling ) {
      FnE_scale = (a_nl_resid.norm0(1) > 1.0e-12) ? a_nl_resid.norm0(1) : 1.0 ;
      FphiV_scale = (a_nl_resid.norm0(0) > 1.0e-12) ? a_nl_resid.norm0(0) : 1.0 ;
      if ( ef_verbose ) {
         amrex::Print() << " F(ne) scaling: " << FnE_scale << "\n";
         amrex::Print() << " F(PhiV) scaling: " << FphiV_scale << "\n";
      }
   }

   a_nl_resid.mult(1.0/FnE_scale,1,1,1);
   a_nl_resid.mult(1.0/FphiV_scale,0,1,1);

   // Update the preconditioner
   if ( update_precond ) {
      ef_setUpPrecond(dt_lcl, nl_state_usc);
   }

}

void PeleC::compPhiVLap(MultiFab& phi,
                        MultiFab& phiLap,
                        MultiFab** gPhiV,
                        ProbParmDevice const& prob_parm){

// Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
   info.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
    const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
    MLEBABecLap phiV_poisson({geom}, {grids}, {dmap}, info, {ebf});
#else
    MLPoisson phiV_poisson({geom}, {grids}, {dmap}, info);
#endif
    phiV_poisson.setMaxOrder(ef_PoissonMaxOrder);

// Set-up BC's
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
   ef_set_PoissonBC(mlmg_lobc, mlmg_hibc);
   phiV_poisson.setDomainBC(mlmg_lobc, mlmg_hibc);

   MultiFab phiV_crse;
   if (level > 0) {
      amrex::Abort("ef_compPhiVLap need to be modified for ML");
      auto& crselev = getLevel(level-1);
      phiV_crse.define(crselev.grids, crselev.dmap, 1, 0, MFInfo(), crselev.Factory());
      //FillPatch(crselev, phiV_crse, 0, time, State_Type, PhiV, 1, 0);
      phiV_poisson.setCoarseFineBC(&phiV_crse, crse_ratio[0]);
   }
   // Set the inhomogenous domain BCs
   phiV_poisson.setLevelBC(0, &phi);

   // for (MFIter mfi(phi,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.growntilebox();
   //     const auto& phi_ar = phi.array(mfi);
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //        // if(i == 4 && k == 4 && j == -1) printf("LOWER PHI DOMAIN BC = %.6e\n", phi_ar(i,j,k));
   //        // if(i == 4 && k == 4 && j == 256) printf("UPPER PHI DOMAIN BC = %.6e\n", phi_ar(i,j,k));
   //     });
   // }

#ifdef AMREX_USE_EB
    // Setup solver coefficient: general form is (ascal * acoef - bscal * div bcoef grad ) phi = rhs
    // For simple Poisson solve: ascal, acoef = 0 and bscal, bcoef = 1
   MultiFab acoef(grids, dmap, 1, 0, MFInfo(), Factory());
   acoef.setVal(0.0);
   phiV_poisson.setACoeffs(0, acoef);
   Array<MultiFab,AMREX_SPACEDIM> bcoef;
   for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
       bcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0, MFInfo(), Factory());
       bcoef[idim].setVal(1.0);
   }
   phiV_poisson.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));
   Real ascal = 0.0;
   Real bscal = -1.0;
   phiV_poisson.setScalars(ascal, bscal);

   // Set the inhomogenous Dirichlet EB BCs
   MultiFab phiV_BC(grids, dmap, 1, 0, MFInfo(), Factory());
   MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
   beta.setVal(1.0);
   // TODO make sure bottom_phiv is updated to be at t=n+1
   const ProbParmDevice* lprobparm = d_prob_parm_device;
   for (MFIter mfi(beta,true); mfi.isValid(); ++mfi)
   {
       const Box& bx = mfi.growntilebox();
       const auto& phiV_ar = phiV_BC.array(mfi);
       const Real* dx      = geom.CellSize();
       const Real* problo  = geom.ProbLo();
       const Real* probhi  = geom.ProbHi();
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           Real y = problo[1] + (j + 0.5)*dx[1];
           if (y >= probhi[1] / 2.0) {
               phiV_ar(i,j,k) = prob_parm.PhiV_top;
               if(i == 4 && k == 4 && j == 366) printf("UPPER EB PHI = %.6e\n", phiV_ar(i,j,k));
           } else {
               phiV_ar(i,j,k) = prob_parm.PhiV_bottom;
               if(i == 4 && k == 4 && j == 110) printf("LOWER EB PHI = %.6e\n",phiV_ar(i,j,k));
           }
       });
   }
   phiV_poisson.setEBDirichlet(0,phiV_BC,beta);
   // phiV_poisson.setEBDirichlet(0,phi,beta);
#endif

// LinearSolver to get divergence
   MLMG solver(phiV_poisson);
   solver.apply({&phiLap},{&phi});
   
// Need the flux (grad(phi))
   Array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(gPhiV[0],gPhiV[1],gPhiV[2])};
   solver.getFluxes({fp},{&phi});
}  

void PeleC::compElecDiffusion(const MultiFab& a_ne,
                                    MultiFab& elecDiff)
{
   // amrex::Print() << " STARTED ELEC DIFF: " << "\n";
   // Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
   info.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
    const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
    MLEBABecLap ne_lapl({geom}, {grids}, {dmap}, info, {ebf});
#else
    MLABecLaplacian ne_lapl({geom}, {grids}, {dmap}, info);
#endif
    ne_lapl.setMaxOrder(ef_PoissonMaxOrder);

   // Set-up BC's
   // Domain BCs
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
   ef_set_neBC(mlmg_lobc, mlmg_hibc);
   ne_lapl.setDomainBC(mlmg_lobc, mlmg_hibc);

   // Level BCs
   // Needed since span-wise BCs are not homogenous
   MultiFab nE_crse;
   if (level > 0) {
      auto& crselev = getLevel(level-1);
      nE_crse.define(crselev.grids, crselev.dmap, 1, 0, MFInfo(), crselev.Factory());
//      FillPatch(crselev, nE_crse, 0, curtime, State_Type, nE, 1, 0);
      ne_lapl.setCoarseFineBC(&nE_crse, crse_ratio[0]);
   }
   ne_lapl.setLevelBC(0, &a_ne);

   // for (MFIter mfi(a_ne,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.growntilebox();
   //     const auto& ne_ar = a_ne.array(mfi);
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //        // if(i == 4 && k == 4 && j == -1) printf("LOWER NE DIFF DOMAIN BC = %.6e\n", ne_ar(i,j,k));
   //        // if(i == 4 && k == 4 && j == 256) printf("UPPER NE DIFF DOMAIN BC = %.6e\n", ne_ar(i,j,k));
   //     });
   // }

   // EB BCs
   // TODO: default for EB is neumann BCs everywhere - do we need to do anything else?
#ifdef PELEC_USE_EB
    MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
    beta.setVal(1.0);

   for (MFIter mfi(beta,true); mfi.isValid(); ++mfi)
   {
       const Box& bx = mfi.growntilebox();
       const auto& ne_ar = a_ne.array(mfi);
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          if(i == 4 && k == 4 && j == 110) printf("LOWER NE DIFF EB BC = %.6e\n", ne_ar(i,j,k));
          if(i == 4 && k == 4 && j == 366) printf("UPPER NE DIFF EB BC = %.6e\n", ne_ar(i,j,k));
       });
   }

    ne_lapl.setEBDirichlet(0,a_ne,beta);
#endif

   // Coeffs
   // TODO: figure out whether edge state accounts for EBs
   ne_lapl.setScalars(0.0, 1.0);
   Array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(De_ec[0],De_ec[1],De_ec[2])};
   ne_lapl.setBCoeffs(0, bcoeffs);

   // LinearSolver to get divergence
   // Need a copy of ne since the linear operator touches the ghost cells
   MultiFab neOp(grids,dmap,1,2); 
   MultiFab::Copy(neOp,a_ne,0,0,1,2);
   MLMG solver(ne_lapl);
   solver.apply({&elecDiff},{&neOp});

   elecDiff.mult(-1.0);

   // for (MFIter mfi(elecDiff,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.growntilebox();
   //     const auto& elecDiff_ar = elecDiff.array(mfi);
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //        // if(i == 4 && k == 4) printf("diff source (%i) = %.6e\n", j, elecDiff_ar(i,j,k));
   //     });
   // }
   // amrex::Print() << " FINISHED ELEC DIFF: " << "\n";

}

void PeleC::compElecAdvection(MultiFab &a_ne,
                              MultiFab &a_phiV,
                              MultiFab *gphiV[AMREX_SPACEDIM],
                              MultiFab &elecAdv)
{

   // TODO: Need to incorporate bulk velocity into function...
   int order = 1;
   // Get the face effective velocity
   // effVel = Umac - \mu_e * gradPhiVcurr
   for (int d = 0; d < AMREX_SPACEDIM; ++d) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(elec_Ueff[d],TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
         const Box& bx = mfi.tilebox();
         auto const& ueff    = elec_Ueff[d].array(mfi);
         auto const& gphi    = gphiV[d]->const_array(mfi);
         //auto const& umac    = u_mac[d].const_array(mfi);
         auto const& kappa_e = Ke_ec[d]->const_array(mfi);
         amrex::ParallelFor(bx, [ueff, gphi, kappa_e]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            // TODO: Nick uses a negative kappa_E -> + Ke*gradPhi
            // TODO: get the a face centered gas velocity in here
            ueff(i,j,k) = kappa_e(i,j,k) * gphi(i,j,k);
         });
      }
   }
   if ( ef_debug ) VisMF::Write(elec_Ueff[0],"NLRes_ElecUeffX_"+std::to_string(level));
   if ( ef_debug ) VisMF::Write(elec_Ueff[1],"NLRes_ElecUeffY_"+std::to_string(level));

   // ----------------------------------------------------
   // Get face-centered E/N
   // Update it at every calls --> maybe not necessary 
   MultiFab EF_cc(grids,dmap,3,0);
   std::array<const amrex::MultiFab* ,AMREX_SPACEDIM> EF_ec{AMREX_D_DECL(gphiV[0], gphiV[1], gphiV[2])};
   average_face_to_cellcenter(EF_cc, 0, EF_ec);

   MultiFab EFMag_cc(grids,dmap,1,1);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(EF_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
      const Box& bx = mfi.tilebox();
      const auto EF_ar = EF_cc.array(mfi);
      const auto EFmag_ar = EFMag_cc.array(mfi);
      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
      {
         EFmag_ar(i,j,k) = std::sqrt( AMREX_D_TERM(  EF_ar(i,j,k,0)*EF_ar(i,j,k,0),
                                                   + EF_ar(i,j,k,1)*EF_ar(i,j,k,1), 
                                                   + EF_ar(i,j,k,2)*EF_ar(i,j,k,2)) );
         EFmag_ar(i,j,k) *= 1e-7 * 1e17;   // Scale erg -> V and to Td. The division by gasN comes later.
      });
   }
   EFMag_cc.FillBoundary(0,1,geom.periodicity());
   Extrapolater::FirstOrderExtrap(EFMag_cc, geom, 0, 1); // Fill ghost cells of EFMag_cc.

   FluxBoxes EoN_fb(this);
   MultiFab** EoN_ec = EoN_fb.get();
   std::array<amrex::MultiFab*,AMREX_SPACEDIM> EoN_arr{AMREX_D_DECL(EoN_ec[0], EoN_ec[1], EoN_ec[2])};
   average_cellcenter_to_face(EoN_arr, EFMag_cc, geom);

   for (int dir = 0; dir < AMREX_SPACEDIM; dir++) 
   {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(*EoN_ec[dir],TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
         const Box& ebx = mfi.tilebox();
         const auto EoN_ar = EoN_ec[dir]->array(mfi);
         const auto gasN_ar = gasN_ec[dir]->const_array(mfi);
         amrex::ParallelFor(ebx,
         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
         {
            EoN_ar(i,j,k) /= gasN_ar(i,j,k);
         });
      }
   }
   // ------------------------------------------------------

   MultiFab nE_new(grids,dmap,1,1);
   nE_new.setVal(0.0);
   MultiFab::Copy(nE_new,a_ne,0,0,1,1);
   // TODO remove?
   // nE_new.FillBoundary();

   {
      FArrayBox cflux[AMREX_SPACEDIM];
      FArrayBox edgstate[AMREX_SPACEDIM];

      const BCRec& bcrec = get_desc_lst()[State_Type].getBC(PhiV);
      const Box& domain = geom.Domain();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      // TODO: could probably condense this loop by putting in loop over each dimension
      for (MFIter mfi(elecAdv,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const Box& bx  = mfi.tilebox();
         const Box& gbx = mfi.growntilebox(1);
         auto const& vol = volume.array(mfi);
         
         // const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM>
         // area_arr{{AMREX_D_DECL(
         // area[0].array(mfi), area[1].array(mfi), area[2].array(mfi))}};

         const amrex::Array4<const amrex::Real> area_x = area[0].array(mfi);
         const amrex::Array4<const amrex::Real> area_y = area[1].array(mfi);
         const amrex::Array4<const amrex::Real> area_z = area[2].array(mfi);

         auto const& ne_arr = nE_new.array(mfi);
         auto const& ne_adv = elecAdv.array(mfi);

         const amrex::Box xbx = amrex::surroundingNodes(bx, 0);
         const amrex::Box ybx = amrex::surroundingNodes(bx, 1);
         const amrex::Box zbx = amrex::surroundingNodes(bx, 2);

         // data arrays
         AMREX_D_TERM( Array4<Real const> u = elec_Ueff[0].const_array(mfi);,
                       Array4<Real const> v = elec_Ueff[1].const_array(mfi);,
                       Array4<Real const> w = elec_Ueff[2].const_array(mfi););
         AMREX_D_TERM( Array4<Real const> EoNx = EoN_ec[0]->const_array(mfi);,
                       Array4<Real const> EoNy = EoN_ec[1]->const_array(mfi);,
                       Array4<Real const> EoNz = EoN_ec[2]->const_array(mfi););
         AMREX_D_TERM( Array4<Real const> ionFx = ionFlx[0]->const_array(mfi);,
                       Array4<Real const> ionFy = ionFlx[1]->const_array(mfi);,
                       Array4<Real const> ionFz = ionFlx[2]->const_array(mfi););
#ifdef PELEC_USE_EB
                       Array4<Real const> ionFeb = ionFlx_eb.const_array(mfi);
#endif

         // Set temporary edge FABs
         AMREX_D_TERM( cflux[0].resize(xbx,1);,
                       cflux[1].resize(ybx,1);,
                       cflux[2].resize(zbx,1););
         AMREX_D_TERM( edgstate[0].resize(xbx,1);,
                       edgstate[1].resize(ybx,1);,
                       edgstate[2].resize(zbx,1););
         AMREX_D_TERM( Array4<Real> xstate = edgstate[0].array();,
                       Array4<Real> ystate = edgstate[1].array();,
                       Array4<Real> zstate = edgstate[2].array(););
         AMREX_D_TERM( Array4<Real> xflux = cflux[0].array();,
                       Array4<Real> yflux = cflux[1].array();,
                       Array4<Real> zflux = cflux[2].array(););

         // Predict edge states
         // X
         {
            // BCs
            const Box& edomain = surroundingNodes(domain,0);
            const auto bc_lo = bcrec.lo(0);
            const auto bc_hi = bcrec.hi(0);

            amrex::ParallelFor(xbx, [ne_arr,u,xstate,bc_lo,bc_hi,edomain,domain,order]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[0] <= edomain.smallEnd(0) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[0] >= edomain.bigEnd(0) ) );
               if (order == 1) {
                  xstate(i,j,k) = ef_edge_state_extdir(i,j,k,0,on_lo,on_hi,ne_arr,u);
               } else if (order == 2) {
                  bool extdir_or_ho_lo = ( bc_lo == amrex::BCType::ext_dir ) || ( bc_lo == amrex::BCType::hoextrap );
                  bool extdir_or_ho_hi = ( bc_hi == amrex::BCType::ext_dir ) || ( bc_hi == amrex::BCType::hoextrap );
                  xstate(i,j,k) = ef_edge_state_2ndO_extdir(i,j,k,0,on_lo,on_hi,extdir_or_ho_lo, extdir_or_ho_hi, 
                                                            domain.smallEnd(0), domain.bigEnd(0), ne_arr,u);
               }
            });
         }
         // Y
         {
            // BCs
            const Box& edomain = surroundingNodes(domain,1);
            const auto bc_lo = bcrec.lo(1);
            const auto bc_hi = bcrec.hi(1);

            amrex::ParallelFor(ybx, [ne_arr,v,ystate,bc_lo,bc_hi,edomain,domain,order]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[1] <= edomain.smallEnd(1) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[1] >= edomain.bigEnd(1) ) );
               if (order == 1) {
                  ystate(i,j,k) = ef_edge_state_extdir(i,j,k,1,on_lo,on_hi,ne_arr,v);
               } else if (order == 2) {
                  bool extdir_or_ho_lo = ( bc_lo == amrex::BCType::ext_dir ) || ( bc_lo == amrex::BCType::hoextrap );
                  bool extdir_or_ho_hi = ( bc_hi == amrex::BCType::ext_dir ) || ( bc_hi == amrex::BCType::hoextrap );
                  ystate(i,j,k) = ef_edge_state_2ndO_extdir(i,j,k,1,on_lo,on_hi,extdir_or_ho_lo, extdir_or_ho_hi,
                                                            domain.smallEnd(1), domain.bigEnd(1),ne_arr,v);
               }
            });
         }
#if ( AMREX_SPACEDIM ==3 )
         // Z
         {
            // BCs
            const Box& edomain = surroundingNodes(domain,2);
            const auto bc_lo = bcrec.lo(2);
            const auto bc_hi = bcrec.hi(2);

            amrex::ParallelFor(zbx, [ne_arr,w,zstate,bc_lo,bc_hi,edomain,domain,order]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[2] <= edomain.smallEnd(2) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[2] >= edomain.bigEnd(2) ) );
               if (order == 1) {
                  zstate(i,j,k) = ef_edge_state_extdir(i,j,k,2,on_lo,on_hi,ne_arr,w);
               } else if (order == 2) {
                  bool extdir_or_ho_lo = ( bc_lo == amrex::BCType::ext_dir ) || ( bc_lo == amrex::BCType::hoextrap );
                  bool extdir_or_ho_hi = ( bc_hi == amrex::BCType::ext_dir ) || ( bc_hi == amrex::BCType::hoextrap );
                  zstate(i,j,k) = ef_edge_state_2ndO_extdir(i,j,k,2,on_lo,on_hi,extdir_or_ho_lo, extdir_or_ho_hi,
                                                            domain.smallEnd(2), domain.bigEnd(2),ne_arr,w);
               }
            });
         }
#endif

         // Computing fluxes
         amrex::ParallelFor(xbx, [u,xstate,xflux,area_x]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            xflux(i,j,k) = u(i,j,k) * xstate(i,j,k) * area_x(i,j,k);
         });
         amrex::ParallelFor(ybx, [v,ystate,yflux,area_y]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            yflux(i,j,k) = v(i,j,k) * ystate(i,j,k) * area_y(i,j,k);
         });
#if ( AMREX_SPACEDIM ==3 )
         amrex::ParallelFor(zbx, [w,zstate,zflux,area_z]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            zflux(i,j,k) = w(i,j,k) * zstate(i,j,k) * area_z(i,j,k);
         });
#endif

         // Overwrite dirichlet BC fluxes
         // TODO: is the 2nd em ion flux quantity already multiplied with area?
         // X
         {
            // BCs
            const Box& edomain = surroundingNodes(domain,0);
            const auto bc_lo = bcrec.lo(0);
            const auto bc_hi = bcrec.hi(0);
            amrex::ParallelFor(xbx, [xflux,xstate,EoNx,ionFx,bc_lo,bc_hi,edomain,area_x]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[0] <= edomain.smallEnd(0) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[0] >= edomain.bigEnd(0) ) );
               if ( on_lo || on_hi ) {
                  // Get the electron temperature
                  amrex::Real Te = 0.0;
                  ExtrapTe(EoNx(idx[0], idx[1], idx[2], 0), &Te);
                  
                  amrex::Real mwt[NUM_SPECIES] = {0.0};
                  auto eos = pele::physics::PhysicsType::eos();
                  eos.molecular_weight(mwt);
 
                  if ( on_lo ) {
                     xflux(i,j,k) = - 0.5 * xstate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );// * a[0](i, j, k);
                     xflux(i,j,k) *= area_x(i,j,k);
                     xflux(i,j,k) -= 2.0 * secondary_em_coef * ionFx(i,j,k);
                  }
                  if ( on_hi ) { 
                     xflux(i,j,k) = 0.5 * xstate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );// * a[0](i, j, k);
                     xflux(i,j,k) *= area_x(i,j,k);
                     xflux(i,j,k) -= 2.0 * secondary_em_coef * ionFx(i,j,k);
                  }
               }
            });
         }
         // Y
         {
            // BCs
            const Box& edomain = surroundingNodes(domain,1);
            const auto bc_lo = bcrec.lo(1);
            const auto bc_hi = bcrec.hi(1);
            amrex::ParallelFor(ybx, [yflux,ystate,EoNy,ionFy,bc_lo,bc_hi,edomain,area_y]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[1] <= edomain.smallEnd(1) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[1] >= edomain.bigEnd(1) ) );
               if ( on_lo || on_hi ) {
                  // Get the electron temperature
                  amrex::Real Te = 0.0;
                  ExtrapTe(EoNy(idx[0], idx[1], idx[2], 0), &Te);
                  
                  amrex::Real mwt[NUM_SPECIES] = {0.0};
                  auto eos = pele::physics::PhysicsType::eos();
                  eos.molecular_weight(mwt);
 
                  if ( on_lo ) {
                     yflux(i,j,k) = -0.5 * ystate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );// * a[0](i, j, k);
                     // if(i == 4 && k == 4) printf("BC flux loss (%i) = %.6e\n", j, -0.5 * ystate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 ));
                     yflux(i,j,k) *= area_y(i,j,k);
                     // if(i == 4 && k == 4) printf("BC ion flux lo (%i) = %.6e\n", j, -2.0 * secondary_em_coef * ionFy(i,j,k));
                     yflux(i,j,k) -= 2.0 * secondary_em_coef * ionFy(i,j,k);
                     // if(i == 4 && k == 4) printf("BCflux 2 lo (%i) = %.6e\n", j, yflux(i,j,k));
                  }
                  if ( on_hi ) { 
                     yflux(i,j,k) = 0.5 * ystate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );// * a[0](i, j, k);
                     // if(i == 4 && k == 4) printf("BCflux hi (%i) = %.6e\n", j, yflux(i,j,k));
                     yflux(i,j,k) *= area_y(i,j,k);
                     // if(i == 4 && k == 4) printf("BC ion flux hi (%i) = %.6e\n", j, 2.0 * secondary_em_coef * ionFy(i,j,k));
                     yflux(i,j,k) -= 2.0 * secondary_em_coef * ionFy(i,j,k);
                     // if(i == 4 && k == 4) printf("BCflux 2 hi (%i) = %.6e\n", j, yflux(i,j,k));
                  }
               }
            });
         }
#if ( AMREX_SPACEDIM ==3 )
         // Z
         {
            // BCs
            const Box& edomain = surroundingNodes(domain,2);
            const auto bc_lo = bcrec.lo(2);
            const auto bc_hi = bcrec.hi(2);
            amrex::ParallelFor(ybx, [zflux,zstate,EoNz,ionFz,bc_lo,bc_hi,edomain,area_z]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[2] <= edomain.smallEnd(2) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[2] >= edomain.bigEnd(2) ) );
               if ( on_lo || on_hi ) {
                  // Get the electron temperature
                  amrex::Real Te = 0.0;
                  ExtrapTe(EoNz(idx[0], idx[1], idx[2], 0), &Te);
                  
                  amrex::Real mwt[NUM_SPECIES] = {0.0};
                  auto eos = pele::physics::PhysicsType::eos();
                  eos.molecular_weight(mwt);
 
                  if ( on_lo ) {
                     zflux(i,j,k) = - 0.5 * zstate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );// * a[0](i, j, k);
                     zflux(i,j,k) *= area_z(i,j,k);
                     zflux(i,j,k) -= 2.0 * secondary_em_coef * ionFz(i,j,k);
                  }
                  if ( on_hi ) { 
                     zflux(i,j,k) = 0.5 * zstate(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );// * a[0](i, j, k);
                     zflux(i,j,k) *= area_z(i,j,k);
                     zflux(i,j,k) -= 2.0 * secondary_em_coef * ionFz(i,j,k);
                  }
               }
            });
         }
#endif

         // Compute divergence
         const auto dxinv = geom.InvCellSizeArray();
         amrex::ParallelFor(bx, [ ne_adv, D_DECL(xflux,yflux,zflux), dxinv,vol]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            ne_adv(i,j,k) =   ((xflux(i+1,j,k) - xflux(i,j,k))
                            + (yflux(i,j+1,k) - yflux(i,j,k))
#if ( AMREX_SPACEDIM ==3 )
                            + (zflux(i,j,k+1) - zflux(i,j,k))
#endif
                              )/ vol(i,j,k)
                            ;
         });

#ifdef AMREX_USE_EB
        // Cell-centered data
        const auto EF_ar = EF_cc.array(mfi);
        const auto gasN_ar = gasN_cc.array(mfi);
        const auto EFmag_ar = EFMag_cc.array(mfi);
        const auto vf = vfrac.array(mfi);

        // Set up cut cell data structures
        int local_i = mfi.LocalIndex();
        int Ncut = (!eb_in_domain) ? 0 : sv_eb_bndry_grad_stencil[local_i].size();
        auto* ebg = (Ncut > 0 ? sv_eb_bndry_geom[local_i].data() : nullptr);
  
        // Get the full area/volume used to calculate EB surface area/cut cell divergence
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const amrex::Real full_area = std::pow(dx[0], AMREX_SPACEDIM - 1);
        amrex::Real full_vol = 1.0;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
          full_vol *= dx[dir];
        }

        // Loop over all cut cells in mfi
        amrex::ParallelFor(Ncut, [=] AMREX_GPU_DEVICE(int L) {
          // Get cut cell indices
          AMREX_D_TERM(const int i = ebg[L].iv[0];, const int j = ebg[L].iv[1]; , const int k = ebg[L].iv[2];)
      
          // TODO: find better way to keep from indexing out of various MFs
          if(i >= bx.smallEnd(0) && i < bx.bigEnd(0) && j >= bx.smallEnd(1) && j < bx.bigEnd(1) && k >= bx.smallEnd(2) && k < bx.bigEnd(2) ){
            // Set up other local variables
            amrex::Real ebflux;
            double Te;
            double EoN = EFmag_ar(i,j,k) / gasN_ar(i,j,k);
            amrex::Real mwt[NUM_SPECIES];
            auto eos = pele::physics::PhysicsType::eos();
            eos.molecular_weight(mwt);
            ExtrapTe(EoN, &Te);
      
            // Calculate the electron flux into the EB
            ebflux = -0.5 * ne_arr(i,j,k) * std::pow( (8.0*EFConst::kB*Te)/(mwt[E_ID]/EFConst::Na * constants::PI()), 0.5 );
            // if(i == 4 && k == 4) printf("ebflux (%i) = %.6e\n", j, ebflux);
            ebflux -= 2.0 * secondary_em_coef * ionFeb(i,j,k); 
            // if(i == 4 && k == 4) printf("eb ion flux (%i) = %.6e\n", j, -2.0 * secondary_em_coef * ionFeb(i,j,k) * ebg[L].eb_area * full_area);
            ebflux *= -1.0 * ebg[L].eb_area * full_area;
            // if(i == 4 && k == 4) printf("ebflux 2 (%i) = %.6e\n", j, ebflux);

            // TODO: Flux interpolation to face centroids...

            // Update the divergence for cut cells
            amrex::Real cutvol = amrex::max<amrex::Real>(vf(i,j,k), 1.0e-12) * full_vol;
            ne_adv(i,j,k) =   ((xflux(i+1,j,k) - xflux(i,j,k))
                              + (yflux(i,j+1,k) - yflux(i,j,k))
#if ( AMREX_SPACEDIM ==3 )
                              + (zflux(i,j,k+1) - zflux(i,j,k))
#endif
                              + ebflux)/ cutvol;
          }
        });
#endif
      }
   }                                         

   elecAdv.mult(-1.0);
  

}

void PeleC::ef_setUpPrecond (const Real &dt_lcl,
                             const MultiFab& a_nl_state) {    
   BL_PROFILE("PLM_EF::ef_setUpPrecond()");

   // amrex::Print() << " STARTED PC SETUP: " << "\n";

   if ( PCLinOp_needUpdate ) {
      LPInfo info;
      info.setAgglomeration(1);
      info.setConsolidation(1);
      info.setMetricTerm(false);

      if ( pnp_pc_drift != nullptr ) {
         delete pnp_pc_drift;
         delete pnp_pc_Stilda;
         delete pnp_pc_diff;
      }

#ifdef PELEC_USE_EB
      const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());

      pnp_pc_drift = new MLEBABecLap({geom}, {grids}, {dmap}, info, {ebf});
      pnp_pc_Stilda = new MLEBABecLap({geom}, {grids}, {dmap}, info, {ebf});
      pnp_pc_diff = new MLEBABecCecLap({geom}, {grids}, {dmap}, info, {ebf});
#else
      pnp_pc_drift = new MLABecLaplacian({geom}, {grids}, {dmap}, info);
      pnp_pc_Stilda = new MLABecLaplacian({geom}, {grids}, {dmap}, info);
      pnp_pc_diff = new MLABecCecLaplacian({geom}, {grids}, {dmap}, info);
#endif
      pnp_pc_Stilda->setMaxOrder(ef_PoissonMaxOrder);
      pnp_pc_diff->setMaxOrder(ef_PoissonMaxOrder);
      pnp_pc_drift->setMaxOrder(ef_PoissonMaxOrder);

      PCLinOp_needUpdate = 0;
   }

   // Set diff/drift operator
   {
      pnp_pc_diff->setScalars(-nE_scale/FnE_scale, -dt_lcl*nE_scale/FnE_scale, dt_lcl*nE_scale/FnE_scale);
      Real omega = 0.7;
      pnp_pc_diff->setRelaxation(omega);
      pnp_pc_diff->setACoeffs(0, 1.0);
      std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(De_ec[0],De_ec[1],De_ec[2])};
      pnp_pc_diff->setBCoeffs(0, bcoeffs);
      std::array<const MultiFab*,AMREX_SPACEDIM> ccoeffs{AMREX_D_DECL(&elec_Ueff[0],&elec_Ueff[1],&elec_Ueff[2])};
      pnp_pc_diff->setCCoeffs(0, ccoeffs);
   }

   MultiFab diagDiff;
   if ( ef_PC_approx == 2 || ef_PC_approx == 3) {
      diagDiff.define(grids,dmap,1,1);
#ifndef PELEC_USE_EB
      pnp_pc_diff->getDiagonal(diagDiff);
      diagDiff.mult(FnE_scale/nE_scale);
      diagDiff.FillBoundary(0,1, geom.periodicity());
      Extrapolater::FirstOrderExtrap(diagDiff, geom, 0, 1);
      if ( ef_PC_approx == 3) {
         diagDiff.plus(-1.0,0,1,1);
      }
#endif
   }

   // Stilda and Drift LinOp
   {
      // Upwinded edge neKe values
      MultiFab nEKe(grids,dmap,1,1);
      MultiFab nE_a(a_nl_state,amrex::make_alias,1,1);  // State is not scale at this point
      MultiFab Schur_nEKe;
      if ( ef_PC_approx == 2 ) {
         Schur_nEKe.define(grids,dmap,1,1);
      }

      // Get molecular weights needed to evaluate transport properties
      amrex::Real mwt[NUM_SPECIES];
      auto eos = pele::physics::PhysicsType::eos();
      eos.molecular_weight(mwt);   // CGS

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(nEKe, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
         const Box& gbx = mfi.growntilebox();
         auto const& neke   = nEKe.array(mfi);
         auto const& ne_arr = nE_a.const_array(mfi);
         auto const& rhoY = Sborder.array(mfi,UFS);
         auto const& redEfab = redEfield.array(mfi);
         auto const& Schur  = ( ef_PC_approx == 2 ) ? Schur_nEKe.array(mfi) : nEKe.array(mfi);
         auto const& diag_a = ( ef_PC_approx == 2 ) ? diagDiff.array(mfi) : nEKe.array(mfi);
         int do_Schur = ( ef_PC_approx == 2 ) ? 1 : 0;
         amrex::ParallelFor(gbx, [neke,Schur,diag_a,ne_arr,dt_lcl,do_Schur,rhoY,redEfab,mwt]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            getKappaE(i,j,k,0,neke,redEfab,rhoY,mwt);
            neke(i,j,k) *= ne_arr(i,j,k) * -1.0;  // invert sign since getKappaE return negative kappa_e
#ifndef PELEC_USE_EB
            if ( do_Schur ) {
               Schur(i,j,k) = - dt_lcl * 0.5 * neke(i,j,k) / diag_a(i,j,k);
            }
#endif
         });
      }
      if ( ef_debug ) {
         if (ef_PC_approx == 2) VisMF::Write(Schur_nEKe,"PC_SchurnEKe_cc_lvl"+std::to_string(level));
         VisMF::Write(nEKe,"PC_nEKe_cc_lvl"+std::to_string(level));
      }

      FluxBoxes edge_fb(this, 1, 1);
      MultiFab** neKe_ec = edge_fb.get();
      FluxBoxes Schur_edge_fb(this, 1, 1);
      MultiFab** Schur_neKe_ec = Schur_edge_fb.get();
      const BCRec& bcrec = get_desc_lst()[State_Type].getBC(nE);
      const Box& domain = geom.Domain();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(nEKe, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
         for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
         {
            const Box ebx = mfi.nodaltilebox(dir);
            const Box& edomain = amrex::surroundingNodes(domain,dir);
            const auto& neke_c    = nEKe.array(mfi);
            const auto& neke_ed   = neKe_ec[dir]->array(mfi);
            const auto& Schur_c   = ( ef_PC_approx == 2 ) ? Schur_nEKe.array(mfi) : nEKe.array(mfi);
            const auto& Schur_ed  = ( ef_PC_approx == 2 ) ? Schur_neKe_ec[dir]->array(mfi) : neKe_ec[dir]->array(mfi);
            const auto& ueff_ed   = elec_Ueff[dir].array(mfi);
            const auto bc_lo = bcrec.lo(dir);
            const auto bc_hi = bcrec.hi(dir);
            int do_Schur = ( ef_PC_approx == 2 ) ? 1 : 0;
            amrex::ParallelFor(ebx, [dir, bc_lo, bc_hi, neke_c, neke_ed, Schur_c, Schur_ed, ueff_ed, edomain, do_Schur]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
               int idx[3] = {i,j,k};
               bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && ( idx[dir] <= edomain.smallEnd(dir) ) );
               bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && ( idx[dir] >= edomain.bigEnd(dir) ) );
               cen2edg_upwind_cpp( i, j, k, dir, 1, on_lo, on_hi, ueff_ed, neke_c, neke_ed);
               if ( do_Schur ) cen2edg_upwind_cpp( i, j, k, dir, 1, on_lo, on_hi, ueff_ed, Schur_c, Schur_ed);
            });
         }
      }

      pnp_pc_drift->setScalars(0.0,0.5*phiV_scale/FnE_scale*dt_lcl);
      {
         std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(neKe_ec[0],neKe_ec[1],neKe_ec[2])};
         pnp_pc_drift->setBCoeffs(0, bcoeffs);
      }
      if ( ef_debug ) {
         VisMF::Write(nEKe,"PC_Drift_nEKe_CC_lvl"+std::to_string(level));
         VisMF::Write(*neKe_ec[0],"PC_Drift_nEKe_edgeX_lvl"+std::to_string(level));
         VisMF::Write(*neKe_ec[1],"PC_Drift_nEKe_edgeY_lvl"+std::to_string(level));
      }

      if ( ef_PC_approx == 1 ) {                      // Simple identity approx
         pnp_pc_Stilda->setScalars(0.0,-1.0);
         Real scalLap = EFConst::eps0_cgs * EFConst::epsr / EFConst::elemCharge;
         for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            neKe_ec[dir]->mult(0.5*dt_lcl,0,1);
            neKe_ec[dir]->plus(scalLap,0,1);
         }
         if ( ef_debug ) {
            VisMF::Write(*neKe_ec[0],"PC_Stilda_nEKepLap_edgeX_lvl"+std::to_string(level));
            VisMF::Write(*neKe_ec[1],"PC_Stilda_nEKepLap_edgeY_lvl"+std::to_string(level));
         }
         {
            std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(neKe_ec[0],neKe_ec[1],neKe_ec[2])};
            pnp_pc_Stilda->setBCoeffs(0, bcoeffs);
         }
      } else if ( ef_PC_approx == 2 ) {               // Inverse diagonal approx
         pnp_pc_Stilda->setScalars(0.0,-1.0);
         Real scalLap = EFConst::eps0_cgs * EFConst::epsr / EFConst::elemCharge;
         for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            Schur_neKe_ec[dir]->plus(scalLap,0,1);
         }
         if ( ef_debug ) {
            VisMF::Write(*Schur_neKe_ec[0],"PC_Stilda2_nEKepLap_edgeX_lvl"+std::to_string(level));
            VisMF::Write(*Schur_neKe_ec[1],"PC_Stilda2_nEKepLap_edgeY_lvl"+std::to_string(level));
         }
         {
            std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(Schur_neKe_ec[0],Schur_neKe_ec[1],Schur_neKe_ec[2])};
            pnp_pc_Stilda->setBCoeffs(0, bcoeffs);
         }
      } else if ( ef_PC_approx == 3 ) {               // Inverse diagonal approx
         pnp_pc_Stilda->setScalars(0.0,-1.0);
      }
   }

   // Set up the domainBCs
   std::array<LinOpBCType,AMREX_SPACEDIM> ne_lobc, ne_hibc;
   std::array<LinOpBCType,AMREX_SPACEDIM> phiV_lobc, phiV_hibc;
   ef_set_PoissonBC(phiV_lobc, phiV_hibc);
   ef_set_neBC(ne_lobc,ne_hibc);
   pnp_pc_Stilda->setDomainBC(phiV_lobc, phiV_hibc);
   pnp_pc_diff->setDomainBC(ne_lobc, ne_hibc);
   pnp_pc_drift->setDomainBC(phiV_lobc, phiV_hibc);

   // Trigger update of the MLMGs
   PCMLMG_needUpdate = 1;

   // amrex::Print() << " FINISHED PC SETUP: " << "\n";
}

void PeleC::ef_applyPrecond (const MultiFab  &v,
                                   MultiFab  &Pv) {
   BL_PROFILE("PC_EF::ef_applyPrecond()");
 
   // amrex::Print() << " STARTED PC APPLY: " << "\n";

   //MultiFab::Copy(Pv,v,0,0,2,v.nGrow());
   //return;

   Real vNorm;
   ef_normMF(v,vNorm);

   // Set up some aliases to make things easier
   MultiFab a_ne(v,amrex::make_alias,1,1);
   MultiFab a_phiV(v,amrex::make_alias,0,1);
   MultiFab a_Pne(Pv,amrex::make_alias,1,1);
   MultiFab a_PphiV(Pv,amrex::make_alias,0,1);

   // TODO: I need to initialize the result to zero otherwise MLMG goes nuts
   // or do I ?
   a_Pne.setVal(0.0,0,1,1);
   a_PphiV.setVal(0.0,0,1,1);

   // Set up the linear solvers BCs
   pnp_pc_diff->setLevelBC(0, &a_Pne);
   pnp_pc_drift->setLevelBC(0, &a_PphiV);
   pnp_pc_Stilda->setLevelBC(0, &a_PphiV);

   // for (MFIter mfi(a_PphiV,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.growntilebox();
   //     const auto& phi_ar = a_PphiV.array(mfi);
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //        if(i == 4 && k == 4 && j == -1) printf("PRECON: LOWER PHI DOMAIN BC = %.6e\n", phi_ar(i,j,k));
   //        if(i == 4 && k == 4 && j == 256) printf("PRECON: UPPER PHI DOMAIN BC = %.6e\n", phi_ar(i,j,k));
   //     });
   // }

#ifdef PELEC_USE_EB
   // Use same approach for EBs as with domain BCs?
   MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
   beta.setVal(1.0);

   // Set the inhomogenous Dirichlet EB BCs for delta phiV
   // Dirichlet for phi is set to zero, since delta update should not update soln at boundary
   // FIXME: Zero Neumann for diff (ABC) operator for now?
   MultiFab phiV_BC(grids, dmap, 1, 0, MFInfo(), Factory());
   phiV_BC.setVal(0.0);
   // const ProbParmDevice* lprobparm = d_prob_parm_device;
   // for (MFIter mfi(beta,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.growntilebox();
   //     const auto& phiV_ar = phiV_BC.array(mfi);
   //     const Real* dx      = geom.CellSize();
   //     const Real* problo  = geom.ProbLo();
   //     const Real* probhi  = geom.ProbHi();
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //         Real y = problo[1] + (j + 0.5)*dx[1];
   //         if (y >= probhi[1] / 2.0) {
   //             // phiV_ar(i,j,k) = lprobparm->PhiV_top;
   //             phiV_ar(i,j,k) = 0.0;
   //         } else {
   //             phiV_ar(i,j,k) = lprobparm->PhiV_bottom;
   //             phiV_ar(i,j,k) = 0.0;
   //         }
   //     });
   // }
   
   pnp_pc_diff->setEBDirichlet(0,a_Pne,beta);
   pnp_pc_drift->setEBDirichlet(0,phiV_BC,beta);
   pnp_pc_Stilda->setEBDirichlet(0,phiV_BC,beta);
   // pnp_pc_drift->setEBDirichlet(0,a_PphiV,beta);
   // pnp_pc_Stilda->setEBDirichlet(0,a_PphiV,beta);
#endif

   // Set Coarse/Fine BCs
   // Assumes it should be at zero.
   if ( level > 0 ) {
      pnp_pc_diff->setCoarseFineBC(nullptr, crse_ratio[0]);
      pnp_pc_drift->setCoarseFineBC(nullptr, crse_ratio[0]);
      pnp_pc_Stilda->setCoarseFineBC(nullptr, crse_ratio[0]);
   }

   // Create MLMGs
   if ( PCMLMG_needUpdate ) {
      if ( mg_diff != nullptr ) {
         delete mg_diff;
         delete mg_drift;
         delete mg_Stilda;
      }
      mg_diff = new MLMG(*pnp_pc_diff);
      mg_drift = new MLMG(*pnp_pc_drift);
      mg_Stilda = new MLMG(*pnp_pc_Stilda);

      PCMLMG_needUpdate = 0;
   }

   mg_diff->setVerbose(2);
   mg_drift->setVerbose(0);
   mg_Stilda->setVerbose(2);
   if ( ef_PC_fixedIter > 0 ) {
      mg_diff->setFixedIter(ef_PC_fixedIter);
      mg_drift->setFixedIter(ef_PC_fixedIter);
      mg_Stilda->setFixedIter(ef_PC_fixedIter);
   }


   Real S_tol     = ef_PC_MG_Tol;
   Real S_tol_abs = a_ne.norm0() * ef_PC_MG_Tol;

   amrex::Print() << " STARTED DIFF MAT SOLVE: " << "\n";

   // Most inner mat
   // --                --
   // | [dtD-I]^-1     0 |
   // |                  |
   // |       0        I |
   // --                --
   // for (MFIter mfi(a_ne,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.tilebox();
   //     const auto& ne_ar = a_ne.array(mfi);
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //       printf("(%i %i %i) a_ne = %.6e\n", i, j, k, ne_ar(i,j,k));
   //     });
   // }
   mg_diff->solve({&a_Pne}, {&a_ne}, S_tol, S_tol_abs);
   MultiFab::Copy(a_PphiV,a_phiV,0,0,1,0);

   amrex::Print() << " FINISHED DIFF MAT SOLVE: " << "\n";

   // Assembling mat
   // --       --
   // |  I    0 |
   // |         |
   // | -Ie   I |
   // --       --
   MultiFab::Saxpy(a_PphiV,nE_scale/FphiV_scale,a_Pne,0,0,1,0);

   // PhiV estimate mat
   // --         --
   // | I     0   |
   // |           |
   // | 0   S*^-1 |
   // --         --
   amrex::Print() << " STARTED STILDE MAT SOLVE: " << "\n";

   MultiFab temp(grids,dmap,1,1);
   temp.setVal(0.0,0,1,0);
   // Scale the solve RHS
   a_PphiV.mult(FphiV_scale/phiV_scale);
   S_tol_abs = a_PphiV.norm0() * ef_PC_MG_Tol;
   mg_Stilda->solve({&temp},{&a_PphiV}, S_tol, S_tol_abs);
   MultiFab::Copy(a_PphiV, temp, 0, 0, 1, 0);

   amrex::Print() << " FINISHED STILDE MAT SOLVE: " << "\n";

   // Final mat
   // --                          --
   // | I       -[dtD - I]^-1 dtDr |
   // |                            |
   // | 0                I         |
   // --                          --

   mg_drift->apply({&temp},{&a_PphiV});
   S_tol_abs = temp.norm0() * ef_PC_MG_Tol;
   MultiFab temp2(grids,dmap,1,1);
   temp2.setVal(0.0,0,1,0);
   amrex::Print() << " STARTED SECOND DIFF MAT SOLVE: " << "\n";
   mg_diff->solve({&temp2},{&temp}, S_tol, S_tol_abs);
   temp2.mult(-1.0);
   MultiFab::Add(a_Pne,temp2,0,0,1,0);

   amrex::Print() << " FINISHED PC APPLY: " << "\n";
}

void PeleC::ef_normMF(const MultiFab &a_vec,
                            Real &norm){
                
    norm = 0.0;
    for ( int comp = 0; comp < a_vec.nComp(); comp++ ) {
       norm += MultiFab::Dot(a_vec,comp,a_vec,comp,1,0);
    }
    norm = std::sqrt(norm);

   // for (MFIter mfi(a_vec,true); mfi.isValid(); ++mfi)
   // {
   //     const Box& bx = mfi.growntilebox();
   //     const auto& vec_ar = a_vec.array(mfi);
   //     amrex::ParallelFor(bx,
   //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
   //     {
   //        if(i == 4 && k == 4) printf("dot term (%i) comp1 = %.6e, comp2 = %.6e\n", j, vec_ar(i,j,k,0) * vec_ar(i,j,k,0));
   //     });
   // }
      for ( int comp = 0; comp < a_vec.nComp(); comp++ ) {
        for (MFIter mfi(a_vec,true); mfi.isValid(); ++mfi)
        {
            int nghost = 0;
            int numcomp = 1;
            Box const& bx = mfi.growntilebox(nghost);
            Array4<Real const> const& xfab = a_vec.const_array(mfi);
            Array4<Real const> const& yfab = a_vec.const_array(mfi);
            AMREX_LOOP_4D(bx, numcomp, i, j, k, n,
            {
                // if(i == 4 && k == 4 && comp == 1) printf("(%i) comp %i val = %.6e\n", j, comp, xfab(i,j,k,comp+n) * yfab(i,j,k,comp+n));
            });
        }
      }
}

void PeleC::ef_normMFv(const MultiFab &a_vec,
                             Vector<Real> &norm){
    for ( int comp = 0; comp < a_vec.nComp(); comp++ ) {
       norm.push_back(MultiFab::Dot(a_vec,comp,a_vec,comp,1,0));
    }
}

void PeleC::compute_bg_charge(const Real &dt_lcl,
                              const MultiFab &state_old,
                              const MultiFab &MOL_src,
                              const MultiFab &I_R) {

   BL_PROFILE("PC_EF::compute_bg_charge()");
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif   
   for (MFIter mfi(bg_charge,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
      const Box& bx = mfi.tilebox();
      auto const& rhoYold  = state_old.const_array(mfi,FirstSpec);
      auto const& srcRhoY  = state_old.const_array(mfi,FirstSpec);
      auto const& reacRhoY = I_R.const_array(mfi,0);
      auto const& charge   = bg_charge.array(mfi);
      Real        factor = 1.0 / EFConst::elemCharge;
      amrex::ParallelFor(bx, [rhoYold,srcRhoY,reacRhoY,charge,dt_lcl,factor,zk,do_react]
      AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
         charge(i,j,k) = 0.0;
         for (int n = 0; n < NUM_SPECIES; n++) {
            Real rhoYpred = rhoYold(i,j,k,n) + dt_lcl * srcRhoY(i,j,k,n);
            if (do_react) rhoYpred += dt_lcl * reacRhoY(i,j,k,n);
            rhoYpred = amrex::max(rhoYpred,0.0);
            charge(i,j,k) += zk[n] * rhoYpred;
         }
         charge(i,j,k) *= factor;
      });
   }
}

void PeleC::compute_gasN(const Real &dt_lcl,
                         const MultiFab &state_old,
                         const MultiFab &MOL_src,
                         const MultiFab &I_R) {

   BL_PROFILE("PC_EF::compute_gasN()");

   // Get a reaction MF with an extrapolated ghost cell layer
   MultiFab I_R_GC(grids,dmap,I_R.nComp(),1);
   if (do_react) {
      MultiFab::Copy(I_R_GC,I_R,0,0,I_R.nComp(),0);
      I_R_GC.FillBoundary(0,I_R.nComp(),geom.periodicity());
      Extrapolater::FirstOrderExtrap(I_R_GC, geom, 0, I_R.nComp());
   }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif   
   for (MFIter mfi(gasN_cc,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
      const Box& bx = mfi.growntilebox();
      auto const& rhoYold  = state_old.const_array(mfi,FirstSpec);
      auto const& srcRhoY  = state_old.const_array(mfi,FirstSpec);
      auto const& reacRhoY = I_R_GC.const_array(mfi,0);
      auto const& gasN     = gasN_cc.array(mfi);
      amrex::ParallelFor(bx, [rhoYold,srcRhoY,reacRhoY,gasN,dt_lcl,do_react]
      AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
         gasN(i,j,k) = 0.0;
         Real mwt[NUM_SPECIES] = {0.0};
         auto eos = pele::physics::PhysicsType::eos();
         eos.molecular_weight(mwt);
         for (int n = 0; n < NUM_SPECIES; n++) {
            Real rhoYpred = rhoYold(i,j,k,n) + dt_lcl * srcRhoY(i,j,k,n);
            if (do_react) rhoYpred += dt_lcl * reacRhoY(i,j,k,n);
            gasN(i,j,k) += rhoYpred * EFConst::Na / mwt[n];
         }
      });
   }

   const Box& domain = geom.Domain();
   bool use_harmonic_avg = def_harm_avg_cen2edge ? true : false;
   const BCRec& bcrec = get_desc_lst()[State_Type].getBC(nE);
 #ifdef _OPENMP
 #pragma omp parallel if (Gpu::notInLaunchRegion())
 #endif
   for (MFIter mfi(gasN_cc,TilingIfNotGPU()); mfi.isValid();++mfi)
   {
      for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
      {
         const Box ebx = mfi.nodaltilebox(dir);
         const Box& edomain = amrex::surroundingNodes(domain,dir);
         const auto& gasN_c  = gasN_cc.array(mfi);
         const auto& gasN_ed = gasN_ec[dir]->array(mfi);
         const auto bc_lo = bcrec.lo(dir);
         const auto bc_hi = bcrec.hi(dir);
         amrex::ParallelFor(ebx, [dir, bc_lo, bc_hi, use_harmonic_avg, gasN_c, gasN_ed, edomain]
         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
            int idx[3] = {i,j,k};
            bool on_lo = ( ( bc_lo == amrex::BCType::ext_dir ) && idx[dir] <= edomain.smallEnd(dir) );
            bool on_hi = ( ( bc_hi == amrex::BCType::ext_dir ) && idx[dir] >= edomain.bigEnd(dir) );
            cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, gasN_c, gasN_ed);
         });
      }
   }
}
