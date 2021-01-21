#include <PeleC.H>
#include <GMRES.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLABecLaplacian.H>
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
                        const MultiFab &I_R_in)
{
   BL_PROFILE("PC_EF::ef_solve_NL()");

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

   // Compute the background charge distribution
   compute_bg_charge(nl_dt, state_in, forcing_state, I_R_in);
   VisMF::Write(bg_charge,"bg_charge");

   // Newton initial guess

   // Initial NL residual: update residual scaling and preconditioner
   ef_nlResidual( nl_dt, nl_state, nl_resid, true, true );
   nl_resid.mult(-1.0,0,2,2);
   ef_normMF(nl_resid,nl_residNorm);
   if ( ef_debug ) VisMF::Write(nl_resid,"NLResInit_Lvl"+std::to_string(level));

   Abort();
}

void PeleC::ef_nlResidual(const Real      &dt_lcl,
                          const MultiFab  &a_nl_state,
                                MultiFab  &a_nl_resid,
                                int       update_res_scaling,
                                int       update_precond){
   BL_PROFILE("PC_EF::ef_nlResidual()");
   Print() << " Computing plasma NL residual \n";

   // Get the unscaled non-linear state
   MultiFab nl_state_usc(grids,dmap,2,2);
   MultiFab::Copy(nl_state_usc, a_nl_state, 0, 0, 2, 2);
   nl_state_usc.mult(nE_scale,1,1,1);
   nl_state_usc.mult(phiV_scale,0,1,1);

   // Get aliases to make it easier
   MultiFab nE_a(nl_state_usc,amrex::make_alias,1,1);
   MultiFab phi_a(nl_state_usc,amrex::make_alias,1,1);

   // Lap(PhiV) and grad(PhiV)
   FluxBoxes gphi_fb(this, 1, 0);
   MultiFab** gphiV = gphi_fb.get();
   MultiFab laplacian_term(grids, dmap, 1, 0);
   compPhiVLap(phi_a,laplacian_term,gphiV);
   if ( ef_debug ) VisMF::Write(laplacian_term,"NLRes_phiVLap_"+std::to_string(level));

   // Diffusion term nE
   MultiFab diffnE(grids, dmap, 1, 0);
   compElecDiffusion(nE_a,diffnE);
   if ( ef_debug ) VisMF::Write(diffnE,"NLRes_ElecDiff_"+std::to_string(level));

   // Advective term nE
   MultiFab advnE(grids, dmap, 1, 0);
   //compElecAdvection(nE_a,phi_a,gphiV,advnE);
   if ( ef_debug ) VisMF::Write(advnE,"NLRes_ElecAdv_"+std::to_string(level));

   // Assemble the non-linear residual
   // res(ne(:)) = dt * ( diff(:) + conv(:) + I_R(:) ) - ( ne(:) - ne_old(:) )
   // res(phiv(:)) = \Sum z_k * \tilde Y_k / q_e - ne + Lapl_PhiV
   a_nl_resid.setVal(0.0);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   for (MFIter mfi(a_nl_resid,TilingIfNotGPU()); mfi.isValid(); ++mfi)
   {
      const Box& bx = mfi.tilebox();
      auto const& I_R_nE   = get_new_data(Reactions_Type).const_array(mfi,NUM_SPECIES);
      auto const& lapPhiV  = laplacian_term.const_array(mfi);
      auto const& ne_diff  = diffnE.const_array(mfi);
      auto const& ne_adv   = advnE.const_array(mfi);
      auto const& ne_curr  = nE_a.const_array(mfi);
      auto const& ne_old   = ef_state_old.const_array(mfi,1);
      auto const& charge   = bg_charge.const_array(mfi);
      auto const& res_nE   = a_nl_resid.array(mfi,1);
      auto const& res_phiV = a_nl_resid.array(mfi,0);
      Real scalLap         = EFConst::eps0 * EFConst::epsr / EFConst::elemCharge;
      amrex::ParallelFor(bx, [ne_curr,ne_old,lapPhiV,I_R_nE,ne_diff,ne_adv,charge,res_nE,res_phiV,dt_lcl,scalLap]
      AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {    
         res_nE(i,j,k) = ne_old(i,j,k) - ne_curr(i,j,k) + dt_lcl * ( ne_diff(i,j,k) + ne_adv(i,j,k) + I_R_nE(i,j,k) );
         res_phiV(i,j,k) = lapPhiV(i,j,k) * scalLap - ne_curr(i,j,k) + charge(i,j,k);
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

   // Update the preconditioner
   if ( update_precond ) {
      //ef_setUpPrecond(dt_lcl, nl_state_usc);
   }

}

void PeleC::compPhiVLap(MultiFab& phi,
                        MultiFab& phiLap,
                        MultiFab** gPhiV){

// Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
   info.setMaxCoarseningLevel(0);
   MLPoisson phiV_poisson({geom}, {grids}, {dmap}, info);
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
   phiV_poisson.setLevelBC(0, &phi);

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
   // Set-up Poisson operator
   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);
   info.setMaxCoarseningLevel(0);
   MLABecLaplacian ne_lapl({geom}, {grids}, {dmap}, info);
   ne_lapl.setMaxOrder(ef_PoissonMaxOrder);

   // Set-up BC's
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
   std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
   ef_set_neBC(mlmg_lobc, mlmg_hibc);
   ne_lapl.setDomainBC(mlmg_lobc, mlmg_hibc);

   MultiFab nE_crse;
   if (level > 0) {
      auto& crselev = getLevel(level-1);
      nE_crse.define(crselev.grids, crselev.dmap, 1, 0, MFInfo(), crselev.Factory());
//      FillPatch(crselev, nE_crse, 0, curtime, State_Type, nE, 1, 0);
      ne_lapl.setCoarseFineBC(&nE_crse, crse_ratio[0]);
   }
   ne_lapl.setLevelBC(0, &a_ne);

   // Coeffs
   ne_lapl.setScalars(0.0, 1.0);
   Array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{D_DECL(De_ec[0],De_ec[1],De_ec[2])};
   ne_lapl.setBCoeffs(0, bcoeffs);

   // LinearSolver to get divergence
   // Need a copy of ne since the linear operator touches the ghost cells
   MultiFab neOp(grids,dmap,1,2); 
   MultiFab::Copy(neOp,a_ne,0,0,1,2);
   MLMG solver(ne_lapl);
   solver.apply({&elecDiff},{&neOp});

   elecDiff.mult(-1.0);
}

void PeleC::ef_applyPrecond (const MultiFab  &v,
                                   MultiFab  &Pv) {
   BL_PROFILE("PC_EF::ef_applyPrecond()");
}

void PeleC::ef_normMF(const MultiFab &a_vec,
                            Real &norm){
    norm = 0.0;
    for ( int comp = 0; comp < a_vec.nComp(); comp++ ) {
       norm += MultiFab::Dot(a_vec,comp,a_vec,comp,1,0);
    }
    norm = std::sqrt(norm);
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
      amrex::ParallelFor(bx, [rhoYold,srcRhoY,reacRhoY,charge,dt_lcl,factor,zk]
      AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
         charge(i,j,k) = 0.0;
         for (int n = 0; n < NUM_SPECIES; n++) {
            Real rhoYpred = rhoYold(i,j,k,n) + dt_lcl * ( srcRhoY(i,j,k,n) + reacRhoY(i,j,k,n) );
            rhoYpred = amrex::max(rhoYpred,0.0);
            charge(i,j,k) += zk[n] * rhoYpred;
         }
         charge(i,j,k) *= factor;
      });
   }
}
