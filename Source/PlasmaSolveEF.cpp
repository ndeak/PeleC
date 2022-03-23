#include <PeleC.H>
#include <AMReX_MLABecLaplacian.H>
#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#endif
#include <Plasma_K.H>
#include <PlasmaBCFill.H>
#include <Plasma.H>

#include <cmath>

using namespace amrex;
using std::string;

void
PeleC::solveEF ( Real time,
                 Real dt,
                 ProbParmDevice const& prob_parm,
                 const amrex::MultiFab& Sbord, 
                 bool lapl_solve)
{
   BL_PROFILE("PeleC::solveEF()");

   amrex::Print() << "Solving for electric field \n";

   Real prev_time = state[State_Type].prevTime();

// Get current PhiV
   MultiFab& Ucurr = (time == prev_time) ? get_old_data(State_Type) : get_new_data(State_Type);

// Build a PhiV with 1 GC properly filled. FillPatch not working in this case.
   MultiFab Sborder(grids, dmap, 1, 1, amrex::MFInfo(), Factory());
   amrex::MultiFab::Copy(Sborder   ,Ucurr  ,PhiV,0,1,0);
   Sborder.FillBoundary(geom.periodicity());
   const BCRec& bcphiV = get_desc_lst()[State_Type].getBC(PhiV);
   const Vector<BCRec>& bc = {bcphiV};
   if (not geom.isAllPeriodic()) {
      const ProbParmDevice* lprobparm = d_prob_parm_device;
      amrex::GpuBndryFuncFab<PhiVFill>  bf(PhiVFill{lprobparm});
      PhysBCFunct<GpuBndryFuncFab<PhiVFill> > phiVf(geom, bc, bf);
      phiVf(Sborder, 0, 1, Sborder.nGrowVect(), time, 0);
   }

   MultiFab phiV_alias(Ucurr, amrex::make_alias, PhiV, 1);
   MultiFab phiV_borders(Sborder, amrex::make_alias, 0, 1);
   // VisMF::Write(phiV_borders,"phiv");

   amrex::Real mwt[NUM_SPECIES];
   auto eos = pele::physics::PhysicsType::eos();
   eos.molecular_weight(mwt);   // CGS

// Charge distribution MF
   MultiFab chargeDistrib(grids,dmap,1,0,MFInfo(),Factory());

#ifdef _OPENMP
#pragma omp parallel
#endif

   // TODO set charge to be equal to sum of ion/electron num densities
   for (MFIter mfi(chargeDistrib,true); mfi.isValid(); ++mfi)
   {   
       const Box& bx = mfi.tilebox();
       const auto& rhoY_ar = Ucurr.array(mfi,UFS);
       const auto& nE_ar   = Ucurr.array(mfi,UFX+1);
       const auto& chrg_ar = chargeDistrib.array(mfi);
       const Real* dx      = geom.CellSize();
       const Real* problo  = geom.ProbLo();
       Real        factor = -1.0 * EFConst::elemCharge / ( EFConst::eps0_cgs  * EFConst::epsr);
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
          if(ef_noSpaceCharge == 0 && !lapl_solve){
            Real tmp_chrg = 0.0;
            Real tmp_val = 0.0;
            for(int n=0; n<NUM_SPECIES; n++) {
              if(n == E_ID) {
                tmp_val = (ef_use_NLsolve == 1 || ef_use_nEimplicit) ? -1.0*nE_ar(i,j,k) : rhoY_ar(i,j,k,n) * (1.0/mwt[n]) * EFConst::Na * zk_num[n];
              }
              else{
                tmp_val = rhoY_ar(i,j,k,n) * (1.0/mwt[n]) * EFConst::Na * zk_num[n];
              }
              tmp_chrg += tmp_val;
            }
            chrg_ar(i,j,k) = tmp_chrg * factor;
          }
          else{
            chrg_ar(i,j,k) = 0.0;
          }
       }); 
   }

  // If using semi-implicit system, start by extrapolating diffusivites from cell centers to edges
  // No need to consider cell flags, since EB Poisson solver must take care of this?
  if(ef_semiImpEfield == 1 && ef_noSpaceCharge != 1){
    spec_edge_mfs = spec_edge.get();
    std::array<amrex::MultiFab*,AMREX_SPACEDIM> spec_edge_arr{AMREX_D_DECL(spec_edge_mfs[0], spec_edge_mfs[1], spec_edge_mfs[2])};
    average_cellcenter_to_face(spec_edge_arr, coeffs_old, geom, NUM_SPECIES+3);
    // Calculate necessary edge values
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        const amrex::GpuArray<const int, 3> bdim{{idim == 0, idim == 1, idim == 2}};
        for (MFIter mfi(*spec_edge_mfs[idim],true); mfi.isValid(); ++mfi)
        {   
            const Box& bx = mfi.tilebox();
            const auto& rhoY_ar = Sbord.array(mfi,UFS);
            const auto& rho_ar = Sbord.array(mfi,0);
            const auto& spec_edge_ar = spec_edge_mfs[idim]->array(mfi);
            const Real* dx      = geom.CellSize();
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const int ii = i - bdim[0];
                const int jj = j - bdim[1];
                const int kk = k - bdim[2];
  
                for (int n = 0; n<NUM_SPECIES; n++){
                  // Divide by density to get rhoD -> D
                  spec_edge_ar(i,j,k,n) /= (rho_ar(i,j,k) + rho_ar(ii,jj,kk)) / 2.0;
                  
                  // Multiply by dn/dx
                  if(n == E_ID && (ef_use_NLsolve == 1 || ef_use_nEimplicit)){
                    spec_edge_ar(i,j,k,n) *= (rho_ar(i,j,k,UFX+1) - rho_ar(i,j,k,UFX+1)) / dx[idim];
                  }
                  else{
                    spec_edge_ar(i,j,k,n) *= (rhoY_ar(i,j,k,n) - rhoY_ar(i,j,k,n)) / dx[idim] * (1.0/mwt[n]) * EFConst::Na;
                  }
                }
            }); 
        }
    }
  
    // Modify the RHS to account for divergence of diffusive term
    amrex::Real vol = 1;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      vol *= geom.CellSize()[dir];
    }

    for (MFIter mfi(chargeDistrib,true); mfi.isValid(); ++mfi)
    {   
        const Box& bx = mfi.tilebox();
        const auto& chrg_ar = chargeDistrib.array(mfi);
        const auto& sec_x_ar = spec_edge_mfs[0]->array(mfi);
        const auto& sec_y_ar = spec_edge_mfs[1]->array(mfi);
        const auto& sec_z_ar = spec_edge_mfs[2]->array(mfi);
#ifdef PELEC_USE_EB
        const auto& vf = vfrac.array(mfi);
#endif
        const amrex::Real volinv = 1.0 / vol;
        Real        factor = -1.0 * EFConst::elemCharge / ( EFConst::eps0_cgs  * EFConst::epsr);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          for(int n = 0; n<NUM_SPECIES; n++){

#ifdef PELEC_USE_EB
            const amrex::Real kappa_inv = 1.0 / amrex::max<amrex::Real>(vf(i,j,k), 1.0e-12);
#else
            const amrex::Real kappa_inv = 1.0;
#endif
            amrex::Real difftemp =
              -(AMREX_D_TERM(
                sec_x_ar(i+1,j,k,n) - sec_x_ar(i,j,k,n), +sec_y_ar(i,j+1,k,n) - sec_y_ar(i,j,k,n),
                +sec_z_ar(i,j,k+1,n) - sec_z_ar(i,j,k,n)))  * volinv * kappa_inv;;

             chrg_ar(i,j,k) -= factor * dt * zk_num[n] * difftemp;
          }
        }); 
    }
  }

// If need be, visualize the charge distribution.
//   VisMF::Write(chargeDistrib,"chargeDistribPhiV_"+std::to_string(level));

/////////////////////////////////////   
// Setup a linear operator
/////////////////////////////////////   

   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);

// Linear operator (EB aware if need be)
#ifdef AMREX_USE_EB
    const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
    MLEBABecLap poissonOP({geom}, {grids}, {dmap}, info, {ebf});
#else
    MLABecLaplacian poissonOP({geom}, {grids}, {dmap}, info);
#endif

   poissonOP.setMaxOrder(2);

// Boundary conditions for the linear operator.
   std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
   std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
   setBCPhiV(bc_lo,bc_hi);
   poissonOP.setDomainBC(bc_lo,bc_hi);   

// Get the coarse level data for AMR cases.
   std::unique_ptr<MultiFab> phiV_crse;
   if (level > 0) {
      auto& crselev = getLevel(level-1);
      phiV_crse.reset(new MultiFab(crselev.boxArray(), crselev.DistributionMap(), 1, 0));
      MultiFab& Coarse_State = (time == prev_time) ? crselev.get_old_data(State_Type) : crselev.get_new_data(State_Type);   
      MultiFab::Copy(*phiV_crse, Coarse_State,PhiV,0,1,0);
      poissonOP.setCoarseFineBC(phiV_crse.get(), crse_ratio[0]);
   }

// Pass the phiV with physical BC filled.
   poissonOP.setLevelBC(0, &phiV_borders);

// Setup solver coefficient: general form is (ascal * acoef - bscal * div bcoef grad ) phi = rhs   
// For simple Poisson solve: ascal, acoef = 0 and bscal, bcoef = 1
// For semi-implicit solve, problem becomes a variable coefficient Poisson problem
   MultiFab acoef(grids, dmap, 1, 0, MFInfo(), Factory());
   acoef.setVal(0.0);
   poissonOP.setACoeffs(0, acoef);
   Array<MultiFab,AMREX_SPACEDIM> bcoef;
   for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
       const amrex::GpuArray<const int, 3> bdim{{idim == 0, idim == 1, idim == 2}};
       bcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0, MFInfo(), Factory());
  
       // TODO: may be more concise to use FluxBoxes, cellcenter_to_face, and MutliFab Mutliply utility functions...
       //       going with uglier approach for now
       if(ef_semiImpEfield == 1 && ef_noSpaceCharge == 0){
          for (MFIter mfi(bcoef[idim],true); mfi.isValid(); ++mfi)
          {   
              const Box& bx = mfi.tilebox();
              const auto& rhoY_ar = Sbord.array(mfi,UFS);
              const auto& nE_ar   = Ucurr.array(mfi,UFX+1);
              const auto& mu_ar = KSpec_old.array(mfi);
              const auto& beta_ar = bcoef[idim].array(mfi);
              amrex::Real factor = dt * EFConst::elemCharge / ( EFConst::eps0_cgs  * EFConst::epsr);

              amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
              {
                  const int ii = i - bdim[0];
                  const int jj = j - bdim[1];
                  const int kk = k - bdim[2];
  
                  beta_ar(i,j,k) = 1.0;
                  amrex::Real temp_coef = 0.0;
                  // Calculate edge state as simple average (unclear how to incorporate upwinding...)
                  for (int n = 0; n<NUM_SPECIES; n++){
                    // Recall mu already incorporates charge number
                    if(n == E_ID && (ef_use_NLsolve || ef_use_nEimplicit)){
                      temp_coef += ((nE_ar(i,j,k)*mu_ar(i,j,k,n) + nE_ar(ii,jj,kk)*mu_ar(ii,jj,kk,n)) / 2.0);
                    }
                    else{
                      temp_coef += ((rhoY_ar(i,j,k,n)*mu_ar(i,j,k,n) + rhoY_ar(ii,jj,kk,n)*mu_ar(ii,jj,kk,n)) / 2.0) * (1.0/mwt[n]) * EFConst::Na;
                    }
                  }
                  temp_coef *= factor;
                  beta_ar(i,j,k) -= temp_coef;
              }); 
          }
       }
       else{
          bcoef[idim].setVal(1.0);
       }
       amrex::Real beta_max= bcoef[idim].max(0, 0, false);
       amrex::Real beta_min= bcoef[idim].min(0, 0, false);
   }
   poissonOP.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));   
   Real ascal = 0.0;
   Real bscal = -1.0;
   poissonOP.setScalars(ascal, bscal);

   // set Dirichlet BC for EB
	// TODO : for now set upper y-dir half to X and lower y-dir to 0
	//        will have to find a better way to specify EB dirich values 
#ifdef AMREX_USE_EB
   MultiFab phiV_BC(grids, dmap, 1, 0, MFInfo(), Factory());
   MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
   beta.setVal(1.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
   // EB Dirichlet conditions for plane plane and pin pin
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
           } else {
               phiV_ar(i,j,k) = prob_parm.PhiV_bottom;
           }   
       }); 
   }

   // EB Dirichlet conditions for spherical discharge test case
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
   //         Real x = problo[0] + (i + 0.5)*dx[0]; 
   //         Real y = problo[1] + (j + 0.5)*dx[1]; 
   //         Real z = problo[2] + (k + 0.5)*dx[2]; 
   //         Real r = sqrt((x-2.0)*(x-2.0) + (y-2.0)*(y-2.0) + (z-2.0)*(z-2.0));
   //         if (r >= 1.0) {
   //             phiV_ar(i,j,k) = prob_parm.PhiV_top;     // outer sphere
   //         } else { 
   //             phiV_ar(i,j,k) = prob_parm.PhiV_bottom;  // inner sphere
   //         }   
   //     }); 
   // }

   poissonOP.setEBDirichlet(0,phiV_BC,beta);
#endif

// If need be, visualize the charge distribution.    
//   VisMF::Write(phiV_BC,"EBDirichPhiV_"+std::to_string(level));
    
/////////////////////////////////////   
// Setup a MG solver
/////////////////////////////////////   
   MLMG mlmg(poissonOP);

   phiV_alias.setVal(0.0); // initial guess for phi

   // relative and absolute tolerances for linear solve
   const Real tol_rel = ef_PoissonTol;
   amrex::Print() << "max charge tol = " << chargeDistrib.norm0()*ef_PoissonTol << " , max phiV tol = " << prob_parm.PhiV_top*ef_PoissonTol << ", abs tol = 1.0e-5\n"; 
   const Real tol_abs = std::max(std::max(chargeDistrib.norm0(),phiV_alias.norm0()) * ef_PoissonTol, 1.0e-5);

   mlmg.setVerbose(ef_PoissonVerbose);
   mlmg.setMaxIter(1000);
       
   // Solve linear system
   mlmg.solve({&phiV_alias}, {&chargeDistrib}, tol_rel, tol_abs);

   // Copy solution into interior of border array
   for (MFIter mfi(phiV_alias,true); mfi.isValid(); ++mfi)
   {
       const Box& bx = mfi.tilebox();
       const auto& phiValias_ar = phiV_alias.array(mfi);
       const auto& phiVborders_ar = phiV_borders.array(mfi);
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
         phiVborders_ar(i, j, k) = phiValias_ar(i, j, k);
       });
   }

   // Calculate efield components
   gphi.clear();
   gphi.define(this,1,numGrow());
   gradPhiV = gphi.get();

   gradPhiV[0]->setVal(0.0);
   gradPhiV[1]->setVal(0.0);
#if AMREX_SPACEDIM == 3
   gradPhiV[2]->setVal(0.0);
#endif
   std::array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(gradPhiV[0],gradPhiV[1],gradPhiV[2])};
   mlmg.getGradSolution({fp});

   for(int d=0; d<AMREX_SPACEDIM; d++){
     for (amrex::MFIter mfi(*gradPhiV[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
       const amrex::Box& tbox = mfi.tilebox();
       int ng = gradPhiV[d]->nGrow();
       const amrex::Box gbox = amrex::grow(tbox, ng);
       std::array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> E_edge_arr = {AMREX_D_DECL(gradPhiV[0]->array(mfi), gradPhiV[1]->array(mfi), gradPhiV[2]->array(mfi))} ;
       const Real* problo  = geom.ProbLo();
       const Real* dx      = geom.CellSize();
         amrex::ParallelFor(
           tbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
             E_edge_arr[d](i, j, k, 0) *= -1.0;
           });
       }
   }

   // Need to unscale fluxes for cut cells
   // By default they are scaled by (EB area/uncut cell face area) 

   Efield_edge = {AMREX_D_DECL(gradPhiV[0], gradPhiV[1], gradPhiV[2])};
#ifdef PELEC_USE_EB
   EB_average_face_to_cellcenter(Efield, 0, Efield_edge);
#else
   average_face_to_cellcenter(Efield, 0, Efield_edge);
#endif

  // Copy Efield cell-center values into State variable MF for plotting (TODO : probably better way to do this...)
  for (amrex::MFIter mfi(Ucurr, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
     const amrex::Box& tbox = mfi.tilebox();
     int ng = Ucurr.nGrow();
     const amrex::Box gbox = amrex::grow(tbox, ng);
     const auto Efab = Efield.array(mfi);
     const auto Sfab = Ucurr.array(mfi);
     amrex::ParallelFor(
       tbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         Sfab(i, j, k, UFX+2) = Efab(i, j, k, 0);
         Sfab(i, j, k, UFX+3) = Efab(i, j, k, 1);
#if AMREX_SPACEDIM == 3
         Sfab(i, j, k, UFX+4) = Efab(i, j, k, 2);
#endif
     });
   }

}

// Evaluate the gap capacitance using a nominal applied voltage
void
PeleC::gapCapacitance (Real time)
{
   BL_PROFILE("PeleC::solveEF()");

   amrex::Print() << "Solving for electric field \n";

   Real prev_time = state[State_Type].prevTime();

// Get current PhiV
   MultiFab& Ucurr = get_new_data(State_Type);

// Build a PhiV with 1 GC properly filled. FillPatch not working in this case.
   MultiFab Sborder(grids, dmap, 1, 1, amrex::MFInfo(), Factory());
   amrex::MultiFab::Copy(Sborder   ,Ucurr  ,PhiV,0,1,0);
   Sborder.FillBoundary(geom.periodicity());
   const BCRec& bcphiV = get_desc_lst()[State_Type].getBC(PhiV);
   const Vector<BCRec>& bc = {bcphiV};
   if (not geom.isAllPeriodic()) {
      const ProbParmDevice* lprobparm = d_prob_parm_device;
      amrex::GpuBndryFuncFab<PhiVFill>  bf(PhiVFill{lprobparm});
      PhysBCFunct<GpuBndryFuncFab<PhiVFill> > phiVf(geom, bc, bf);
      phiVf(Sborder, 0, 1, Sborder.nGrowVect(), time, 0);
   }

   MultiFab phiV_alias(Ucurr, amrex::make_alias, PhiV, 1);
   MultiFab phiV_borders(Sborder, amrex::make_alias, 0, 1);
   // VisMF::Write(phiV_borders,"phiv");

   // Charge distribution MF
   MultiFab chargeDistrib(grids,dmap,1,0,MFInfo(),Factory()); chargeDistrib.setVal(0.0);

/////////////////////////////////////   
// Setup a linear operator
/////////////////////////////////////   

   LPInfo info;
   info.setAgglomeration(1);
   info.setConsolidation(1);
   info.setMetricTerm(false);

// Linear operator (EB aware if need be)
#ifdef AMREX_USE_EB
    const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
    MLEBABecLap poissonOP({geom}, {grids}, {dmap}, info, {ebf});
#else
    MLABecLaplacian poissonOP({geom}, {grids}, {dmap}, info);
#endif

   poissonOP.setMaxOrder(2);

// Boundary conditions for the linear operator.
   std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
   std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
   setBCPhiV(bc_lo,bc_hi);
   poissonOP.setDomainBC(bc_lo,bc_hi);   

// Get the coarse level data for AMR cases.
   std::unique_ptr<MultiFab> phiV_crse;
   if (level > 0) {
      auto& crselev = getLevel(level-1);
      phiV_crse.reset(new MultiFab(crselev.boxArray(), crselev.DistributionMap(), 1, 0));
      MultiFab& Coarse_State = crselev.get_new_data(State_Type);   
      MultiFab::Copy(*phiV_crse, Coarse_State,PhiV,0,1,0);
      poissonOP.setCoarseFineBC(phiV_crse.get(), crse_ratio[0]);
   }

// Pass the phiV with physical BC filled.
   poissonOP.setLevelBC(0, &phiV_borders);

// Setup solver coefficient: general form is (ascal * acoef - bscal * div bcoef grad ) phi = rhs   
// For simple Poisson solve: ascal, acoef = 0 and bscal, bcoef = 1
   MultiFab acoef(grids, dmap, 1, 0, MFInfo(), Factory());
   acoef.setVal(0.0);
   poissonOP.setACoeffs(0, acoef);
   Array<MultiFab,AMREX_SPACEDIM> bcoef;
   for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
       bcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0, MFInfo(), Factory());
       bcoef[idim].setVal(1.0);
   }
   poissonOP.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));   
   Real ascal = 0.0;
   Real bscal = -1.0;
   poissonOP.setScalars(ascal, bscal);

   // set Dirichlet BC for EB
	// TODO : for now set upper y-dir half to X and lower y-dir to 0
	//        will have to find a better way to specify EB dirich values 
#ifdef AMREX_USE_EB
   MultiFab phiV_BC(grids, dmap, 1, 0, MFInfo(), Factory());
   MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
   beta.setVal(1.0);
#ifdef _OPENMP
#pragma omp parallel
#endif
   // EB Dirichlet conditions for plane plane and pin pin
   // Note: in this function we use a nominal voltage
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
               phiV_ar(i,j,k) = 1.0;
           } else {
               phiV_ar(i,j,k) = 0.0;
           }   
       }); 
   }

   poissonOP.setEBDirichlet(0,phiV_BC,beta);
#endif

/////////////////////////////////////   
// Setup a MG solver
/////////////////////////////////////   
   MLMG mlmg(poissonOP);

   phiV_alias.setVal(0.0); // initial guess for phi

   // relative and absolute tolerances for linear solve
   const Real tol_rel = ef_PoissonTol;
   const Real tol_abs = 1.0e-8;

   mlmg.setVerbose(ef_PoissonVerbose);
   mlmg.setMaxIter(1000);
       
   // Solve linear system
   mlmg.solve({&phiV_alias}, {&chargeDistrib}, tol_rel, tol_abs);

   // Copy solution into interior of border array
   for (MFIter mfi(phiV_alias,true); mfi.isValid(); ++mfi)
   {
       const Box& bx = mfi.tilebox();
       const auto& phiValias_ar = phiV_alias.array(mfi);
       const auto& phiVborders_ar = phiV_borders.array(mfi);
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
         phiVborders_ar(i, j, k) = phiValias_ar(i, j, k);
       });
   }

   // Calculate efield components
   gphi.clear();
   gphi.define(this,1,numGrow());
   gradPhiV = gphi.get();

   gradPhiV[0]->setVal(0.0);
   gradPhiV[1]->setVal(0.0);
#if AMREX_SPACEDIM == 3
   gradPhiV[2]->setVal(0.0);
#endif
   std::array<MultiFab*,AMREX_SPACEDIM> fp{D_DECL(gradPhiV[0],gradPhiV[1],gradPhiV[2])};
   mlmg.getGradSolution({fp});

   // Need to unscale fluxes for cut cells
   // By default they are scaled by (EB area/uncut cell face area) 

   Efield_edge = {AMREX_D_DECL(gradPhiV[0], gradPhiV[1], gradPhiV[2])};
#ifdef PELEC_USE_EB
   EB_average_face_to_cellcenter(Efield_L_p2, 0, Efield_edge);
#else
   average_face_to_cellcenter(Efield_L_p2, 0, Efield_edge);
#endif

  // Dotting Efield with itself before performing integration
  for (amrex::MFIter mfi(Efield_L_p2, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
     const amrex::Box& tbox = mfi.tilebox();
     const auto Efab = Efield_L_p2.array(mfi);
     amrex::ParallelFor(
       tbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         Efab(i, j, k, 0) = EFConst::eps0_cgs * Efab(i, j, k, 0) * Efab(i, j, k, 0);
         Efab(i, j, k, 1) = EFConst::eps0_cgs * Efab(i, j, k, 1) * Efab(i, j, k, 1);
#if AMREX_SPACEDIM == 3
         Efab(i, j, k, 2) = EFConst::eps0_cgs * Efab(i, j, k, 2) * Efab(i, j, k, 2);
#endif
     });
   }

   // Now performing integration over non-covered component of current level
   level_capacitance = volWgtSumMF(Efield_L_p2, 0, false, true);

   // If we are at the finest level, we need to calculate displacement current as the sum from each previous level
   if(level == parent->finestLevel()){
     int lidx = 0;
     while(lidx < parent->finestLevel()) {
       auto& crselev = getLevel(lidx);
       level_capacitance += crselev.getGapCapacitance();
       lidx++;
     }
     // Now need to set the correct displace current at each coarse level
     lidx = 0;
     while(lidx < parent->finestLevel()) {
       auto& crselev = getLevel(lidx);
       crselev.setGapCapacitance(level_capacitance);
       lidx++;
     }
   }
}

