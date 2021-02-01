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
                 Real dt )
{
   BL_PROFILE("PeleC::solveEF()");

   amrex::Print() << "Solving for electric field \n";

   Real prev_time = state[State_Type].prevTime();

// Get current PhiV
   MultiFab& Ucurr = (time == prev_time) ? get_old_data(State_Type) : get_new_data(State_Type);

// Build a PhiV with 1 GC properly filled. FillPatch not working in this case.
   MultiFab Sborder(grids, dmap, 1, 1, amrex::MFInfo(), Factory());
   Sborder.copy(Ucurr,PhiV,0,1,0,0);
   Sborder.FillBoundary(geom.periodicity());
   const BCRec& bcphiV = get_desc_lst()[State_Type].getBC(PhiV);
   const Vector<BCRec>& bc = {bcphiV};
   if (not geom.isAllPeriodic()) {
      GpuBndryFuncFab<PhiVFill> bf(PhiVFill{});
      PhysBCFunct<GpuBndryFuncFab<PhiVFill> > phiVf(geom, bc, bf);
      phiVf(Sborder, 0, 1, Sborder.nGrowVect(), time, 0);
   }

   MultiFab phiV_alias(Ucurr, amrex::make_alias, PhiV, 1);
   MultiFab phiV_borders(Sborder, amrex::make_alias, 0, 1);

// Setup a dummy charge distribution MF
   MultiFab chargeDistib(grids,dmap,1,0,MFInfo(),Factory());

#ifdef _OPENMP
#pragma omp parallel
#endif
   // TODO set charge to be equal to sum of ion/electron num densities
   for (MFIter mfi(chargeDistib,true); mfi.isValid(); ++mfi)
   {   
       const Box& bx = mfi.tilebox();
       const auto& chrg_ar = chargeDistib.array(mfi);
       const Real* dx      = geom.CellSize();
       const Real* problo  = geom.ProbLo();
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {
           Real x_rel = problo[0] + (i + 0.5)*dx[0] - 2.0;
           Real y_rel = problo[1] + (j + 0.5)*dx[1] - 2.0;
           Real z_rel = problo[2] + (k + 0.5)*dx[2] - 2.0;
           Real r = std::sqrt(x_rel*x_rel+2.0*y_rel*y_rel+z_rel*z_rel);
           if (r < 0.5) {
               //chrg_ar(i,j,k) = 1.0e4*(0.5 - r)/0.5;
               chrg_ar(i,j,k) = 0.0;
           } else {
               chrg_ar(i,j,k) = 0.0;
           }   
       }); 
   }
// If need be, visualize the charge distribution.
//   VisMF::Write(chargeDistib,"chargeDistibPhiV_"+std::to_string(level));

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
   Real bscal = 1.0;
   poissonOP.setScalars(ascal, bscal);

   // set Dirichlet BC for EB
	// TODO : for now set upper y-dir half to X and lower y-dir to 0
	//        will have to find a better way to specify EB dirich values 
#ifdef AMREX_USE_EB
   MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
   beta.setVal(1.0);
   MultiFab phiV_BC(grids, dmap, 1, 0, MFInfo(), Factory());
#ifdef _OPENMP
#pragma omp parallel
#endif
   for (MFIter mfi(beta,true); mfi.isValid(); ++mfi)
   {   
       const Box& bx = mfi.growntilebox();
       const auto& phiV_ar = phiV_BC.array(mfi);
       const Real* dx      = geom.CellSize();
       const Real* problo  = geom.ProbLo();
       amrex::ParallelFor(bx,
       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
       {   
           Real y = problo[1] + (j + 0.5)*dx[1]; 
           if (y >= 5.5) {
               phiV_ar(i,j,k) = 1.0;
           } else {
               phiV_ar(i,j,k) = 0.0;
           }   
       }); 
   }

   poissonOP.setEBDirichlet(0,phiV_BC,beta);
#endif

// If need be, visualize the charge distribution.    
//   VisMF::Write(phiV_BC,"EBDirichPhiV_"+std::to_string(level));
    
/////////////////////////////////////   
// Setup a MG solver
/////////////////////////////////////   
   MLMG mlmg(poissonOP);

   // relative and absolute tolerances for linear solve
   const Real tol_rel = 1.e-8;
   const Real tol_abs = 1.e-6;

   mlmg.setVerbose(1);
       
   // Solve linear system
   //phiV_alias.setVal(0.0); // initial guess for phi
   mlmg.solve({&phiV_alias}, {&chargeDistib}, tol_rel, tol_abs);

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
   gphi.define(this,1,NUM_GROW);
   gradPhiV = gphi.get();

   ef_calcGradPhiV(time, phiV_borders, gradPhiV);
   Efield_edge = {AMREX_D_DECL(gradPhiV[0], gradPhiV[1], gradPhiV[2])};
   average_face_to_cellcenter(Efield, 0, Efield_edge);
}

