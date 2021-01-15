#include <PeleC.H>
#include <AMReX_MLABecLaplacian.H>
#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#endif
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <SolveEfield.H>
#include "prob.H"

#include <cmath>

using namespace amrex;

namespace EFConst{
    AMREX_GPU_DEVICE_MANAGED amrex::Real eps0 = 8.854187817e-12;          //Free space permittivity (C/(V.m))
    AMREX_GPU_DEVICE_MANAGED amrex::Real epsr = 1.0;
    AMREX_GPU_DEVICE_MANAGED amrex::Real elemCharge = 1.60217662e-19;     //Coulomb per charge
    AMREX_GPU_DEVICE_MANAGED amrex::Real Na = 6.022e23;                   //Avogadro's number
    AMREX_GPU_DEVICE_MANAGED amrex::Real PP_RU_MKS = 8.31446261815324;    //Universal gas constant (J/mol-K)
}

using std::string;

struct PhiVFill
{
    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real time,
                     const BCRec* bcr, const int bcomp,
                     const int orig_comp) const
    {
       const int* domlo = geom.Domain().loVect();
       const int* domhi = geom.Domain().hiVect();
       const amrex::Real* dx = geom.CellSize();
       const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
         domlo[0] + (iv[0] + 0.5) * dx[0], domlo[1] + (iv[1] + 0.5) * dx[1],
         domlo[2] + (iv[2] + 0.5) * dx[2])};
       const int* bc = bcr->data();

       amrex::Real s_int[NVAR] = {0.0};
       amrex::Real s_ext[NVAR] = {0.0};

       for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
          if ((bc[idir] == amrex::BCType::ext_dir) and (iv[idir] < domlo[idir])) {
             bcnormal(x, s_int, s_ext, idir, +1, time, geom);
             dest(iv, dcomp) = s_ext[UFX];
          }
          if ((bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) and (iv[idir] > domhi[idir])) {
             bcnormal(x, s_int, s_ext, idir, -1, time, geom);
             dest(iv, dcomp) = s_ext[UFX];
          }
       }
    }
};

void 
PeleC::ef_init() 
{
    amrex::Print() << " Init EFIELD solve options \n";

    // Params defaults
    ef_verbose = 0;
    ef_debug = 0;
    def_harm_avg_cen2edge  = false;

    // User input parser to query efield inputs
    amrex::ParmParse pp("ef");
    pp.query("verbose",ef_verbose);
    pp.query("debug",ef_debug);
    pp.query("def_harm_avg_cen2edge",def_harm_avg_cen2edge);

    // ndeak add - hard-coding charges for now
    // zk[0] = -1.0;
    // zk[1] =  0.0;
    // zk[2] =  0.0;
    // zk[3] =  0.0;
    // zk[4] =  1.0;
    // zk[5] =  1.0;
    // zk[6] =  1.0;
    // zk[7] =  1.0;
    // zk[8] =  1.0;
    // zk[9] = -1.0;

    // get charge per unit  mass (C/kg)
    Real zk_temp[NUM_SPECIES];
    EOS::charge_mass(zk_temp);
    for (int k = 0; k < NUM_SPECIES; k++) {
       zk[k] = zk_temp[k];
    }
}

void PeleC::ef_define_data() {
   BL_PROFILE("PELC_EF::ef_define_data()");

   // De_ec = ;
   // Ke_ec = ;

}

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
   // ndeak add
   printf("finished MLMH setup!\n");

   // relative and absolute tolerances for linear solve
   const Real tol_rel = 1.e-8;
   const Real tol_abs = 1.e-6;

   mlmg.setVerbose(1);
       
   // Solve linear system
   //phiV_alias.setVal(0.0); // initial guess for phi
   mlmg.solve({&phiV_alias}, {&chargeDistib}, tol_rel, tol_abs);
   // ndeak add
   printf("finished MLMH solve!\n");

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

void PeleC::ef_calc_transport(const amrex::MultiFab& S, const amrex::Real &time) {
  BL_PROFILE("PeleC::ef_calc_transport()");
 
  // ndeak note - since only MOL is being used for now, it is assumed all data MFs are at time t=n

  if ( ef_verbose ) amrex::Print() << " Compute EF transport prop.\n";

  const TimeLevel whichTime = which_time(State_Type, time);

  // BL_ASSERT(whichTime == AmrOldTime || whichTime == AmrNewTime);

  // MultiFab& S     = (whichTime == AmrOldTime) ? get_old_data(State_Type) : get_new_data(State_Type);
  // MultiFab& diff  = (whichTime == AmrOldTime) ? (*diffn_cc) : (*diffnp1_cc);
  // MultiFab& Kspec = (whichTime == AmrOldTime) ? KSpec_old : KSpec_new;

  // Get the cc transport coeffs. These are temporary.
  // MultiFab Ke_cc(grids,dmap,1,1);
  // MultiFab De_cc(grids,dmap,1,1);

  // Fillpatch the state 
  // ndeak note - not needed since S with boundary cells is provided as input
  // FillPatchIterator fpi(*this,S,Ke_cc.nGrow(),time,State_Type,UFS,NUM_SPECIES+3);
  // MultiFab& S_cc = fpi.get_mf();

  // ndeak add - get BCs for species (used in center->edge extrap)
  const amrex::BCRec& bcspec = get_desc_lst()[State_Type].getBC(UFS);
  const int* bcrec = bcspec.data();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
     const amrex::Box& tbox = mfi.tilebox();
     int ng = S.nGrow();
     const amrex::Box gbox = amrex::grow(tbox, ng);
     auto const& rhoY = S.array(mfi,UFS);
     auto const& T    = Q_ext.array(mfi,QTEMP);
     auto const& rhoD = coeffs_old.array(mfi,dComp_rhoD);
     // auto const& Ke   = Ke_cc.array(mfi);
     // auto const& De   = De_cc.array(mfi);
     auto const& Ks   = KSpec_old.array(mfi);
     Real factor = EFConst::PP_RU_MKS / ( EFConst::Na * EFConst::elemCharge );
     amrex::ParallelFor(gbox, [rhoY, T, factor, Ks, rhoD]
     AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        getKappaE(i,j,k,E_ID,Ks);
        getDiffE(i,j,k,E_ID,factor,T,rhoY,Ks,rhoD);
     });
     Real mwt[NUM_SPECIES];
     EOS::molecular_weight(mwt);
     amrex::ParallelFor(gbox, [rhoY, rhoD, T, Ks, mwt]
     AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        getKappaSp(i,j,k, mwt, zk, rhoY, rhoD, T, Ks);
     });
  }
  if ( ef_debug ) {
     std::string timetag = (whichTime == AmrOldTime) ? "old" : "new";
     VisMF::Write(KSpec_old,"KappaSpec"+timetag+"_Lvl"+std::to_string(level));
  }

//   // CC -> EC transport coeffs. These are PeleC class object used in the non-linear residual.
//   // ndeak TODO: check to make sure we are checking all the necessary BCTypes for on_lo/hi
//   const Box& domain = geom.Domain();
//   bool use_harmonic_avg = def_harm_avg_cen2edge ? true : false;
// #ifdef _OPENMP
// #pragma omp parallel if (Gpu::notInLaunchRegion())
// #endif
//   for (MFIter mfi(De_cc,TilingIfNotGPU()); mfi.isValid();++mfi)
//   {
//      for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
//      {
//         const Box ebx = mfi.nodaltilebox(dir);
//         const Box& edomain = amrex::surroundingNodes(domain,dir);
//         const auto& diff_c  = De_cc.array(mfi);
//         const auto& diff_ed = De_ec[dir]->array(mfi);
//         const auto& kappa_c  = Ke_cc.array(mfi);
//         const auto& kappa_ed = Ke_ec[dir]->array(mfi);
//         amrex::ParallelFor(ebx, [dir, bc_lo, bc_hi, use_harmonic_avg, diff_c, diff_ed,
//                                  kappa_c, kappa_ed, edomain]
//         AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//         {
//            int idx[3] = {i,j,k};
//            bool on_lo = ( ( bcrec[dir] == amrex::BCType::ext_dir ) && k <= domain.smallEnd(dir) );
//            bool on_hi = ( ( bcrec[AMREX_SPACEDIM+dir] == amrex::BCType::ext_dir ) && k >= domain.bigEnd(dir) );
//            cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, diff_c, diff_ed);
//            cen2edg_cpp( i, j, k, dir, 1, use_harmonic_avg, on_lo, on_hi, kappa_c, kappa_ed);
//         });
//      }
//   }
}

