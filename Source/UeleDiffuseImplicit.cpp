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
PeleC::UeleDiffuseImplicit ( Real time,
                         Real dt,
                         const amrex::MultiFab & Sbord,
                               amrex::MultiFab & forcing)
{
  BL_PROFILE("PeleC::UeleSolveImplicit()");

  Real prev_time = state[State_Type].prevTime();
  amrex::Real me_g = 9.10938356e-28;         // electron mass (g)

  // Create the linear operator
  LPInfo info;
  info.setAgglomeration(1);
  info.setConsolidation(1);
  info.setMetricTerm(false);
#ifdef PELEC_USE_EB
  const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
  MLEBABecLap     UeleOp({geom}, {grids}, {dmap}, info, {ebf});
#else
  MLABecLaplacian UeleOp({geom}, {grids}, {dmap}, info);
#endif

  amrex::Print() << "Solving for electron drift/diffusion \n";

  // Build a Uele with 1 GC properly filled from Sbord MF passed in
  MultiFab Sborder(grids, dmap, 1, 1, amrex::MFInfo(), Factory());
  amrex::MultiFab::Copy(Sborder,Sbord,UFX+5,0,1,1);
  amrex::MultiFab::Copy(Uele_state,Sbord,UFX+5,0,1,1);

  // Create MF aliases
  MultiFab Uele_borders(Sborder, amrex::make_alias, 0, 1);
 
  // RHS MF
  MultiFab Ueleresid(grids,dmap,1,0,MFInfo(),Factory());

#ifdef _OPENMP
#pragma omp parallel
#endif

  // Set up the RHS of the linear problem
  for (MFIter mfi(Ueleresid,true); mfi.isValid(); ++mfi)
  {   
      const Box& bx = mfi.tilebox();
      const auto& Uele_ar   = Uele_state.array(mfi);
      const auto& Ueleresid_ar = Ueleresid.array(mfi);
      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        Ueleresid_ar(i,j,k) = Uele_ar(i,j,k);
      }); 
  }

  // Set domain BCs
  std::array<LinOpBCType,AMREX_SPACEDIM> uele_lobc, uele_hibc;
  ef_set_neBC(uele_lobc,uele_hibc);
  UeleOp.setDomainBC(uele_lobc, uele_hibc);

  // Set BCs at coarse/fine boundary
  std::unique_ptr<MultiFab> Uele_crse;
  if (level > 0) {
     auto& crselev = getLevel(level-1);
     Uele_crse.reset(new MultiFab(crselev.boxArray(), crselev.DistributionMap(), 1, 0));
     MultiFab& Coarse_State = (time == prev_time) ? crselev.get_old_data(State_Type) : crselev.get_new_data(State_Type);   
     MultiFab::Copy(*Uele_crse, Coarse_State,UFX+5,0,1,0);
     Uele_crse->mult((1.0/me_g), 0, 1,0);
     UeleOp.setCoarseFineBC(Uele_crse.get(), crse_ratio[0]);
  }

  // Set level BCs
  UeleOp.setLevelBC(0, &Uele_borders);

  // NOTE: for now assumine zero gradient at all EB faces
  
  // Setup solver coefficient: general form is (A * alpha - B * div beta grad) phi = rhs   
  // For Uele system solve: A, alpha = 1     B = dt, beta = D_e

  // Set scalar coefficients
  UeleOp.setScalars(1.0, dt*(5.0/3.0));

  // Set alpha coefficient
  MultiFab acoef(grids, dmap, 1, 0, MFInfo(), Factory());
  acoef.setVal(1.0);
  UeleOp.setACoeffs(0, acoef);

  // Set beta coefficients
  std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(De_ec[0],De_ec[1],De_ec[2])};
  UeleOp.setBCoeffs(0, bcoeffs);

  UeleOp.setMaxOrder(2);

  MLMG mlmg(UeleOp);

  // relative and absolute tolerances for linear solve
  const Real tol_rel = ef_PoissonTol;
  const Real tol_abs = std::max(std::max(Ueleresid.norm0(),Uele_state.norm0()) * ef_PoissonTol * 1.0e1, 1.0e-20);

  mlmg.setVerbose(ef_PoissonVerbose);
  mlmg.setMaxIter(1000);
       
  // Solve linear system
  mlmg.solve({&Uele_state}, {&Ueleresid}, tol_rel, tol_abs);

  // Calculate the mass density forcing term (g/cm3-s)
  for (MFIter mfi(Ueleresid,true); mfi.isValid(); ++mfi)
  {   
      const Box& bx = mfi.tilebox();
      const auto& Uele_new = Uele_state.array(mfi);
      const auto& Uele_old = Sborder.array(mfi);
      const auto& Uele_frc = forcing.array(mfi);
      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        Uele_frc(i,j,k) = (Uele_new(i,j,k) - Uele_old(i,j,k)) / dt;
      }); 
  }
}
