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
PeleC::nESolveImplicit ( Real time,
                         Real dt,
                         const amrex::MultiFab & I_R_in,
                         const amrex::MultiFab & Sbord,
                               amrex::MultiFab & forcing_nE)
{
   BL_PROFILE("PeleC::nESolveImplicit()");

   Real prev_time = state[State_Type].prevTime();

  // Create the linear operator
  LPInfo info;
  info.setAgglomeration(1);
  info.setConsolidation(1);
  info.setMetricTerm(false);
#ifdef PELEC_USE_EB
  const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
  MLEBABecCecLap     nEOp({geom}, {grids}, {dmap}, info, {ebf});
#else
  MLABecCecLaplacian nEOp({geom}, {grids}, {dmap}, info);
#endif

  amrex::Print() << "Solving for electron drift/diffusion \n";

  // Build a nE with 1 GC properly filled from Sbord MF passed in
  MultiFab Sborder(grids, dmap, 1, 1, amrex::MFInfo(), Factory());
  amrex::MultiFab::Copy(Sborder,Sbord,UFX+1,0,1,1);
  amrex::MultiFab::Copy(nE_state,Sbord,UFX+1,0,1,2);
  amrex::MultiFab::Copy(nE_state_old,Sbord,UFX+1,0,1,2);

  // Create MF aliases
  MultiFab nE_borders(Sborder, amrex::make_alias, 0, 1);
 
  // Charge distribution MF
  MultiFab nEresid(grids,dmap,1,0,MFInfo(),Factory());

#ifdef _OPENMP
#pragma omp parallel
#endif

  // Set up the RHS of the linear problem
  for (MFIter mfi(nEresid,true); mfi.isValid(); ++mfi)
  {   
      const Box& bx = mfi.tilebox();
      const auto& nE_ar   = nE_state.array(mfi);
      const auto& I_R_ar   = I_R_in.array(mfi,NUM_SPECIES+2);
      const auto& nEresid_ar = nEresid.array(mfi);
      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        nEresid_ar(i,j,k) = nE_ar(i,j,k);
        if(do_react) nEresid_ar(i,j,k) += dt*I_R_ar(i,j,k);
      }); 
  }

  // Set domain BCs
  std::array<LinOpBCType,AMREX_SPACEDIM> ne_lobc, ne_hibc;
  ef_set_neBC(ne_lobc,ne_hibc);
  nEOp.setDomainBC(ne_lobc, ne_hibc);

  // Set BCs at coarse/fine boundary
  std::unique_ptr<MultiFab> nE_crse;
  if (level > 0) {
     auto& crselev = getLevel(level-1);
     nE_crse.reset(new MultiFab(crselev.boxArray(), crselev.DistributionMap(), 1, 0));
     MultiFab& Coarse_State = (time == prev_time) ? crselev.get_old_data(State_Type) : crselev.get_new_data(State_Type);   
     MultiFab::Copy(*nE_crse, Coarse_State,UFX+1,0,1,0);
     nEOp.setCoarseFineBC(nE_crse.get(), crse_ratio[0]);
  }

  // Set level BCs
  nEOp.setLevelBC(0, &nE_borders);

  amrex::Real omega = 1.0;
  nEOp.setRelaxation(omega);
  nEOp.setUWOrder(2);

  // NOTE: for now assumine zero gradient at all EB faces
  
  // Setup solver coefficient: general form is (A * alpha - B * div beta grad - C * eta) phi = rhs   
  // For nE system solve: A, alpha = 1     B = dt, beta = D_e    C = -dt, eta = mu_e E

  // Set scalar coefficients
  nEOp.setScalars(1.0, dt, -1.0*dt);

  // Set alpha coefficient
  MultiFab acoef(grids, dmap, 1, 0, MFInfo(), Factory());
  acoef.setVal(1.0);
  nEOp.setACoeffs(0, acoef);

  // Set beta coefficients
  std::array<const MultiFab*,AMREX_SPACEDIM> bcoeffs{AMREX_D_DECL(De_ec[0],De_ec[1],De_ec[2])};
  nEOp.setBCoeffs(0, bcoeffs);

  // Set eta coefficients
  Array<MultiFab,AMREX_SPACEDIM> ccoef;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    ccoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0, MFInfo(), Factory());
    for (MFIter mfi(ccoef[idim],true); mfi.isValid(); ++mfi)
    {
      const Box& bx = mfi.tilebox();
      auto const& Ke_ar = Ke_ec[idim]->const_array(mfi);
      const auto& E_ar   = Efield_edge[idim]->array(mfi);
      const auto& eta_ar = ccoef[idim].array(mfi);

      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        eta_ar(i,j,k) = Ke_ar(i,j,k) * E_ar(i,j,k);
      });
    }
  }
  nEOp.setCCoeffs(0, amrex::GetArrOfConstPtrs(ccoef));

  nEOp.setMaxOrder(2);

  MLMG mlmg(nEOp);
  // nE_state.setVal(0.0); // initial guess for phi

  // relative and absolute tolerances for linear solve
  const Real tol_rel = ef_PoissonTol;
  const Real tol_abs = std::max(std::max(nEresid.norm0(),nE_state.norm0()) * ef_PoissonTol, 1.0e-5);

  mlmg.setVerbose(ef_PoissonVerbose);
  mlmg.setMaxIter(1000);
       
  // Solve linear system
  mlmg.solve({&nE_state}, {&nEresid}, tol_rel, tol_abs);

  for (MFIter mfi(forcing_nE,TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
     const Box& bx = mfi.tilebox();
     const amrex::Box cbox = amrex::grow(bx, nl_state.nGrow()-1);
     auto const& old_nE   = nE_state_old.const_array(mfi);
     auto const& new_nE   = nE_state.array(mfi);
     auto const& I_R_nE   = I_R_in.const_array(mfi,NUM_SPECIES+2);
     auto const& force    = forcing_nE.array(mfi);
     Real dtinv           = 1.0 / dt;
     amrex::ParallelFor(bx, [old_nE, new_nE, I_R_nE, force, dtinv]
     AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        if(new_nE(i,j,k) < 0.0) new_nE(i,j,k) = 1.0e-30;
        force(i,j,k) = (new_nE(i,j,k) - old_nE(i,j,k)) * dtinv;
        if (do_react) force(i,j,k) -= I_R_nE(i,j,k);
     });

  }
}
