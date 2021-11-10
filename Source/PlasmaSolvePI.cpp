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
PeleC::solvePI ( Real time,
                 Real dt,
                 ProbParmDevice const& prob_parm )
{
    BL_PROFILE("PeleC::solvePI()");

    amrex::Print() << "Solving for photoionization source \n";

    // Get current PhiV
    Real prev_time = state[State_Type].prevTime();
    MultiFab& Ucurr = (time == prev_time) ? get_old_data(State_Type) : get_new_data(State_Type);

    // Set up helmholtz operator
    LPInfo info;
    info.setAgglomeration(1);
    info.setConsolidation(1);
    info.setMetricTerm(false);

#ifdef AMREX_USE_EB
    const auto& ebf = &dynamic_cast<EBFArrayBoxFactory const&>((parent->getLevel(level)).Factory());
    MLEBABecLap helmholtzOP({geom}, {grids}, {dmap}, info, {ebf});
#else
    MLABecLaplacian helmholtzOP({geom}, {grids}, {dmap}, info);
#endif

    helmholtzOP.setMaxOrder(2);

    // Setup solver coefficient: general form is (ascal * acoef - bscal * div bcoef grad ) phi = rhs   
    // For simple Helmholtz solve: acoef = 1 and bcoef = 1
    MultiFab acoef(grids, dmap, 1, 0, MFInfo(), Factory());
    acoef.setVal(1.0);
    helmholtzOP.setACoeffs(0, acoef);
    Array<MultiFab,AMREX_SPACEDIM> bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      bcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0, MFInfo(), Factory());
      bcoef[idim].setVal(1.0);
    }
    helmholtzOP.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));   

    // PI MF with no border cells
    MultiFab PI_comp(grids, dmap, 1, 0, amrex::MFInfo(), Factory());

    // Calculate the ionization rate for all interior points (used in RHS)
    MultiFab ionRate(grids,dmap,1,0,MFInfo(),Factory());
    MultiFab helmholtzRHS(grids,dmap,1,0,MFInfo(),Factory());

    // Zero Dirichlet conditions used for all PI components at EB
#ifdef AMREX_USE_EB
    MultiFab PI_BC(grids, dmap, 1, 0, MFInfo(), Factory());
    PI_BC.setVal(0.0);
    MultiFab beta(grids, dmap, 1, 0, MFInfo(), Factory());
    beta.setVal(1.0);
#endif

    // Quenching pressure factor, see Pancheshnyi "Photoionization produced by low-current discharges in O2, air, N2 and CO2" (2015)
    // FIXME fix hard-coded to assume atmospheric pressure 
    amrex::Real pfact = 30.0 / (760 + 30.0);

    // Improved fit coefficients
    double N2fit_coefs[] = {-26685061586.6066, 1307959249.29072, -24087664.8330414, 208547.477008891, -1250.43557426934, -12.8703331099001};
    double O2fit_coefs[] = {-28753255434.8899, 1411237102.41166, -26070786.1973826, 227356.558100037, -1235.08040930513, -12.8088319148813}; 

    // Efficiency factor, see Breden "A numerical study of high-pressure non-equilibrium streamers for combustion ignition application" (2013)
    // Also see Luque "Photoionization in negative streamers: Fast computations and two propagation modes" (2007)
    // Quantity has a small dependence on E/N but can be considered approximately constant
    amrex::Real efact = 0.02;

    amrex::Real mwt[NUM_SPECIES];
    auto eos = pele::physics::PhysicsType::eos();
    eos.molecular_weight(mwt);   // CGS
    for (MFIter mfi(ionRate,true); mfi.isValid(); ++mfi)
    {   
        const Box& bx = mfi.tilebox();
        const auto& rhoY_ar = Ucurr.array(mfi,UFS);
        const auto& ion_ar = ionRate.array(mfi);
        const auto& eon_ar = redEfield.array(mfi);
        int useNL = ef_use_NLsolve;
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          // Recalculate forward rates for E + O2/N2 -> E + E + O2/N2
          // Improved ionization fits
          amrex::Real frateN2 = 0.0; 
          amrex::Real frateO2 = 0.0; 

          // Extrapolate to get ionization rate
          // amrex::Real logRateN2;
          // amrex::Real logRateO2;
          // ExtrapIonN2(eon_ar(i,j,k), &logRateN2);
          // ExtrapIonO2(eon_ar(i,j,k), &logRateO2);

          // frateN2 *= pow(10, logRateN2) * 1.0e6;
          // frateO2 *= pow(10, logRateO2) * 1.0e6;

          // amrex::Real tempsum = 0.0;
          // if(eon_ar(i,j,k) > 1.0e-10){
          //   for(int idx = 0; idx<6; idx++) tempsum += N2fit_coefs[idx] * pow(eon_ar(i,j,k), -5+idx);
          //   frateN2 = pow(10, tempsum) * 1.0e6;  // Converting m3/s -> cm3/s
          // }
  
          // tempsum = 0.0;
          // if(eon_ar(i,j,k) > 1.0e-10){
          //   for(int idx = 0; idx<6; idx++) tempsum += O2fit_coefs[idx] * pow(eon_ar(i,j,k), -5+idx);
          //   frateO2 = pow(10, tempsum) * 1.0e6; // Converting m3/s -> cm3/s
          // }          

          // Old fit data, only valid at low E/N (<300 Td)
          if(eon_ar(i,j,k) > 1.0e-10){
            // frateN2 = pow(10, -8.3 - 365.0/eon_ar(i,j,k)); 
            // frateO2 = pow(10, -8.8 - 281.0/eon_ar(i,j,k)); 
            frateN2 = pow(10, -7.6 - 600.0/eon_ar(i,j,k)); 
            frateO2 = pow(10, -8.0 - 400.0/eon_ar(i,j,k)); 
            // frateN2 = pow(10, -7.2 - 700.0/eon_ar(i,j,k)); 
            // frateO2 = pow(10, -7.4 - 550.0/eon_ar(i,j,k)); 
          }
    
          // Convert mass to number density
          amrex::Real nEl = rhoY_ar(i,j,k,0) / EFConst::me_cgs;
          amrex::Real nO2 = rhoY_ar(i,j,k,1) * EFConst::Na / mwt[1];
          amrex::Real nN2 = rhoY_ar(i,j,k,2) * EFConst::Na / mwt[2];

          // Calculate PI emission rate [1/cm3-s]
          if(nEl > 0.0){
            ion_ar(i,j,k) = efact * pfact * nEl * (frateO2*nO2 + frateN2*nN2);
          }
          else{
            ion_ar(i,j,k) = 0.0;
          }
          // printf("ion (%i %i %i) = %.6e, eon = %.6e, nEl = %.6e, nO2 = %.6e, nN2 = %.6e, frateO2 = %.6e, frateN2 = %.6e\n", i, j, k, ion_ar(i,j,k), eon_ar(i,j,k),  nEl, nO2, nN2, frateO2, frateN2);
        }); 
    }


    // See Bourdon "Efficient models for photoionization produced by non-thermal gas discharges in air based on radiative transfer and the Helmholtz equations" (2007)
    // Using 3 exponential model
    // (-(lambda_j P_O2)^2 + nabla^2)S_ph^j = -A_j P_O2^2 I_r
    // FIXME Hard-coding O2 partial pressure for now...
    amrex::Real P_O2 = 1012350 * 0.21;
    amrex::Real PI_lambda[3] = {4.14785e-5, 1.095e-4, 6.6756e-4};   // [cm-1 Ba-1]
    amrex::Real PI_A[3] = {1.1173e-10, 2.869e-9, 2.7488e-7};        // [cm-2 Ba-2]
    
    // amrex::Real PI_lambda[3] = {3.35278e-5, 8.4082e-5, 4.49588e-4};   // [cm-1 Ba-1]
    // amrex::Real PI_A[3] = {5.025427e-6, 2.59522e-5, 2.294445e-4};        // [cm-1 Ba-1]
    
    // 2 exponential fit
    // amrex::Real PI_lambda[3] = {7.3056e-5, 4.408e-4, 0.0};   // [cm-1 Ba-1]
    // amrex::Real PI_A[3] = {1.18145e-9, 9.986e-8, 0.0};        // [cm-2 Ba-2]
    
    for(int n = 0; n<3; n++){

      // Copy current values from PI_source as initial guess     
      // Set up RHS as well
      for (MFIter mfi(PI_comp,true); mfi.isValid(); ++mfi)
      {   
          const Box& bx = mfi.tilebox();
          const auto& pi_ar = PI_source.array(mfi,1);
          const auto& pic_ar = PI_comp.array(mfi);
          const auto& ion_ar = ionRate.array(mfi);
          const auto& rhs_ar = helmholtzRHS.array(mfi);
          amrex::ParallelFor(bx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            pic_ar(i,j,k) = pi_ar(i,j,k,n);
            rhs_ar(i,j,k) = -ion_ar(i,j,k) * PI_A[n] * pow(P_O2,2);
          }); 
      }
      
      // Set A and B scalars
      Real ascal = -pow(PI_lambda[n]*P_O2,2);
      Real bscal = -1.0;
      helmholtzOP.setScalars(ascal, bscal);

      // Boundary conditions for the linear operator.
      std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
      std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
      setBCPI(bc_lo,bc_hi);
      helmholtzOP.setDomainBC(bc_lo,bc_hi);   

      // Get the coarse level data for AMR cases.
      std::unique_ptr<MultiFab> PI_crse;
      if (level > 0) {
        auto& crselev = getLevel(level-1);
        PI_crse.reset(new MultiFab(crselev.boxArray(), crselev.DistributionMap(), 1, 0));
        MultiFab& Coarse_State = crselev.get_PI_data();   
        MultiFab::Copy(*PI_crse, Coarse_State,n+1,0,1,0);
        helmholtzOP.setCoarseFineBC(PI_crse.get(), crse_ratio[0]);
      }
      
      // Pass the PI borders MF with zero in all ghost cells for now.
      MultiFab PI_borders(PI_source, amrex::make_alias, n+1, 1);
      helmholtzOP.setLevelBC(0, &PI_borders);

      // Fill in zero Dirichlet BCs at electrodes for now..
#ifdef PELEC_USE_EB
      helmholtzOP.setEBDirichlet(0,PI_BC,beta);
#endif
      /////////////////////////////////////   
      // Setup a MG solver
      /////////////////////////////////////   
      MLMG mlmg(helmholtzOP);

      // relative and absolute tolerances for linear solve
      const Real tol_rel = ef_PoissonTol;
      const Real tol_abs = std::max(std::max(helmholtzRHS.norm0(),PI_comp.norm0()) * ef_PoissonTol, 1.0e-15);
  
      // Set solver verbosity
      mlmg.setVerbose(ef_PoissonVerbose);
       
      // Solve linear system
      mlmg.solve({&PI_comp}, {&helmholtzRHS}, 1.0e1*tol_rel, tol_abs);

      // Copy solution back into PI_sources
      for (MFIter mfi(PI_comp,true); mfi.isValid(); ++mfi)
      {
          const Box& bx = mfi.tilebox();
          const auto& pi_ar = PI_source.array(mfi);
          const auto& pic_ar = PI_comp.array(mfi);
          amrex::ParallelFor(bx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            pi_ar(i, j, k, n+1) = pic_ar(i, j, k);
          });
      }

    }

    // Calculate the PI source term from the three components
    for (MFIter mfi(PI_source,true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& pi_ar = PI_source.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          pi_ar(i, j, k, 0) = pi_ar(i, j, k, 1) + pi_ar(i, j, k, 2) + pi_ar(i, j, k, 3);
        });
    }
}

