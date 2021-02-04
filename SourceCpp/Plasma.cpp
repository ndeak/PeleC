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
   amrex::Real eps0 = 8.854187817e-12;                //Free space permittivity (C/(V.m))
   amrex::Real eps0_cgs = 8.854187817e-12 * 1.0e-9;   //Free space permittivity (C/(erg.cm))
   amrex::Real epsr = 1.0;
   amrex::Real elemCharge = 1.60217662e-19;     //Coulomb per charge
   amrex::Real Na = 6.022e23;                   //Avogadro's number
   amrex::Real PP_RU_MKS = 8.31446261815324;    //Universal gas constant (J/mol-K)
   amrex::Real PP_RU_CGS = 83144626.1815324;    // (erg/mol-K)
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

    pp.query("JFNK_newtonTol",ef_newtonTol);
    pp.query("JFNK_maxNewton",ef_maxNewtonIter);
    pp.query("JFNK_difftype",ef_diffT_jfnk);
    pp.query("JFNK_lambda",ef_lambda_jfnk);
    pp.query("GMRES_restart_size",ef_GMRES_size);
    pp.query("GMRES_rel_tol",ef_GMRES_reltol);
    pp.query("GMRES_max_restart",ef_GMRES_maxRst);
    pp.query("GMRES_verbose",ef_GMRES_verbose);

    pp.query("Precond_MG_tol",ef_PC_MG_Tol);
    pp.query("Precond_fixedIter",ef_PC_fixedIter);
    pp.query("Precond_SchurApprox",ef_PC_approx);

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

    // get charge per unit mass (C/g) CGS
    Real zk_temp[NUM_SPECIES];
    EOS::charge_mass(zk_temp);
    for (int k = 0; k < NUM_SPECIES; k++) {
       zk[k] = zk_temp[k];
    }
}

void PeleC::plasma_define_data() {
   if (ef_use_NLsolve) {
      nl_state.define(grids,dmap,2,2);
      nl_resid.define(grids,dmap,2,2);
      bg_charge.define(grids,dmap,1,1);
      ef_state_old.define(grids,dmap,2,2);

      if (elec_Ueff != 0) delete [] elec_Ueff;

      elec_Ueff = new MultiFab[AMREX_SPACEDIM];
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
         const BoxArray& edgeba = getEdgeBoxArray(d);
         elec_Ueff[d].define(edgeba, dmap, 1, 1,MFInfo(),Factory());
      }

      // Transport coefficients
      diff_e.define(this);
      De_ec = diff_e.get();
      mob_e.define(this);
      Ke_ec = mob_e.get();
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
   MLPoisson poisson({geom}, {grids}, {dmap}, info);
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

  // BL_ASSERT(whichTime == AmrOldTime || whichTime == AmrNewTime);

  // MultiFab& S     = (whichTime == AmrOldTime) ? get_old_data(State_Type) : get_new_data(State_Type);
  // MultiFab& diff  = (whichTime == AmrOldTime) ? (*diffn_cc) : (*diffnp1_cc);
  // MultiFab& Kspec = (whichTime == AmrOldTime) ? KSpec_old : KSpec_new;

  // Get the cc transport coeffs. These are temporary.
  MultiFab Ke_cc(grids,dmap,1,S.nGrow());
  MultiFab De_cc(grids,dmap,1,S.nGrow());

  // Fillpatch the state 
  // ndeak note - not needed since S with boundary cells is provided as input
  // FillPatchIterator fpi(*this,S,Ke_cc.nGrow(),time,State_Type,UFS,NUM_SPECIES+3);
  // MultiFab& S_cc = fpi.get_mf();

  // ndeak add - get BCs for species (used in center->edge extrap)
  const amrex::BCRec& bcspec = get_desc_lst()[State_Type].getBC(UFS);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
     const amrex::Box& gbox = mfi.growntilebox();
     auto const& rhoY = S.array(mfi,UFS);
     auto const& T    = Q_ext.array(mfi,QTEMP);
     auto const& rhoD = coeffs_old.array(mfi,dComp_rhoD);
     auto const& Ke   = Ke_cc.array(mfi);
     auto const& De   = De_cc.array(mfi);
     auto const& Ks   = KSpec_old.array(mfi);
     Real factor = EFConst::PP_RU_CGS / ( EFConst::Na * EFConst::elemCharge );
     int useNL   = ef_use_NLsolve;
     amrex::ParallelFor(gbox, [rhoY, T, factor, Ks, rhoD, Ke, De, useNL]
     AMREX_GPU_DEVICE (int i, int j, int k) noexcept
     {
        if (useNL) {
           getKappaE(i,j,k,0,Ke);
           getDiffE(i,j,k,0,useNL,factor,T,rhoY,Ke,De);
        } else {
           getKappaE(i,j,k,E_ID,Ks);
           getDiffE(i,j,k,E_ID,useNL,factor,T,rhoY,Ks,rhoD);
        }
     });
     Real mwt[NUM_SPECIES];
     EOS::molecular_weight(mwt);  // Return mwt in CGS
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

  if ( ef_use_NLsolve ) {
     // CC -> EC transport coeffs. These are PeleC class object used in the non-linear residual.
     // ndeak TODO: check to make sure we are checking all the necessary BCTypes for on_lo/hi
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
