#ifndef Plasma_K_H_
#define Plasma_K_H_

#include <AMReX_REAL.H>
#include <AMReX_Slopes_K.H>

static constexpr amrex::Real small_vel = 1.e-10;

#ifdef PELEC_USE_PLASMA
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaE(int i, int j, int k, int Ecomp,
          amrex::Array4<      amrex::Real> const& Ks,
          amrex::Array4<const amrex::Real> const& EoN,
#ifdef PELEC_USE_TWO_TEMP
          amrex::Array4<const amrex::Real> const& Te,
#endif
          amrex::Array4<const amrex::Real> const& rhoY,
          const amrex::Real *mwt) noexcept
{
  // TODO : CALCULATE ELECTRON MOBILITY AS A FUNCTION OF E/N
  // ndeak note - calculates mobility in units of cm2-C/erg-s
   using namespace amrex::literals;

   Ks(i,j,k,Ecomp) = 0.4_rt;
   amrex::Real Na = 6.0221409e23;     // 1/mol 

   // TODO remove after testing against 1D code
   Ks(i,j,k,Ecomp) = -483.13744177895e-7; // CGS cm2/erg-s

   // Truncate value for low and high reduced electric fields
#ifdef PELEC_USE_TWO_TEMP
   amrex::Real TeeV = Te(i,j,k,0) / 11604.51812;
   if(TeeV < 2.5e-2) Ks(i,j,k,Ecomp) = 0.402E+26;    // units 1/m-V-s
   else if(TeeV > 18.0) Ks(i,j,k,Ecomp) = 0.5E+24;    // units 1/m-V-s
   else{
     // Perform JANEV fitting to calculate electron mobility based on E/N
     double Jfit_coefs[] = {-3.40823148867166,
                            -0.583328353480193,
                             0.370656206209368,
                            -0.00784238163450707,
                            -0.0773717373972128,
                            -6.68545439104852e-05,
                             0.00656784390919327,
                             0.000113198307893528,
                            -0.000156778529032339};

     double JfitA = 4.02e25;       // Dimensional coefficient [1/m-V-s]
     double Janev_sum = 0.0;      // Dimensionless JANEV sum
     // Calculate JANEV fit for mu*N
     for(int n=0; n<9; n++){
       Janev_sum += Jfit_coefs[n] * pow(log(TeeV), n);
     }
     Ks(i,j,k,Ecomp) = JfitA * exp(Janev_sum);    // 1/m-V-s
   }
#else
   if(EoN(i,j,k,0) < 1.0e-2) Ks(i,j,k,Ecomp) = 0.2580E+26;    // units 1/m-V-s
   else if(EoN(i,j,k,0) > 2.0e3) Ks(i,j,k,Ecomp) = 0.5156E+24;    // units 1/m-V-s
   else{
     // Perform JANEV fitting to calculate electron mobility based on E/N
     double Jfit_coefs[] = {-1.54145799020700,
                    -0.367441476385848,
                   -0.0294116708783720,
                   0.00389647757942970,
                   0.00160781442476554,
                 -5.18533083493594e-05,
                 -3.30064130960685e-05,
                  2.85900722096601e-07,
                  2.03708407167435e-07};

     double JfitA = 2.58300000000000e+25;       // Dimensional coefficient [1/m-V-s]
     double Janev_sum = 0.0;      // Dimensionless JANEV sum
     // Calculate JANEV fit for mu*N
     for(int n=0; n<9; n++){
       Janev_sum += Jfit_coefs[n] * pow(log(EoN(i,j,k,0)), n);
     }
     Ks(i,j,k,Ecomp) = JfitA * exp(Janev_sum);    // 1/m-V-s
   }
#endif
   Ks(i,j,k,Ecomp) *= 1.0e-9;           // Convert to C/cm-erg-s

   amrex::Real ndens = 0.0;
   for(int n=0; n<NUM_SPECIES; n++) ndens += rhoY(i,j,k,n) * (1.0/mwt[n]) * Na;
   Ks(i,j,k,Ecomp) *= (-1.0/ndens);    //  Convert to cm2-C/erg-s (negative sign to account for charge)
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getDiffE(int i, int j, int k, 
         int Ecomp, int use_NL,
         const amrex::Real &relax_factor,
         amrex::Array4<const amrex::Real> const& rhoY,
         amrex::Array4<      amrex::Real> const& rhoD,
         amrex::Array4<const amrex::Real> const& EoN,
#ifdef PELEC_USE_TWO_TEMP
         amrex::Array4<const amrex::Real> const& Te,
#endif
         const amrex::Real * mwt) noexcept
{
  // ndeak note - calculates rho*diffusivity in units of (cm2/s)*(g/cm3)
   using namespace amrex::literals;

   amrex::Real Na = 6.0221409e23;  // 1/mol

   amrex::Real rho = 0.0;        // g/cm3
   for (int n = 0; n < NUM_SPECIES; n++) {
      rho += rhoY(i,j,k,n);      // g/cm3
   }

#ifdef PELEC_USE_TWO_TEMP
   amrex::Real TeeV = Te(i,j,k,0) / 11604.51812;
   // Truncate value for low and high electron temperatures
   if(TeeV < 2.5e-2) rhoD(i,j,k,Ecomp) = 0.103E+25;     // 1/m-s
   else if(TeeV > 18.0) rhoD(i,j,k,Ecomp) = 0.942E+25;   // 1/m-s
   else{
     // Perform JANEV fitting to calculate electron mobility based on E/N
     double Jfit_coefs[] = {-1.57080706711061,
                             0.328206580751563,
                             0.118157576335441,
                             0.0418210131675193,
                            -0.0104639774039364,
                            -0.00948800714914011,
                            -0.000115379181169528,
                             0.000597315536837015,
                             6.84799993667713e-05};

     double JfitA = 9.42e+24;     // Dimensional JANEV coefficient [1/m-s]
     double Janev_sum = 0.0;                   // Dimensionless JANEV sum

     // Calculate JANEV fit for mu*N
     for(int n=0; n<9; n++){
       Janev_sum += Jfit_coefs[n] * pow(log(TeeV), n);
     }
     rhoD(i,j,k,Ecomp) = JfitA * exp(Janev_sum);     // [1/m-s]
   }
#else
   // Truncate value for low and high reduced electric fields
   if(EoN(i,j,k,0) < 1.0e-2) rhoD(i,j,k,Ecomp) = 0.1170E+25;     // 1/m-s
   else if(EoN(i,j,k,0) > 2.0e3) rhoD(i,j,k,Ecomp) = 0.1036E+26;   // 1/m-s
   else{
     // Perform JANEV fitting to calculate electron mobility based on E/N
     double Jfit_coefs[] = {-2.02172418780373,
                 0.0974868059493971,
                0.00706739883513016,
               -0.00362581265760427,
               7.20264662847402e-05,
               0.000192117046111647,
               1.01967072551187e-05,
              -2.03891006949100e-06,
              -1.58208775370330e-07};

     double JfitA = 1.036000000000000e+25;     // Dimensional JANEV coefficient [1/m-s]
     double Janev_sum = 0.0;                   // Dimensionless JANEV sum

     // Calculate JANEV fit for mu*N
     for(int n=0; n<9; n++){
       Janev_sum += Jfit_coefs[n] * pow(log(EoN(i,j,k,0)), n);
     }
     rhoD(i,j,k,Ecomp) = JfitA * exp(Janev_sum);     // [1/m-s]
   }
#endif
   rhoD(i,j,k,Ecomp) *= 1.0e-2;           // Convert to 1/cm-s

   amrex::Real ndens = 0.0;
   for(int n=0; n<NUM_SPECIES; n++) ndens += rhoY(i,j,k,n) * (1.0/mwt[n]) * Na;
   rhoD(i,j,k,Ecomp) *= (1.0/ndens) * relax_factor;       // Convert to (cm2/s)
   if(!use_NL) rhoD(i,j,k,Ecomp) *= rho;   // Convert to (g/cm-s) unless using coupled system
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaSp(int i, int j, int k,
           amrex::GpuArray<int, NUM_SPECIES> a_zk_num,
           amrex::Array4<      amrex::Real> const& Ksp) noexcept
{
   // TODO : CALCULATE MOBILITIES FROM DIFFUSIVITIES
   // ndeak note - returns mobility in units of cm2/erg-s
   // mwt in units of g/mol
   // a_zk in units of C/g
   // PP_RU_CGS in units of erg/mol-K
   // mobility already takes into account different drift direction of + and - ions due to a_zk
   using namespace amrex::literals;

   for (int n = 0; n < NUM_SPECIES; n++) {
      if(n != E_ID) Ksp(i,j,k,n) = 0.0;
   }

   // Using constant mobilities for now...
   for (int n = 0; n < NUM_SPECIES; n++) {
      if(n != E_ID && a_zk_num[n] != 0) Ksp(i,j,k,n) = 2.738979530e-7 * amrex::Math::copysign(1.0, a_zk_num[n]);
   }
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
cen2edg_cpp(int i, int j, int k,
            int dir,
            int ncomp,
            const bool                       do_harmonic,
            const bool                       on_lo,
            const bool                       on_hi,
            amrex::Array4<const amrex::Real> const& cfab,
            amrex::Array4<      amrex::Real> const& efab) noexcept
{
   using namespace amrex::literals;

   // Default behavior: id_l -1 in dir
   // Handle the BCs
   // Need -1 in id_l and id_h in dir on low Dirichlet BC
   // Need nothing on high Dirichlet BC
   int  id_l[3] = {i,j,k};
   int  id_h[3] = {i,j,k};
   if (!on_lo ) { 
      if ( !on_hi ) {
         id_l[dir] -= 1;
      }  
   } else {
      id_l[dir] -= 1;
      id_h[dir] -= 1;
   }  
   
   if (do_harmonic) {
      for (int n = 0; n < ncomp; n++) {
         if ( ( cfab(id_l[0],id_l[1],id_l[2],n) * cfab(id_h[0],id_h[1],id_h[2],n) ) > 0.0_rt ) {
            efab(i,j,k,n) = 2.0_rt * ( cfab(id_l[0],id_l[1],id_l[2],n) * cfab(id_h[0],id_h[1],id_h[2],n) )
                                   / ( cfab(id_l[0],id_l[1],id_l[2],n) + cfab(id_h[0],id_h[1],id_h[2],n) );
         } else {                  
            efab(i,j,k,n) = 0.0_rt;
         }  
      }  
   } else {
      for (int n = 0; n < ncomp; n++) {
         efab(i,j,k,n) = 0.5_rt*(cfab(id_l[0],id_l[1],id_l[2],n) + cfab(id_h[0],id_h[1],id_h[2],n));
      }  
   }  
}  

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ef_edge_state(int i,int j,int k,
              int dir,
              amrex::Array4<amrex::Real const> const& state,
              amrex::Array4<amrex::Real const> const& effvel ) noexcept
{
   using namespace amrex::literals;
   int idx_hi[3] = {i,j,k};
   int idx_lo[3] = {i,j,k};
   idx_lo[dir] -= 1;
   amrex::Real state_ed;
   if ( effvel(i,j,k) > small_vel) {
      state_ed = state(idx_lo[0],idx_lo[1],idx_lo[2]);
   } else if ( effvel(i,j,k) < -small_vel) {
      state_ed = state(idx_hi[0],idx_hi[1],idx_hi[2]);
   } else {
      state_ed = 0.5_rt * ( state(idx_lo[0],idx_lo[1],idx_lo[2]) +
                            state(idx_hi[0],idx_hi[1],idx_hi[2]) );
   }   
   return state_ed;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ef_edge_state_extdir(int i,int j,int k,
                     int dir,
                     const bool                       on_lo,
                     const bool                       on_hi,
                     amrex::Array4<amrex::Real const> const& state,
                     amrex::Array4<amrex::Real const> const& effvel ) noexcept
{
   using namespace amrex::literals;

   // Default behavior: id_l -1 in dir
   // Handle the BCs
   // Need -1 in id_l and id_h in dir on low Dirichlet BC
   // Need nothing on high Dirichlet BC
   int  id_l[3] = {i,j,k};
   int  id_h[3] = {i,j,k};
   if (!on_lo ) {
      if ( !on_hi ) {
         id_l[dir] -= 1;
      }
   } else {
      id_l[dir] -= 1;
      id_h[dir] -= 1;
   }

   amrex::Real state_ed;
   if ( effvel(i,j,k) > small_vel) {
      state_ed = state(id_l[0],id_l[1],id_l[2]);
   } else if ( effvel(i,j,k) < -small_vel) {
      state_ed = state(id_h[0],id_h[1],id_h[2]);
   } else {
      state_ed = 0.5_rt * ( state(id_l[0],id_l[1],id_l[2]) +
                            state(id_h[0],id_h[1],id_h[2]) );
   }
   return state_ed;
}

// AMREX_GPU_HOST_DEVICE
// AMREX_FORCE_INLINE

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ef_edge_state_2ndO_extdir(int i,int j,int k,
                          int dir,
                          const bool                       on_lo,
                          const bool                       on_hi,
                          amrex::Array4<amrex::Real const> const& state,
                          amrex::Array4<amrex::Real const> const& slopes,
                          amrex::Array4<amrex::Real const> const& effvel,
                          const amrex::Box& box ) noexcept
{
   using namespace amrex::literals;

   // Default behavior: id_l -1 in dir
   // Handle the BCs
   // Need -1 in id_l and id_h in dir on low Dirichlet BC
   // Need nothing on high Dirichlet BC
   int  id_l[3] = {i,j,k};
   int  id_h[3] = {i,j,k};
   if (!on_lo ) {
      if ( !on_hi ) {
         id_l[dir] -= 1;
      }
   } else {
      id_l[dir] -= 1;
      id_h[dir] -= 1;
   }

   int order = 2;

   amrex::Real state_ed;
   amrex::Real state_pls = state(id_h[0],id_h[1],id_h[2]);
   if ( dir == 0 ) {
      state_pls -= 0.5 * slopes(id_h[0],id_h[1],id_h[2], 0);
   } else if ( dir == 1 ) {
      state_pls -= 0.5 * slopes(id_h[0],id_h[1],id_h[2], 1);
#if (AMREX_SPACEDIM == 3)
   } else {
      state_pls -= 0.5 * slopes(id_h[0],id_h[1],id_h[2], 2);
#endif
   }
   amrex::Real state_mns = state(id_l[0],id_l[1],id_l[2]);
   if ( dir == 0 ) {
      state_mns += 0.5 * slopes(id_l[0],id_l[1],id_l[2],0);
   } else if ( dir == 1 ) {
      state_mns += 0.5 * slopes(id_l[0],id_l[1],id_l[2],1);
#if (AMREX_SPACEDIM == 3)
   } else {
      state_mns += 0.5 * slopes(id_l[0],id_l[1],id_l[2],2);
#endif
   }
   if ( effvel(i,j,k) > small_vel) {
      state_ed = state_mns;
   } else if ( effvel(i,j,k) < -small_vel) {
      state_ed = state_pls;
   } else {
      state_ed = 0.5_rt * ( state_pls + state_mns );
   }
   return state_ed;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
cen2edg_upwind_cpp(int i, int j, int k,
                   int dir,
                   int ncomp,
                   const bool                       on_lo,
                   const bool                       on_hi,
                   amrex::Array4<const amrex::Real> const& ueff,
                   amrex::Array4<const amrex::Real> const& cfab,
                   amrex::Array4<      amrex::Real> const& efab) noexcept
{
   using namespace amrex::literals;

   // Default behavior: id_l -1 in dir
   // Handle the BCs
   // Need -1 in id_l and id_h in dir on low Dirichlet BC
   // Need nothing on high Dirichlet BC
   int  id_l[3] = {i,j,k};
   int  id_h[3] = {i,j,k};
   if (!on_lo ) {
      if ( !on_hi ) {
         id_l[dir] -= 1;
      }
   } else {
      id_l[dir] -= 1;
      id_h[dir] -= 1;
   }

   for (int n = 0; n < ncomp; n++) {
      if ( ueff(i,j,k) > small_vel ) {
         efab(i,j,k,n) = cfab(id_l[0],id_l[1],id_l[2],n);
      } else if ( ueff(i,j,k) < -small_vel ) {
         efab(i,j,k,n) = cfab(id_h[0],id_h[1],id_h[2],n);
      } else {
         efab(i,j,k,n) = 0.5_rt*(cfab(id_l[0],id_l[1],id_l[2],n) + cfab(id_h[0],id_h[1],id_h[2],n));
      }
   }
}

#endif
#endif
