#ifndef Plasma_K_H_
#define Plasma_K_H_

#include <AMReX_REAL.H>
#include <AMReX_Slopes_K.H>

static constexpr amrex::Real small_vel = 1.e-10;

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaE(int i, int j, int k, int Ecomp,
          amrex::Array4<      amrex::Real> const& Ks) noexcept
{
  // ndeak note - calculates mobility in units of cm2/erg-s
   using namespace amrex::literals;

   Ks(i,j,k,Ecomp) = 0.4_rt;
  
   // TODO remove after testing against 1D code
   Ks(i,j,k,Ecomp) = -483.13744177895e-7; // CGS cm2/erg-s
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getDiffE(int i, int j, int k, int Ecomp, 
         int use_NL,
         const amrex::Real &factor,
         amrex::Array4<const amrex::Real> const& T,
         amrex::Array4<const amrex::Real> const& rhoY,
         amrex::Array4<const amrex::Real> const& Ks,
         amrex::Array4<      amrex::Real> const& rhoD) noexcept
{
  // ndeak note - calculates diffusivity in units of (cm2/s)*(g/cm3)
  // factor has units of J/K-C
   using namespace amrex::literals;

   if ( use_NL ) {  // If using non-linear solve, don't * rho
      rhoD(i,j,k,Ecomp) = Ks(i,j,k,Ecomp) * T(i,j,k) * factor;
      // TODO remove after testing 1D code
      rhoD(i,j,k,Ecomp) = 846.28133089;
   } else {
      amrex::Real rho = 0.0;
      for (int n = 0; n < NUM_SPECIES; n++) {
         rho += rhoY(i,j,k,n);
      }
      rhoD(i,j,k,Ecomp) = rho * Ks(i,j,k,Ecomp) * T(i,j,k) * factor;

      // TODO remove after testing 1D code
      rhoD(i,j,k,Ecomp) = rho * 846.28133089;
   }

}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaSp(int i, int j, int k,
           amrex::Real const mwt[NUM_SPECIES],
           amrex::GpuArray<amrex::Real, NUM_SPECIES> a_zk,
           amrex::Array4<const amrex::Real> const& rhoY,
           amrex::Array4<      amrex::Real> const& rhoD,
           amrex::Array4<const amrex::Real> const& T,
           amrex::Array4<      amrex::Real> const& Ksp) noexcept
{
   // ndeak note - returns mobility in units of cm2/erg-s
   // mwt in units of g/mol
   // a_zk in units of C/g
   // PP_RU_CGS in units of erg/mol-K
   // mobility already takes into account different drift direction of + and - ions due to a_zk
   using namespace amrex::literals;
   amrex::Real rho = 0.0;
   amrex::Real PP_RU_CGS = 83144626.1815324;
   for (int n = 0; n < NUM_SPECIES; n++) {
      rho += rhoY(i,j,k,n);
   }
   for (int n = 0; n < NUM_SPECIES; n++) {
   //   if(n != E_ID) Ksp(i,j,k,n) = rhoD(i,j,k,n) * mwt[n] * 0.001 * a_zk[n] / ( rho * PP_RU_CGS * T(i,j,k) );
   //   // if ( n == E_ID ) Ksp(i,j,k,n) = 0.0;         // Set the E species mobility to zero. Electron are treated separately.
   //   // if ( n == 0 ) Ksp(i,j,k,n) = 0.0;         // Set the E species mobility to zero. Electron are treated separately.
       if(n != E_ID) Ksp(i,j,k,n) = 0.0;
   }

   // TODO remove after testing 1D code
   // for (int n = 0; n < NUM_SPECIES; n++) {
   //    if(n != E_ID) Ksp(i,j,k,n) = 2.738979530 * amrex::Math::copysign(1.0, a_zk[n]);
   // }
   Ksp(i, j, k, 4) = 2.738979530e-7;
   Ksp(i, j, k, 5) = 2.738979530e-7;
   Ksp(i, j, k, 6) = 2.738979530e-7;
   Ksp(i, j, k, 7) = 2.738979530e-7;
   Ksp(i, j, k, 8) = 2.738979530e-7;
   Ksp(i, j, k, 9) = -2.738979530e-7;


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

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ef_edge_state_2ndO_extdir(int i,int j,int k,
                          int dir,
                          const bool                       on_lo,
                          const bool                       on_hi,
                          const bool             extdir_or_ho_lo,
                          const bool             extdir_or_ho_hi,
                          int                              domlo,
                          int                              domhi,
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

   int order = 2;

   amrex::Real state_ed;
   amrex::Real state_pls = state(id_h[0],id_h[1],id_h[2]);
   if ( dir == 0 ) {
      state_pls -= 0.5 * amrex_calc_xslope_extdir(id_h[0],id_h[1],id_h[2],0,order,state,extdir_or_ho_lo,extdir_or_ho_hi,domlo,domhi);
   } else if ( dir == 1 ) {
      state_pls -= 0.5 * amrex_calc_yslope_extdir(id_h[0],id_h[1],id_h[2],0,order,state,extdir_or_ho_lo,extdir_or_ho_hi,domlo,domhi);
#if (AMREX_SPACEDIM == 3)
   } else {
      state_pls -= 0.5 * amrex_calc_zslope_extdir(id_h[0],id_h[1],id_h[2],0,order,state,extdir_or_ho_lo,extdir_or_ho_hi,domlo,domhi);
#endif
   }
   amrex::Real state_mns = state(id_l[0],id_l[1],id_l[2]);
   if ( dir == 0 ) {
      state_pls += 0.5 * amrex_calc_xslope_extdir(id_l[0],id_l[1],id_l[2],0,order,state,extdir_or_ho_lo,extdir_or_ho_hi,domlo,domhi);
   } else if ( dir == 1 ) {
      state_pls += 0.5 * amrex_calc_yslope_extdir(id_l[0],id_l[1],id_l[2],0,order,state,extdir_or_ho_lo,extdir_or_ho_hi,domlo,domhi);
#if (AMREX_SPACEDIM == 3)
   } else {
      state_pls += 0.5 * amrex_calc_zslope_extdir(id_l[0],id_l[1],id_l[2],0,order,state,extdir_or_ho_lo,extdir_or_ho_hi,domlo,domhi);
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
