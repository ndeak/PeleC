#ifndef Plasma_K_H_
#define Plasma_K_H_

#include <AMReX_REAL.H>

static constexpr amrex::Real small_vel = 1.e-10;

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaE(int i, int j, int k, int Ecomp,
          amrex::Array4<      amrex::Real> const& Ks) noexcept
{
  // ndeak note - calculates mobility in units of cm2/V-s
   using namespace amrex::literals;

   Ks(i,j,k,Ecomp) = 0.4_rt;
  
   // TODO remove after testing against 1D code
   Ks(i,j,k,Ecomp) = -483.13744177895;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getDiffE(int i, int j, int k, int Ecomp, 
         const amrex::Real &factor,
         amrex::Array4<const amrex::Real> const& T,
         amrex::Array4<const amrex::Real> const& rhoY,
         amrex::Array4<const amrex::Real> const& Ks,
         amrex::Array4<      amrex::Real> const& rhoD) noexcept
{
  // ndeak note - calculates diffusivity in units of (cm2/s)*(g/cm3)
  // factor has units of J/K-C
   using namespace amrex::literals;

   amrex::Real rho = 0.0;
   for (int n = 0; n < NUM_SPECIES; n++) {
      rho += rhoY(i,j,k,n);
   }
   rhoD(i,j,k,Ecomp) = rho * Ks(i,j,k,Ecomp) * T(i,j,k) * factor;

  // TODO remove after testing 1D code
  rhoD(i,j,k,Ecomp) = rho * 846.28133089;

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
   // ndeak note - returns mobility in units of cm2/V-s
   // mwt in units of g/mol
   // a_zk in units of C/kg
   // PP_RU_MKS in units of J/mol-K
   // mobility already takes into account different drift direction of + and - ions due to a_zk
   using namespace amrex::literals;
   amrex::Real rho = 0.0;
   amrex::Real PP_RU_MKS = 8.31446261815324;
   for (int n = 0; n < NUM_SPECIES; n++) {
      rho += rhoY(i,j,k,n);
   }
   for (int n = 0; n < NUM_SPECIES; n++) {
   //   if(n != E_ID) Ksp(i,j,k,n) = rhoD(i,j,k,n) * mwt[n] * 0.001 * a_zk[n] / ( rho * PP_RU_MKS * T(i,j,k) );
   //   // if ( n == E_ID ) Ksp(i,j,k,n) = 0.0;         // Set the E species mobility to zero. Electron are treated separately.
   //   // if ( n == 0 ) Ksp(i,j,k,n) = 0.0;         // Set the E species mobility to zero. Electron are treated separately.
       if(n != E_ID) Ksp(i,j,k,n) = 0.0;
   }

   // TODO remove after testing 1D code
   // for (int n = 0; n < NUM_SPECIES; n++) {
   //    if(n != E_ID) Ksp(i,j,k,n) = 2.738979530 * amrex::Math::copysign(1.0, a_zk[n]);
   // }
   Ksp(i, j, k, 4) = 2.738979530;
   Ksp(i, j, k, 5) = 2.738979530;
   Ksp(i, j, k, 6) = 2.738979530;
   Ksp(i, j, k, 7) = 2.738979530;
   Ksp(i, j, k, 8) = 2.738979530;
   Ksp(i, j, k, 9) = -2.738979530;


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
