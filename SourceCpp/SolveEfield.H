#ifndef PeleLM_EF_K_H_
#define PeleLM_EF_K_H_

#include <AMReX_REAL.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getKappaE(int i, int j, int k, int Ecomp,
          amrex::Array4<      amrex::Real> const& Ks) noexcept
{
  // ndeak note - calculates mobility in units of cm2/V-s
   using namespace amrex::literals;

   Ks(i,j,k,Ecomp) = 0.4_rt;
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
  // ndeak note - calculates diffusivity in units of cm2/s
   using namespace amrex::literals;

   amrex::Real rho = 0.0;
   for (int n = 0; n < NUM_SPECIES; n++) {
      rho += rhoY(i,j,k,n);
   }
   rhoD(i,j,k,Ecomp) = rho * Ks(i,j,k,Ecomp) * T(i,j,k) * factor;
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
      if(n != E_ID) Ksp(i,j,k,n) = rhoD(i,j,k,n) * mwt[n] * 0.001 * a_zk[n] / ( rho * PP_RU_MKS * T(i,j,k) );
      // if ( n == E_ID ) Ksp(i,j,k,n) = 0.0;         // Set the E species mobility to zero. Electron are treated separately.
      // if ( n == 0 ) Ksp(i,j,k,n) = 0.0;         // Set the E species mobility to zero. Electron are treated separately.
   }
}

#endif
