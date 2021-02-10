#ifndef _PLASMA_H_
#define _PLASMA_H_

// Definition of global plasma objects

namespace EFConst
{
   extern amrex::Real eps0;          //Free space permittivity (C/(V.m))
   extern amrex::Real eps0_cgs;      //Free space permittivity (C/(erg.cm))
   extern amrex::Real epsr; 
   extern amrex::Real elemCharge;    //Coulomb per charge
   extern amrex::Real Na;            //Avogadro's number
   extern amrex::Real kB;            //Boltzman constant
   extern amrex::Real PP_RU_MKS;     //Universal gas constant (J/mol-K)
   extern amrex::Real PP_RU_CGS;     //(erg/mol-K)
}
#endif
