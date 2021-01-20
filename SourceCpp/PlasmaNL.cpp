#include <PeleC.H>

using namespace amrex;

void PeleC::jtimesv(const MultiFab &v,
                          MultiFab &Jv)
{
    Real vNorm;
    ef_normMF(v,vNorm);
 
    // x is zero, Ax is zero and return
    if ( vNorm == 0.0 ) {
       Jv.setVal(0.0);
       return;
    }
 
    Real delta_pert = ef_lambda_jfnk * ( ef_lambda_jfnk + nl_stateNorm / vNorm );
 
    if ( ef_diffT_jfnk == 1 ) {
       // Create perturbed state
       MultiFab S_pert(grids,dmap,2,2);
       MultiFab::Copy(S_pert, nl_state, 0, 0, 2, 2);
       MultiFab::Saxpy(S_pert,delta_pert,v, 0, 0, 2 ,0);
 
       // Get perturbed residual
       MultiFab res_pert(grids,dmap,2,1);
       ef_nlResidual(nl_dt,S_pert,res_pert);
       res_pert.mult(-1.0);
 
       // Get Ax by finite differece
       MultiFab::LinComb(Jv,1.0,res_pert,0,-1.0,nl_resid,0,0,2,0);
       Jv.mult(-1.0/delta_pert);
    } else if ( ef_diffT_jfnk == 2 ) {
       // Create perturbed states
       MultiFab S_pertm(grids,dmap,2,2);
       MultiFab S_pertp(grids,dmap,2,2);
       MultiFab::Copy(S_pertp, nl_state, 0, 0, 2, 2);
       MultiFab::Copy(S_pertm, nl_state, 0, 0, 2, 2);
       MultiFab::Saxpy(S_pertp,delta_pert,v, 0, 0, 2 ,0);
       MultiFab::Saxpy(S_pertm,-delta_pert,v, 0, 0, 2 ,0);
 
       // Get perturbed residuals
       MultiFab res_pertp(grids,dmap,2,1);
       MultiFab res_pertm(grids,dmap,2,1);
       ef_nlResidual(nl_dt,S_pertp,res_pertp);
       ef_nlResidual(nl_dt,S_pertm,res_pertm);
       res_pertm.mult(-1.0);
       res_pertp.mult(-1.0);
 
       // Get Ax by finite differece
       MultiFab::LinComb(Jv,1.0,res_pertp,0,-1.0,res_pertm,0,0,2,0);
       Jv.mult(-0.5/delta_pert);
    } else {
       Abort(" Unrecognized ef_diffT_jfnk. Should be either 1 (one-sided) or 2 (centered)");
    }
 
}

void PeleC::ef_solve_NL(const Real     &dt,
                        const Real     &time,
                              MultiFab &ForcingnE)
{
   BL_PROFILE("PC_EF::ef_solve_NL()");
}

void PeleC::ef_nlResidual(const Real      &dt_lcl,
                          const MultiFab  &a_nl_state,
                                MultiFab  &a_nl_resid,
                                int       update_res_scaling,
                                int       update_precond){
   BL_PROFILE("PC_EF::ef_nlResidual()");
}

void PeleC::ef_applyPrecond (const MultiFab  &v,
                                   MultiFab  &Pv) {
   BL_PROFILE("PC_EF::ef_applyPrecond()");
}

void PeleC::ef_normMF(const MultiFab &a_vec,
                            Real &norm){
    norm = 0.0;
    for ( int comp = 0; comp < a_vec.nComp(); comp++ ) {
       norm += MultiFab::Dot(a_vec,comp,a_vec,comp,1,0);
    }
    norm = std::sqrt(norm);
}
