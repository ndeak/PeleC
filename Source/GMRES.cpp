#include <GMRES.H>

using namespace amrex;

GMRESSolver::GMRESSolver() {
   m_verbose = 0;
   m_restart = 1;
}
GMRESSolver::~GMRESSolver() {}

void
GMRESSolver::define(PeleC* a_level,
                    const int a_KrylovSize, 
                    const int a_nComp,
                    const int a_nGrow
#ifdef PELEC_USE_EB
                  , const amrex::MFInfo& info,
                    const amrex::FabFactory<amrex::FArrayBox>& factory
#endif
              )
{
   BL_PROFILE("GMRESSolver::define()");

   m_level = a_level;
   m_geom = m_level->Geom();
   m_grids = m_level->boxArray();
   m_dmap = m_level->DistributionMap();
   m_nComp = a_nComp;
   m_nGrow = a_nGrow;
   m_krylovSize = a_KrylovSize;

// Build krylov base memory
   KspBase.resize(m_krylovSize+1);
#ifdef PELEC_USE_EB
   for (int n = 0; n <= m_krylovSize ; ++n) {
      KspBase[n].define(m_grids,m_dmap,m_nComp,m_nGrow,info,factory);
   }

// Work MultiFabs
   Ax.define(m_grids,m_dmap,m_nComp,m_nGrow,info,factory);
   res.define(m_grids,m_dmap,m_nComp,m_nGrow,info,factory);
#else
   for (int n = 0; n <= m_krylovSize ; ++n) {
      KspBase[n].define(m_grids,m_dmap,m_nComp,m_nGrow);
   }

// Work MultiFabs
   Ax.define(m_grids,m_dmap,m_nComp,m_nGrow);
   res.define(m_grids,m_dmap,m_nComp,m_nGrow);
#endif

// Work Reals
   H.resize(m_krylovSize+1);
   givens.resize(m_krylovSize+1);
   for (int n = 0; n <= m_krylovSize ; ++n) {
      H[n].resize(m_krylovSize,0.0);
      givens[n].resize(2,0.0);
   }
   y.resize(m_krylovSize+1,0.0);
   g.resize(m_krylovSize+1,0.0);
}

JtimesVFunc
GMRESSolver::jtimesv() const noexcept
{
   return m_jtv;
}

PrecondFunc
GMRESSolver::precond() const noexcept
{
   return m_prec;
}

NormFunc
GMRESSolver::norm() const noexcept
{
   return m_norm;
}

void
GMRESSolver::setJtimesV(JtimesVFunc a_jtv)
{
   m_jtv = a_jtv;
}

void
GMRESSolver::setPrecond(PrecondFunc a_prec)
{
   m_prec = a_prec;
}

void
GMRESSolver::setNorm(NormFunc a_norm)
{
   m_norm = a_norm;
}

int
GMRESSolver::solve(MultiFab& a_sol,
                   const MultiFab& a_rhs,
                   amrex::Real a_abs_tol,
                   amrex::Real a_rel_tol,
                   amrex::GpuArray<amrex::Real, 10000> lin_residuals)
{
   BL_PROFILE("GMRESSolver::solve()");

// Checks
   AMREX_ALWAYS_ASSERT(m_jtv != nullptr);
   AMREX_ALWAYS_ASSERT(a_sol.nComp() == m_nComp && a_rhs.nComp() == m_nComp);
   if ( m_prec == nullptr ) Print() << "Using unpreconditioned GMRES ! Might take a while to converge ...\n";


   initResNorm = computeResidualNorm(a_sol,a_rhs);                      // Initial resisual norm
   Real rhsNorm  = computeNorm(a_rhs);                                  // RHS norm
   targetResNorm = initResNorm * a_rel_tol;                             // Target relative tolerance

   if ( m_verbose > 0 ) {
      amrex::Print() << "GMRES: Initial rhs = " << rhsNorm << "\n";
      amrex::Print() << "GMRES: Initial residual = " << initResNorm << "\n";
   }
   // if ( initResNorm < a_abs_tol ) {
   //    amrex::Print() << "GMRES: no need for iterations, abs_tol = " << a_abs_tol << "\n" ;
   //    return 0;
   // }

   iter_count = 0;
   restart_count = 0;
   m_converged = false;
   do {
//    Prepare for solve
      amrex::Print() << "DOIN A RESTART!\n";
      prepareForSolve();
      one_restart(a_sol,a_rhs, lin_residuals);
      restart_count++;
   } while( !m_converged && restart_count < m_restart );

   Real finalResNorm = computeResidualNorm(a_sol,a_rhs);                // Final resisual norm
   if ( m_verbose > 0 ) amrex::Print() << "GMRES: Final residual, resid/resid0 = " << finalResNorm << ", "
                                       << finalResNorm/initResNorm << "\n";
   return iter_count;
}

void
GMRESSolver::prepareForSolve()
{
   // Initilize or cleanup GMRES data
   for ( int k = 0 ; k <= m_krylovSize ; ++k ) {
      g[k] = 0.0;
      y[k] = 0.0;
      for ( int n = 0 ; n < m_krylovSize ; ++n ) {
         H[k][n] = 0.0;
      }
      givens[k][0] = 0.0;
      givens[k][1] = 0.0;
   }
}

void
GMRESSolver::one_restart(MultiFab& a_x, const MultiFab& a_rhs, amrex::GpuArray<amrex::Real, 10000> lin_residuals)
{
   computeResidual(a_x,a_rhs,res);

   Real resNorm_0 = computeNorm(res);
   if ( m_verbose > 1 ) amrex::Print() << "     [Restart:"<< restart_count << "] initial relative res: " << resNorm_0/initResNorm << "\n";

   // Initialize KspBase with normalized residual
   g[0] = resNorm_0;
   if(resNorm_0 == 0.0) {
      amrex::Print() << "WARNING: RESNORM IN GMRES ONE RESTART US ZERO!\n";
   }
   res.mult(1.0/resNorm_0);
   MultiFab::Copy(KspBase[0],res, 0 ,0, m_nComp, 0);

   int k_end;
   Real resNorm = resNorm_0;

   for ( int k = 0 ; k < m_krylovSize ; ++k ) {
      // Do one GMRES iteration, update the residual norm
      one_iter(k,resNorm);
      lin_residuals[iter_count] = resNorm;
      iter_count++;

      // Test exit condition
      if ( resNorm < targetResNorm ) {
         amrex::Print() << "EXIT CONDITION MET ON ITER " << k << "\n";
         m_converged = true;
         k_end = k;
         break;
      }

      // Last iteration and not converged yet, just set k_end
      if ( k == m_krylovSize - 1 ) k_end = m_krylovSize - 1;
   }

   // Solve the minimization problem H.y = g
   y[k_end] = g[k_end]/H[k_end][k_end];
   for ( int k = k_end-1; k >= 0; --k ) {
      Real sum_tmp = 0.0;
      for ( int j = k+1; j <= k_end; ++j ) {
         sum_tmp += H[k][j] * y[j];
      }
      y[k] = ( g[k] - sum_tmp ) / H[k][k];
   }

   // Compute solution update
   MultiFab update(m_grids,m_dmap,m_nComp,0);
   update.setVal(0.0);
   for ( int i = k_end; i >= 0; --i ) {
      MultiFab::Saxpy(update,y[i],KspBase[i], 0, 0, m_nComp, 0);
   }

   // Update the solution MF
   MultiFab::Add(a_x,update,0,0,m_nComp,0);
}

void
GMRESSolver::one_iter(const int iter, Real &resNorm)
{
   BL_PROFILE("GMRESSolver::one_iter()");
   if ( m_verbose > 1 ) amrex::Print() << "     [Iter:" << iter_count << "] residual norm: " << resNorm / initResNorm << "\n";
   appendBasisVector(iter,KspBase);
   gramSchmidtOrtho(iter,KspBase);
   resNorm = givensRotation(iter);
}

void
GMRESSolver::appendBasisVector(const int iter, Vector<MultiFab>& Base)
{
   if ( m_prec == nullptr ) {
      MEMBER_FUNC_PTR(*m_level,m_jtv)(Base[iter],Base[iter+1]);
   } else {
      MEMBER_FUNC_PTR(*m_level,m_jtv)(Base[iter],Ax);
      MEMBER_FUNC_PTR(*m_level,m_prec)(Ax,Base[iter+1]);
   }
}

void
GMRESSolver::gramSchmidtOrtho(const int iter, Vector<MultiFab>& Base)
{

#ifdef PELEC_USE_EB
    auto const& fact =
      dynamic_cast<amrex::EBFArrayBoxFactory const&>(Base[iter].Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();
#endif

    for ( int row = 0; row <= iter; ++row ) {
      Real Hsum = 0.0;
      int nghost = 0;
      for(int n = 0; n<m_nComp; n++){
        for (MFIter mfi(Base[iter+1],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
          const Box& bx = mfi.growntilebox(nghost);
          auto const& comp1_ar  = Base[iter+1].const_array(mfi,n);
          auto const& comp2_ar  = Base[row].const_array(mfi,n);
#ifdef PELEC_USE_EB
          auto flag_arr = flags.const_array(mfi);
#endif
          amrex::ParallelFor(bx, [comp1_ar, comp2_ar, &Hsum
#ifdef PELEC_USE_EB
                              , flag_arr
#endif
                                        ]
          AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
#ifdef PELEC_USE_EB
            if(!flag_arr(i,j,k).isCovered()){
#endif
              Hsum += comp1_ar(i,j,k) * comp2_ar(i,j,k);
#ifdef PELEC_USE_EB
            }
#endif
          });
        }
      }

      ParallelAllReduce::Sum(Hsum, ParallelContext::CommunicatorSub());
      H[row][iter] = Hsum;
      // H[row][iter] = MultiFab::Dot(Base[iter+1],0,Base[row],0,m_nComp,0);
      Real GS_corr = - H[row][iter];
      MultiFab::Saxpy(Base[iter+1],GS_corr,Base[row],0,0,m_nComp,0);
      if ( check_GramSchmidtOrtho ) {
        Hsum = 0.0;
        for(int n = 0; n<m_nComp; n++){
          for (MFIter mfi(Base[iter+1],TilingIfNotGPU()); mfi.isValid(); ++mfi)
          {
            const Box& bx = mfi.growntilebox(nghost);
            auto const& comp1_ar  = Base[iter+1].const_array(mfi,n);
            auto const& comp2_ar  = Base[row].const_array(mfi,n);
#ifdef PELEC_USE_EB
            auto flag_arr = flags.const_array(mfi);
#endif
            amrex::ParallelFor(bx, [comp1_ar, comp2_ar, &Hsum
#ifdef PELEC_USE_EB
                                , flag_arr
#endif
                                          ]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
#ifdef PELEC_USE_EB
              if(!flag_arr(i,j,k).isCovered()){
#endif
                Hsum += comp1_ar(i,j,k) * comp2_ar(i,j,k);
#ifdef PELEC_USE_EB
              }
#endif
            });
          }
        }
        ParallelAllReduce::Sum(Hsum, ParallelContext::CommunicatorSub());

        Real Hcorr = Hsum;
        // Real Hcorr = MultiFab::Dot(Base[iter+1],0,Base[row],0,m_nComp,0);
        if ( std::fabs(Hcorr) > 1.0e-15 ) {
           H[row][iter] += Hcorr;
           GS_corr = - Hcorr;
           MultiFab::Saxpy(Base[iter+1],GS_corr,Base[row],0,0,m_nComp,0);
        }
      }
   }
   Real normNewVec = computeNorm(Base[iter+1]);
   H[iter+1][iter] = normNewVec;
   if ( normNewVec > 0 ) Base[iter+1].mult(1.0/normNewVec);
}

Real
GMRESSolver::givensRotation(const int iter)
{
   for ( int row = 0; row < iter; ++row ) {
      Real v1 =   givens[row][0] * H[row][iter] - givens[row][1] * H[row+1][iter];
      Real v2 =   givens[row][1] * H[row][iter] + givens[row][0] * H[row+1][iter];
      H[row][iter] = v1;
      H[row+1][iter] = v2;
   }
   Real norm_Hlast = std::sqrt(H[iter][iter]*H[iter][iter] + H[iter+1][iter]*H[iter+1][iter]);
   if ( norm_Hlast > 0.0 ) {
      givens[iter][0] =  H[iter][iter] / norm_Hlast;
      givens[iter][1] = -H[iter+1][iter] / norm_Hlast;
      H[iter][iter] = givens[iter][0] * H[iter][iter] - givens[iter][1] * H[iter+1][iter];
      H[iter+1][iter] = 0.0;
      Real v1 = givens[iter][0] * g[iter] - givens[iter][1] * g[iter+1];
      Real v2 = givens[iter][1] * g[iter] + givens[iter][0] * g[iter+1];
      g[iter] = v1;
      g[iter+1] = v2;
   }
   return std::fabs(g[iter+1]);
}

Real
GMRESSolver::computeResidualNorm(const MultiFab& a_x, const MultiFab& a_rhs)
{
   computeResidual(a_x,a_rhs,res);
   return computeNorm(res);
}

Real
GMRESSolver::computeNorm(const MultiFab& a_vec)
{
   Real norm = 0.0;
   if ( m_norm != nullptr ) {
      MEMBER_FUNC_PTR(*m_level,m_norm)(a_vec,norm);
   } else {
      for ( int comp = 0; comp < m_nComp; comp++ ) {
         norm += MultiFab::Dot(a_vec,comp,a_vec,comp,1,0);
      }
      norm = std::sqrt(norm);
   }
   return norm;
}

void
GMRESSolver::computeResidual(const MultiFab& a_x, const MultiFab& a_rhs, MultiFab& a_res)
{
   BL_PROFILE("GMRESSolver::computeResidual()");
   MEMBER_FUNC_PTR(*m_level,m_jtv)(a_x,Ax);
   if ( m_prec != nullptr ) {
      MultiFab::Xpay(Ax, -1.0, a_rhs, 0, 0, m_nComp, 0);
      MEMBER_FUNC_PTR(*m_level,m_prec)(Ax,a_res);
   } else {
      MultiFab::LinComb(a_res,1.0,Ax,0,-1.0,a_rhs,0,0,m_nComp,0);
   }
}
