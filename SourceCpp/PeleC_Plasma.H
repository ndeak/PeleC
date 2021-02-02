// Definition of objects and methods appended to PeleC class.
#ifndef _PELEC_PLASMA_H_
#define _PELEC_PLASMA_H_

  //-------------------------------------
  // Data

  // Static variables/options
  static int         PhiV, nE;                              // Additional scalar indices
  static int         ef_verbose;                            // Runtime verbose
  static int         ef_debug;                              // Use VisMF to dump data (a lot ...)
  static int         ef_use_NLsolve;                        // Switch nonlinear solve ON/OFF
  static bool        def_harm_avg_cen2edge;                 // Face averaging type
  static int         ion_bc_type;
  static amrex::Real secondary_em_coef;                     // Boundary emission
  static amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;      // Species charges

  // Efield Poisson solve options
  static amrex::Real ef_PoissonTol;
  static int         ef_PoissonMaxOrder;
  static int         ef_PoissonVerbose;
  static int         ef_PoissonMaxIter;

  // Non-linear solve options
  static int         ef_use_PETSC_direct;      // Flag to trigger using PETSC direct solve
  static amrex::Real ef_lambda_jfnk;           // Perturbation mag. for JFNK differencing
  static int         ef_diffT_jfnk;            // Differencing type: 1 - one-sided first order, 2 - centered second order
  static int         ef_maxNewtonIter;         // Maximum number of newton iterations
  static amrex::Real ef_newtonTol;             // Tolerance of newton iteration (on inf. norm.)
  static int         ef_GMRES_size;            // Size of the GMRES Krylov base
  static amrex::Real ef_GMRES_reltol;          // Relarive tolerance of the GMRES solve
  static int         ef_GMRES_maxRst;          // Maximum number of GMRES restart
  static int         ef_GMRES_verbose;         // GMRES verbosity

  // Schur preconditioner options
  static amrex::Real ef_PC_MG_Tol;             // Preconditioner linear solver tolerance
  static int         ef_PC_fixedIter;          // Preconditioner fixed number of iterations
  static int         ef_PC_approx;             // Type of approximation of Stilda

  // MFs
  amrex::MultiFab     Efield;
  amrex::MultiFab     redEfield;
  amrex::MultiFab     KSpec_old;     // Need both old and new since adv is at t^n+1/2
  amrex::MultiFab     KSpec_new;
  amrex::MultiFab     spec_drift;
  amrex::MultiFab     coeffs_old;         // Species diffusivities (used to calculate mobilities)
  amrex::MultiFab     Q_ext;  
  amrex::MultiFab     Qaux_ext;  
  std::array<const amrex::MultiFab* ,AMREX_SPACEDIM> Efield_edge;
  FluxBoxes gphi;
  amrex::MultiFab** gradPhiV;

  // Non-linear solve MFs/data
  // Transport
  FluxBoxes mob_e;
  amrex::MultiFab**   Ke_ec;
  FluxBoxes diff_e;
  amrex::MultiFab**   De_ec;

  // Scaling
  amrex::Real                nE_scale;
  amrex::Real                phiV_scale;
  amrex::Real                FnE_scale = 1.0;
  amrex::Real                FphiV_scale = 1.0;

  // Temporary holders
  int                        PCLinOp_needUpdate; // Flag to update the preconditioner linear operators
  int                        PCMLMG_needUpdate;  // Flag to update the preconditioner MLMGs
  amrex::Real                nl_dt;
  amrex::Real                nl_stateNorm;
  amrex::Real                nl_residNorm;
  amrex::MultiFab            bg_charge;
  amrex::MultiFab            ef_state_old;
  amrex::MultiFab            nl_state;
  amrex::MultiFab            nl_resid;
  amrex::MultiFab*           elec_Ueff;

  // Preconditioner data
  amrex::MLABecCecLaplacian   *pnp_pc_diff = nullptr;
  amrex::MLABecLaplacian      *pnp_pc_drift = nullptr;
  amrex::MLABecLaplacian      *pnp_pc_Stilda = nullptr;
  amrex::MLMG                 *mg_diff = nullptr;
  amrex::MLMG                 *mg_drift = nullptr;
  amrex::MLMG                 *mg_Stilda = nullptr;

  //-------------------------------------
  // Methods

  // Initialize user-defined options and such
  static void plasma_init();

  // Build MFs/FluxBoxes ...
  void plasma_define_data();

  // Electric potential Poisson solve
  void solveEF (amrex::Real time,
                amrex::Real dt );

  // Transport coefficients
  void ef_calc_transport(const amrex::MultiFab& S, const amrex::Real &time);
  
  // Boundary conditions
  void setBCPhiV ( std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &linOp_lobc,
                   std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &linOp_hibc);

// Setup PhiV BC for linear Poisson solve  
  void ef_set_PoissonBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                        std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc);

// Setup nE BC
  void ef_set_neBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                   std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc);


// non-linear solve functions
  void ef_solve_NL( const amrex::Real      &dt,
                    const amrex::Real      &time,
                    const amrex::MultiFab  &state_in,
                    const amrex::MultiFab  &Forcing_state,
                    const amrex::MultiFab  &I_R_in);

// Newton exit test
  int testExitNewton(const amrex::MultiFab  &res,
                           int              newtonIter);

  void compute_bg_charge( const amrex::Real      &dt,
                          const amrex::MultiFab &state_old,
                          const amrex::MultiFab &MOL_src,
                          const amrex::MultiFab &I_R_in);

  void ef_nlResidual(const amrex::Real        &dt,
                     const amrex::MultiFab    &a_nl_state,
                           amrex::MultiFab    &a_nl_resid,
                           int                update_res_scaling = 0,
                           int                update_precond = 0);

  void ef_calcGradPhiV(const amrex::Real&       a_time,
                             amrex::MultiFab    &a_phiv,
                             amrex::MultiFab*   gradphi[AMREX_SPACEDIM]);

// Laplacian of phiV
  void compPhiVLap(amrex::MultiFab    &a_phi,
                   amrex::MultiFab    &phiLap,
                   amrex::MultiFab**  a_gphiV);

// Electron diffusion term
  void compElecDiffusion(const amrex::MultiFab    &a_nE,
                               amrex::MultiFab    &elecDiff);  

// Electron diffusion term
  void compElecAdvection(amrex::MultiFab    &a_nE,
                         amrex::MultiFab    &a_phiV,
                         amrex::MultiFab**  a_gphiV,
                         amrex::MultiFab    &elecDiff);

  void ef_normMFv(const amrex::MultiFab  &a_vec,
                        amrex::Vector<amrex::Real>  &norm);

// GMRES signatures
  void ef_normMF(const amrex::MultiFab  &a_vec,
                       amrex::Real  &norm);

  void jtimesv(const amrex::MultiFab  &x, amrex::MultiFab  &Ax);

  void ef_setUpPrecond(const amrex::Real &dt_lcl,
                       const amrex::MultiFab& a_nl_state);

  void ef_applyPrecond(const amrex::MultiFab  &v,
                             amrex::MultiFab  &Pv);

#endif
