// Definition of objects and methods appended to PeleC class.
#ifndef _PELEC_PLASMA_H_
#define _PELEC_PLASMA_H_


  //-------------------------------------
  // Data

  // Static variables/options
  static int         PhiV, nE, Efieldx, Efieldy, Efieldz;   // Additional scalar indices
  static int         ef_verbose;                            // Runtime verbose
  static int         ef_debug;                              // Use VisMF to dump data (a lot ...)
  static int         ef_use_NLsolve;                        // Switch nonlinear solve ON/OFF
  static int         ef_use_nEimplicit;                     // Switch nE implicit linear system solve ON/OFF
  static int         ef_use_nEDiffImp;                      // Switch implicit diffusion for electrons ON/OFF
  static bool        def_harm_avg_cen2edge;                 // Face averaging type
  static int         ion_bc_type;                           // Sets the ion electrode loss type (0 = weak efield approx., 1 = strong efield approx.)
  static int         zero_bc_flux;                          // Set to 1 to zero out all electrode fluxes for electrons and ions
  static int         zero_bc_grad;                          // Set to 1 to set a flux consistent with a zero-grad BC
  static int         ef_use_joule_heating;                  // Set to 1 to include joule heating effects
  static int         ef_use_radiative_losses;               // Include radiative heat losses
  static amrex::Real secondary_em_coef;                     // Boundary emission
  static amrex::Real electron_emit_const;                     // Boundary emission
  static amrex::Real pulse_freq;                            // Pulse frequency (Hz)
  static amrex::Real pulse_fwhm;                            // Pulse FWHM (s)
  static amrex::Real pulse_peak;                            // Pulse peak voltage (erg/C)
  static amrex::Real pulse_timing;                          // Timing of first pulse peak (s)
  static int         pulse_num;                             // Number of pulses
  static amrex::Real pulse_dt;                              // Constant time step size taken around each plasma pulse (s)
  static int         ef_triangle_pulse;                     // Use a triangular pulse with pulse_fwhm rise time?
  static int         ef_trapezoidal_pulse;                  // Use a trapezoidal pulse with pulse_fwhm/2.0 rise time, and pulse_fwhm plateau?
  static int         ef_sigmoid_pulse;                      // Use a sigmoid pulse with pulse_fwhm/2.0 rise time, and pulse_fwhm plateau?
  static amrex::Real ef_pulse_delay;                        // Delay time for sigmoid pulse profile (s)
  static amrex::Real ef_pulse_rise;                         // Rise time for the sigmoid pulse profile (s)
  static amrex::Real ef_pulse_plateau;                       // Plateau time for the sigmoid pulse profile (s)
  static amrex::Real pulse_sigma;     
  static amrex::Real curr_voltage;                          // Voltage at time t=n (erg/C)
  static amrex::Real dfact;
  static amrex::Real sfact;
  static int         pac_mechanism;            // If using PAC mechanism, output maximum reactor-type data in monitor file
  static int         ion_rate_type;           // Controls the ionization rate calculations used for photoionization (0 = upd. Kossyi, 1 = Morrow, 2 = BOLSIG+)
  static amrex::Real ef_electron_heating_pct;                // Percentage of electron energy that results in fast heating (treated as instantaneous)
  static int         ef_ambiDiff;                            // Use an ambipolar diffusion model for charged particle transport?
  
  // Transport options
  static int         ef_constEleTransport;                   // Use constant electron mobility (and diffusivity, via Einsten relation)
  static amrex::Real ef_eleMobility;                         // (Normalized) constant electron mobility mu*N (1/V-m-s)  
  static amrex::Real ef_eleDiffusivity;                      // (Normalized) constant electron diffusivity De*N (1/m-s)  
  static int         ef_relax_eleDiff;                       // Use a factor to reduce electron mobility by order of magnitude in between pulses?

  // Plotting options
  static bool plot_numdens;                                   // Plot number densities?
  
  // Circuit model options
  static int         ef_circuit_model;                         // Set to 0 for no circuit model, 1 for simple model
  static amrex::Real ef_circuit_time_delay;                           // Time taken for signal to cross transmission line (s)
  static amrex::Real ef_circuit_impedance;                           // Characteristic impedance Z_0, (erg-s/C^2)
  static amrex::Real ef_circuit_upstream_resistance;                  // upstream resistance (erg-s/C^2)
  static amrex::Real ef_circuit_capacitance;                  // Circuit capacitance (C^2/erg, should be calculated ahead of time) 
  static amrex::GpuArray<amrex::Real, NUM_SPECIES> zk;      // Species charge per unit mass (C/g)
  static amrex::GpuArray<int, NUM_SPECIES> zk_num;      // Species charge number 
  static amrex::GpuArray<amrex::Real, 20000>   time_ts;
  static amrex::GpuArray<amrex::Real, 20000>   sourceVoltage_ts;
  static amrex::GpuArray<amrex::Real, 20000>   eleVoltage_ts;
  static amrex::GpuArray<amrex::Real, 20000>   incidentWave_ts;
  static amrex::GpuArray<amrex::Real, 20000>   reflectedWave_ts;
  static amrex::GpuArray<amrex::Real, 20000>   sourceCurrent_ts;
  static amrex::GpuArray<amrex::Real, 20000>   eleCurrent_ts;
  static amrex::GpuArray<amrex::Real, 1000>   lin_residuals;
  static bool ef_circuit_load_data;                           // Load data from 'circuit.dat' file in current directory
  static int ef_circuit_load_num;                             // Number of previous time steps taken if loading from circuit.dat file

  // Efield Poisson solve options
  static amrex::Real ef_PoissonTol;
  static int         ef_PoissonMaxOrder;
  static int         ef_PoissonVerbose;
  static int         ef_PoissonMaxIter;
  static int         ef_noSpaceCharge;        // Solve the Poisson problem without accounting for space charge effects
  static int         ef_constVoltage;         // Apply a constant voltage (no time variance)
  static int         ef_do_drift;             // Include drift processes for charged species
  static int         ef_do_photoionization;   // Include photoionization
  static int         ef_star_update;         // Do update after just first start state (=1), and second star state (=2), neither (=0)
  static int         ef_semiImpEfield;         // Use a semi-implicit approach to solve for the reduced electric field

  // Non-linear solve options and variables
  static int         ef_use_PETSC_direct;      // Flag to trigger using PETSC direct solve
  static amrex::Real ef_lambda_jfnk;           // Perturbation mag. for JFNK differencing
  static int         ef_diffT_jfnk;            // Differencing type: 1 - one-sided first order, 2 - centered second order
  static int         ef_maxNewtonIter;         // Maximum number of newton iterations
  static amrex::Real ef_newtonTol;             // Tolerance of newton iteration (on inf. norm.)
  static int         ef_GMRES_size;            // Size of the GMRES Krylov base
  static amrex::Real ef_GMRES_reltol;          // Relarive tolerance of the GMRES solve
  static int         ef_GMRES_maxRst;          // Maximum number of GMRES restart
  static int         ef_GMRES_verbose;         // GMRES verbosity
  static int         ef_time_order;            // Order of temporal discretization in coupled system (1 or 2)
  static int         ef_space_order;           // Order of spatial discretization for the advective operator in coupled system (1 or 2)
  static int         ef_pred_charge;           // Use ion MOLSrc and I_R to predict charge at time n+1 in the coupled system

  // Schur preconditioner options
  static amrex::Real ef_PC_MG_Tol;             // Preconditioner linear solver tolerance
  static int         ef_PC_fixedIter;          // Preconditioner fixed number of iterations
  static int         ef_PC_approx;             // Type of approximation of Stilda

  // O2 and N2 species indicies - used for photoionization calculations
  static int         O2_idx;
  static int         N2_idx;

  // MFs
  amrex::MultiFab     Efield;
  amrex::MultiFab     Efield_L_p2;                // "Laplacian" electric field dotted with itself (used for calculating capacitance)
  amrex::MultiFab     old_Efield;
  amrex::MultiFab     redEfield;
  amrex::MultiFab     KSpec_old;     // Need both old and new since adv is at t^n+1/2
  amrex::MultiFab     KSpec_new;
  amrex::MultiFab     spec_drift;
  amrex::MultiFab     coeffs_old;         // Species diffusivities (used to calculate mobilities)
  amrex::MultiFab     Q_ext;  
  amrex::MultiFab     Qaux_ext;  
  std::array<const amrex::MultiFab* ,AMREX_SPACEDIM> Efield_edge;
  FluxBoxes gphi;
  FluxBoxes DegnE;
  amrex::MultiFab** gradPhiV;
  amrex::MultiFab** DegradnE;
  amrex::MultiFab     PI_source;  
  amrex::MultiFab     dielectric_ts;
  amrex::MultiFab     flux_current_mf;
  amrex::MultiFab     Ke_cc_mf;
  amrex::MultiFab     De_cc_mf;
  amrex::MultiFab     joule_heating;  
  amrex::MultiFab     radiative_losses;  

  amrex::MultiFab     spec_2ndo_gradients;    // 2nd order gradients used in circuit model displacement current calculation

  // Non-linear solve MFs/data
  // Transport
  FluxBoxes mob_e;
  amrex::MultiFab**   Ke_ec;
  FluxBoxes diff_e;
  amrex::MultiFab**   De_ec;
  FluxBoxes gasN_fb;
  amrex::MultiFab**   gasN_ec;
  FluxBoxes ionFlx_fb;
  amrex::MultiFab**   ionFlx;

  // Flux boxes for all species (used in semi-implicit efield calculations)
  FluxBoxes spec_edge;
  amrex::MultiFab** spec_edge_mfs;
  
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
  amrex::Real                nl_residNormnE;
  amrex::Real                nl_residNormphiV;
  amrex::Real                nl_curr_residNorm;           
  amrex::Real                nl_correctionNorm;
  amrex::Real                nl_time_in;
  amrex::MultiFab            bg_charge;
  amrex::MultiFab            ef_state_old;
  amrex::MultiFab            nl_state;
  amrex::MultiFab            nl_resid;
  amrex::MultiFab            nl_F_old;        // F(U^n) (used for the nE component in the coupled system trapezoidal time integration)
  amrex::MultiFab            nl_nE_2ndo_slopes;
  amrex::MultiFab            elec_Ueff[3];
  amrex::MultiFab            gasN_cc;
  amrex::MultiFab            ionFlx_eb;
  amrex::MultiFab            tmp_nE_forcing;
  amrex::Real                disp_current = 0.0;
  amrex::Real                flux_current = 0.0;
  amrex::Real                total_current = 0.0;
  amrex::Real                level_capacitance;     // Individual capacitance component of a given level
  amrex::MultiFab            nE_state;
  amrex::MultiFab            nE_state_old;

  // Preconditioner data
#ifdef PELEC_USE_EB
  amrex::MLEBABecCecLap   *pnp_pc_diff = nullptr;
  amrex::MLEBABecLap      *pnp_pc_drift = nullptr;
  amrex::MLEBABecLap      *pnp_pc_Stilda = nullptr;
#else
  amrex::MLABecCecLaplacian   *pnp_pc_diff = nullptr;
  amrex::MLABecLaplacian      *pnp_pc_drift = nullptr;
  amrex::MLABecLaplacian      *pnp_pc_Stilda = nullptr;
#endif
  amrex::MLMG                 *mg_diff = nullptr;
  amrex::MLMG                 *mg_drift = nullptr;
  amrex::MLMG                 *mg_Stilda = nullptr;

  //-------------------------------------
  // Methods

  // Initialize user-defined options and such
  static void plasma_init();

  // Build MFs/FluxBoxes ...
  void plasma_define_data();

  // Set/get voltage at a given time
  void setCurrVoltage(amrex::Real time);
  amrex::Real getCurrVoltage(amrex::Real time);

  // Electric potential Poisson solve
  void solveEF (amrex::Real time,
                amrex::Real dt, 
                ProbParmDevice const& prob_parm, 
                const amrex::MultiFab& Sbord,
                bool lapl_solve = false);

  // Implicit nE solve
  void nESolveImplicit (amrex::Real time,
                amrex::Real dt, 
                const amrex::MultiFab &I_R_in,
                const amrex::MultiFab & Sbord,
                      amrex::MultiFab & forcing_nE);

  // Implicit nE diffusion solve
  void nEDiffuseImplicit (amrex::Real time,
                amrex::Real dt, 
                const amrex::MultiFab & Sbord);

  // Photoionization helmholtz solve
  void solvePI (amrex::Real time,
                amrex::Real dt, 
                ProbParmDevice const& prob_parm);

  // Transport coefficients
  void ef_calc_transport(const amrex::MultiFab& S, const amrex::Real &time);

  // Version of ef transport that fits inside Diffusion.cpp loop
  void ef_calc_transport(amrex::Box const& bx,
                         amrex::Array4<const amrex::Real> const& rhoY_in,
                         amrex::Array4<const amrex::Real> const& EoN_in,
                         amrex::Array4<amrex::Real> const& Ke_out,
                         amrex::Array4<amrex::Real> const& rhoDe_out,
                         amrex::Array4<amrex::Real> const& K_out) ;
  
  // Boundary conditions
  void setBCPhiV ( std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &linOp_lobc,
                   std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &linOp_hibc);

  void setBCPI ( std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &linOp_lobc,
                   std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &linOp_hibc);

// Setup PhiV BC for linear Poisson solve  
  void ef_set_PoissonBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                        std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc);

// Setup nE BC
  void ef_set_neBC(std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_lobc,
                   std::array<amrex::LinOpBCType,AMREX_SPACEDIM> &mlmg_hibc);

// Circuit model functions

  void ef_circuitModel(amrex::Real time, amrex::Real dt);

  void ef_fluxCurrent(const amrex::MultiFab &state_curr,
                      const amrex::MultiFab &mu_curr,
                      const amrex::MultiFab &E_curr,
                      const amrex::MultiFab &D_curr);
  
  void ef_waveVoltageSources(amrex::Real time);
  
  void ef_voltagesCurrents(amrex::Real time, amrex::Real dt);

  void gapCapacitance(amrex::Real time);

  void circuitFileSetup();

  void ef_loadCircuitData(amrex::Real time);

  void writeCircuitFile(amrex::Real time, int step_num);

// non-linear solve functions
  void ef_solve_NL( const amrex::Real      &dt,
                    const amrex::Real      &time,
                    const amrex::MultiFab  &state_in,
                    const amrex::MultiFab  &Forcing_state,
                    const amrex::MultiFab  &I_R_in,
                          amrex::MultiFab  &Forcing_nE);

// Newton exit test
  int testExitNewton(const amrex::MultiFab  &res,
                     const amrex::MultiFab  &nDir,
                           int              newtonIter);

  void compute_bg_charge( const amrex::Real      &dt,
                          const amrex::MultiFab &state_old,
                          const amrex::MultiFab &MOL_src,
                          const amrex::MultiFab &I_R_in);

  void compute_gasN( const amrex::Real      &dt,
                     const amrex::MultiFab &state_old,
                     const amrex::MultiFab &MOL_src,
                     const amrex::MultiFab &I_R_in);

  void ef_nlResidual(const amrex::Real        &dt,
                     const amrex::MultiFab    &a_nl_state,
                           amrex::MultiFab    &a_nl_resid,
                           int                update_res_scaling = 0,
                           int                update_precond = 0,
                           int                evalF = 0);

  void ef_calcGradPhiV(const amrex::Real&       a_time,
                             amrex::MultiFab    &a_phiv,
                             amrex::MultiFab*   gradphi[AMREX_SPACEDIM]);

// Laplacian of phiV
  void compPhiVLap(amrex::MultiFab    &a_phi,
                   amrex::MultiFab    &phiLap,
                   amrex::MultiFab**  a_gphiV,
                   ProbParmDevice const& prob_parm);

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

  void ef_normMFsingle(const amrex::MultiFab  &a_vec,
                       amrex::Real  &norm,
                       int compNum);

  void jtimesv(const amrex::MultiFab  &x, amrex::MultiFab  &Ax);

  void ef_setUpPrecond(const amrex::Real &dt_lcl,
                       const amrex::MultiFab& a_nl_state);

  void ef_applyPrecond(const amrex::MultiFab  &v,
                             amrex::MultiFab  &Pv);


  amrex::MultiFab& get_PI_data () noexcept { return PI_source; }

  amrex::Real getFluxCurrent () noexcept {return flux_current; }
  void setFluxCurrent (amrex::Real current_in) noexcept {flux_current = current_in; }

  amrex::Real getGapCapacitance () noexcept {return level_capacitance; }
  void setGapCapacitance (amrex::Real capacitance_in) noexcept {level_capacitance = capacitance_in; }

  // This value is for setting the circuit values calculated at the finest level at coarser levels
  void setCircuitValues(int step_num, amrex::Real timeval,
                        amrex::Real sourceVolt, amrex::Real eleVolt, 
                        amrex::Real sourceCurr, amrex::Real eleCurr, 
                        amrex::Real incWave, amrex::Real refWave)
                        noexcept {time_ts[step_num] = timeval; 
                                  sourceVoltage_ts[step_num] = sourceVolt;
                                  eleVoltage_ts[step_num + 1] = eleVolt;
                                  sourceCurrent_ts[step_num] = sourceCurr;
                                  eleCurrent_ts[step_num] = eleCurr;
                                  incidentWave_ts[step_num] = incWave;
                                  reflectedWave_ts[step_num] = refWave;}     

#endif
