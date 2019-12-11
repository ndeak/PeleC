# ------------------  INPUTS TO MAIN PROGRAM  -------------------
max_step = 400

# PROBLEM SIZE & GEOMETRY
geometry.is_periodic =  0 0 0
geometry.coord_sys   =  0       # 0 => cart
geometry.prob_lo     =  -1.0 -1.0 -1.0
geometry.prob_hi     =   1.0  1.0  1.0
amr.n_cell           =    16   16   16

# >>>>>>>>>>>>>  BC KEYWORDS <<<<<<<<<<<<<<<<<<<<<<
# Interior, UserBC, Symmetry, SlipWall, NoSlipWall
# >>>>>>>>>>>>>  BC KEYWORDS <<<<<<<<<<<<<<<<<<<<<<
pelec.lo_bc       =   "FOExtrap" "FOExtrap" "FOExtrap"
pelec.hi_bc       =   "FOExtrap" "FOExtrap" "FOExtrap"

# WHICH PHYSICS
pelec.do_hydro = 0
pelec.do_mol_AD = 0
pelec.nscbc_adv = 0
pelec.do_react = 0
pelec.ppm_type = 2
pelec.allow_negative_energy = 0
pelec.diffuse_temp = 1
pelec.diffuse_vel  = 0
pelec.diffuse_spec = 0
pelec.diffuse_enth = 1
pelec.diffuse_aux  = 1

# TIME STEP CONTROL
pelec.dt_cutoff      = 5.e-20  # level 0 timestep below which we halt
pelec.fixed_dt         = 0.00005

# DIAGNOSTICS & VERBOSITY
pelec.sum_interval   = 1       # timesteps between computing mass
pelec.v              = 1       # verbosity in PeleC cpp files
amr.v                = 1       # verbosity in Amr.cpp

# REFINEMENT / REGRIDDING
amr.max_level       = 0       # maximum level number allowed
amr.ref_ratio       = 2 2 2 2 # refinement ratio
amr.regrid_int      = 20       # how often to regrid
amr.blocking_factor = 4       # block factor in grid generation
amr.max_grid_size   = 16

# CHECKPOINT FILES
amr.check_file      = chk      # root name of checkpoint file
amr.check_int       = 5000       # number of timesteps between checkpoints

# PLOTFILES
amr.plot_file       = plt
amr.plot_int        = 400
amr.derive_plot_vars= ALL
pelec.plot_massfrac = 1

# PROBIN FILENAME
amr.probin_file = diff-3d.probin

