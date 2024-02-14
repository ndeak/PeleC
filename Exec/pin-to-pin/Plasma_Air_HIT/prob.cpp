#include "prob.H"

void
pc_prob_close()
{
}

extern "C" {
void
amrex_probinit(
  const int* /*init*/,
  const int* /*name*/,
  const int* /*namelen*/,
  const amrex_real* problo,
  const amrex_real* probhi)
{
  // Parse params
  {
    amrex::ParmParse pp("prob");
    pp.query("p", PeleC::h_prob_parm_device->p);
    pp.query("rho", PeleC::h_prob_parm_device->rho);
    pp.query("iname", PeleC::prob_parm_host->iname);
    pp.query("binfmt", PeleC::h_prob_parm_device->binfmt);
    pp.query("restart", PeleC::h_prob_parm_device->restart);
    pp.query("inres", PeleC::h_prob_parm_device->inres);
    pp.query("uin_norm", PeleC::h_prob_parm_device->uin_norm);
    pp.query("phiV_top", PeleC::h_prob_parm_device->PhiV_top);
    pp.query("phiV_bottom", PeleC::h_prob_parm_device->PhiV_bottom);
  }

  // Define the length scale
  PeleC::h_prob_parm_device->L_x = probhi[0] - problo[0];
  PeleC::h_prob_parm_device->L_y = probhi[1] - problo[1];
  PeleC::h_prob_parm_device->L_z = probhi[2] - problo[2];

  // Initial density, velocity, and material properties
  amrex::Real cs;
  amrex::Real cp;
  // Air mass fractions
  for (int n = 0; n < NUM_SPECIES; n++) PeleC::h_prob_parm_device->massfrac[n] = 0.0;
  PeleC::h_prob_parm_device->massfrac[4] = 0.218157027610499;
  PeleC::h_prob_parm_device->massfrac[7] = 0.718100215884559;
  //PeleC::h_prob_parm_device->massfrac[21] = 0.063742756504943;

  auto eos = pele::physics::PhysicsType::eos();
  eos.PYT2RE(
    PeleC::h_prob_parm_device->p, PeleC::h_prob_parm_device->massfrac.begin(), PeleC::h_prob_parm_device->T,
    PeleC::h_prob_parm_device->rho, PeleC::h_prob_parm_device->eint);
  // eos.RTY2Cs(
  //   PeleC::h_prob_parm_device->rho, PeleC::h_prob_parm_device->T, PeleC::h_prob_parm_device->massfrac.begin(),
  //   cs);
  eos.TY2Cp(PeleC::h_prob_parm_device->T, PeleC::h_prob_parm_device->massfrac.begin(), cp);

  // auto& trans_parm = PeleC::trans_parms.host_trans_parm();
  // PeleC::trans_parms.sync_to_device();

  // Load velocity fields from file. Assume data set ordered in Fortran
  // format and reshape the data accordingly. One thing to keep in mind
  // is that this contains the entire input data. We will interpolate
  // this data later to just match our box. Another assumption is that
  // the input data is a periodic cube. If the input cube is smaller
  // than our domain size, the cube will be repeated throughout the
  // domain (hence the mod operations in the interpolation).
  if (PeleC::h_prob_parm_device->restart) {
    amrex::Print() << "Skipping input file reading and assuming restart."
                   << std::endl;
  } else {
#ifdef AMREX_USE_FLOAT
    amrex::Abort("HIT cannot run in single precision at the moment.");
#else
    const size_t nx = PeleC::h_prob_parm_device->inres;
    const size_t ny = PeleC::h_prob_parm_device->inres;
    const size_t nz = PeleC::h_prob_parm_device->inres;
    amrex::Vector<amrex::Real> data(
      nx * ny * nz * 4); /* this needs to be double */
    if (PeleC::h_prob_parm_device->binfmt) {
      read_binary(PeleC::prob_parm_host->iname, nx, ny, nz, 4, data);
    } else {
      read_csv(PeleC::prob_parm_host->iname, nx, ny, nz, data);
    }

    // Extract position and velocities
    PeleC::prob_parm_host->h_xinput.resize(nx * ny * nz);
    PeleC::prob_parm_host->h_uinput.resize(nx * ny * nz);
    for (long i = 0; i < PeleC::prob_parm_host->h_xinput.size(); i++) {
      PeleC::prob_parm_host->h_xinput[i] = data[0 + i * 4] * 100.0;
      PeleC::prob_parm_host->h_uinput[i] = data[3 + i * 4];
    }

    // Get the xarray table and the differences.
    PeleC::prob_parm_host->h_xarray.resize(nx);
    for (long i = 0; i < PeleC::prob_parm_host->h_xarray.size(); i++) {
      PeleC::prob_parm_host->h_xarray[i] = PeleC::prob_parm_host->h_xinput[i];
    }
    PeleC::prob_parm_host->h_xdiff.resize(nx);
    std::adjacent_difference(
      PeleC::prob_parm_host->h_xarray.begin(),
      PeleC::prob_parm_host->h_xarray.end(),
      PeleC::prob_parm_host->h_xdiff.begin());
    PeleC::prob_parm_host->h_xdiff[0] = PeleC::prob_parm_host->h_xdiff[1];

    // Make sure the search array is increasing
    if (!std::is_sorted(
          PeleC::prob_parm_host->h_xarray.begin(),
          PeleC::prob_parm_host->h_xarray.end())) {
      amrex::Abort("Error: non ascending x-coordinate array.");
    }

    // Get pointer to the data
    PeleC::prob_parm_host->xinput.resize(
      PeleC::prob_parm_host->h_xinput.size());
    PeleC::prob_parm_host->uinput.resize(
      PeleC::prob_parm_host->h_uinput.size());
    PeleC::prob_parm_host->xarray.resize(
      PeleC::prob_parm_host->h_xarray.size());
    PeleC::prob_parm_host->xdiff.resize(PeleC::prob_parm_host->h_xdiff.size());
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, PeleC::prob_parm_host->h_xinput.begin(),
      PeleC::prob_parm_host->h_xinput.end(),
      PeleC::prob_parm_host->xinput.begin());
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, PeleC::prob_parm_host->h_uinput.begin(),
      PeleC::prob_parm_host->h_uinput.end(),
      PeleC::prob_parm_host->uinput.begin());
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, PeleC::prob_parm_host->h_xarray.begin(),
      PeleC::prob_parm_host->h_xarray.end(),
      PeleC::prob_parm_host->xarray.begin());
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, PeleC::prob_parm_host->h_xdiff.begin(),
      PeleC::prob_parm_host->h_xdiff.end(),
      PeleC::prob_parm_host->xdiff.begin());

    PeleC::h_prob_parm_device->d_xinput = PeleC::prob_parm_host->xinput.data();
    PeleC::h_prob_parm_device->d_uinput = PeleC::prob_parm_host->uinput.data();
    PeleC::h_prob_parm_device->d_xarray = PeleC::prob_parm_host->xarray.data();
    PeleC::h_prob_parm_device->d_xdiff = PeleC::prob_parm_host->xdiff.data();

    // Dimensions of the input box.
    PeleC::h_prob_parm_device->Linput =
      PeleC::prob_parm_host->h_xarray[nx - 1] +
      0.5 * PeleC::prob_parm_host->h_xdiff[nx - 1];
#endif
  }
}
}

void
PeleC::problem_post_timestep()
{
}

void
PeleC::problem_post_init()
{
}

void
PeleC::problem_post_restart()
{
}
