#include <memory>

#include "hydro_redistribution.H"
#include "EB.H"
#include "prob.H"
#include "Utilities.H"
#include "Geometry.H"

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#endif

inline bool
PeleC::ebInitialized()
{
  return eb_initialized;
}

void
PeleC::init_eb()
{
  if (!eb_in_domain) {
    return;
  }

  // Build the geometry information; this is done for each new set of grids
  initialize_eb2_structs();

  // Set up CC signed distance container to control EB refinement
  initialize_signed_distance();
}

static
void cylinderpin_bottom_profile(std::vector<amrex::RealVect> &points) {
  amrex::RealVect p;

  p = amrex::RealVect(D_DECL(0.25, 0.1, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24975,0.0999937460888595, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2495,0.099974937185533, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24925,0.09994342998332, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.249,0.099898979485566356, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24875,0.099841229182759, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2485,0.099769696007, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24825,0.0996837484988, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.248,0.09958257569495584, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24775,0.09946514277487, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2475,0.099330127, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24725,0.099175823272, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.247,0.099, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24675,0.09879967103839, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2465,0.09857071421427, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24625,0.09830718913883, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.246,0.098, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24575,0.0976339134382, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2455,0.09717944947177, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24525,0.09656125, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.245, 0.095, 0.25));
  points.push_back(p);

}

static
void cylinderpintall_bottom_profile(std::vector<amrex::RealVect> &points) {
  amrex::RealVect p;

  p = amrex::RealVect(D_DECL(0.25, 0.35, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24975,0.3499937460888595, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2495, 0.349974937185533, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24925,0.34994342998332, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.249,0.349898979485566356, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24875,0.349841229182759, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2485,0.349769696007, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24825,0.3496837484988, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.248,0.34958257569495584, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24775,0.34946514277487, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2475,0.349330127, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24725,0.349175823272, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.247,0.349, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24675,0.34879967103839, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2465,0.34857071421427, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24625,0.34830718913883, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.246,0.348, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24575,0.3476339134382, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2455,0.34717944947177, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24525,0.34656125, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.245, 0.345, 0.25));
  points.push_back(p);

}

static
void cylinderpin_top_profile(std::vector<amrex::RealVect> &points) {
  amrex::RealVect p;

  p = amrex::RealVect(D_DECL(0.245, 0.405, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24525,0.40343875, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2455,0.40282055052823, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24575,0.4023660865618, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.246,0.402, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24625,0.40169281086117, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2465,0.40142928578573, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24675,0.40120032896161, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.247,0.401, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24725,0.400824176728, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2475,0.400669873, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24775,0.40053485722513, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.248,0.40041742430504416, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24825,0.4003162515012, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2485,0.400230303993, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24875,0.400158770817241, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.249,0.400101020514433644, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24925,0.40005657001668, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2495,0.400025062814467, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24975,0.4000062539111405, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.25, 0.4, 0.25));
  points.push_back(p);
}

static
void cylinderpintall_top_profile(std::vector<amrex::RealVect> &points) {
  amrex::RealVect p;

  p = amrex::RealVect(D_DECL(0.245, 0.655, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24525,0.65343875, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2455,0.65282055052823, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24575,0.6523660865618, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.246,0.652, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24625,0.65169281086117, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2465,0.65142928578573, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24675,0.65120032896161, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.247,0.651, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24725,0.650824176728, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2475,0.650669873, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24775,0.65053485722513, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.248,0.65041742430504416, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24825,0.6503162515012, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2485,0.650230303993, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24875,0.650158770817241, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.249,0.650101020514433644, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24925,0.65005657001668, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.2495,0.650025062814467, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.24975,0.6500062539111405, 0.25));
  points.push_back(p);
  p = amrex::RealVect(D_DECL(0.25, 0.65, 0.25));
  points.push_back(p);
}

// Set up PeleC EB Datastructures from AMReX EB2 constructs
// At the end of this routine, the following structures are populated:
//  - MultiFAB vfrac
//  - sv_eb_bndry_geom

void
PeleC::initialize_eb2_structs()
{
  BL_PROFILE("PeleC::initialize_eb2_structs()");
  amrex::Print() << "Initializing EB2 structs" << std::endl;

  static_assert(
    std::is_standard_layout<EBBndryGeom>::value,
    "EBBndryGeom is not standard layout");

  const auto& ebfactory =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(Factory());

  // These are the data sources
  amrex::MultiFab::Copy(vfrac, ebfactory.getVolFrac(), 0, 0, 1, numGrow());
  const amrex::MultiCutFab* bndrycent = &(ebfactory.getBndryCent());
  areafrac = ebfactory.getAreaFrac();
  facecent = ebfactory.getFaceCent();

  // First pass over fabs to fill sparse per cut-cell ebg structures
  sv_eb_bndry_geom.resize(vfrac.local_size());
  sv_eb_bndry_grad_stencil.resize(vfrac.local_size());
  sv_eb_flux.resize(vfrac.local_size());
  sv_eb_bcval.resize(vfrac.local_size());

  auto const& flags = ebfactory.getMultiEBCellFlagFab();

  // Boundary stencil option: 0 = original, 1 = amrex way, 2 = least squares
  amrex::ParmParse pp("ebd");

  int bgs = 0;
  pp.query("boundary_grad_stencil_type", bgs);

  if (bgs == 0) {
    amrex::Print() << "Using quadratic stencil for the EB gradient\n";
  } else if (bgs == 1) {
    amrex::Print() << "Using least-squares stencil for the EB gradient\n";
  } else {
    amrex::Print() << "Unknown or unspecified EB gradient stencil type:" << bgs
                   << std::endl;
    amrex::Abort();
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(vfrac, false); mfi.isValid(); ++mfi) {
    const amrex::Box tbox = mfi.growntilebox();
    const amrex::EBCellFlagFab& flagfab = flags[mfi];

    amrex::FabType typ = flagfab.getType(tbox);
    int iLocal = mfi.LocalIndex();

    if ((typ == amrex::FabType::regular) || (typ == amrex::FabType::covered)) {
      // do nothing
    } else if (typ == amrex::FabType::singlevalued) {
      const int Ncut = flagfab.getNumCutCells(tbox);
      sv_eb_bndry_geom[iLocal].resize(Ncut);
      auto const& flag_arr = flags.const_array(mfi);
      EBBndryGeom* d_sv_eb_bndry_geom = sv_eb_bndry_geom[iLocal].data();
      const auto lo = amrex::lbound(tbox);
      const auto hi = amrex::ubound(tbox);
      amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*dummy*/) noexcept {
        int ivec = 0;
        for (int i = lo.x; i <= hi.x; ++i) {
          for (int j = lo.y; j <= hi.y; ++j) {
            for (int k = lo.z; k <= hi.z; ++k) {
              const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
              const amrex::EBCellFlag& flag = flag_arr(iv);
              if (!(flag.isRegular() || flag.isCovered())) {
                d_sv_eb_bndry_geom[ivec].iv = iv;
                ivec++;
              }
            }
          }
        }
      });

      // int Nebg = sv_eb_bndry_geom[iLocal].size();

      // Now fill the sv_eb_bndry_geom
      auto const& vfrac_arr = vfrac.array(mfi);
      auto const& bndrycent_arr = bndrycent->array(mfi);
      AMREX_D_TERM(auto const& areafrac_arr_0 = areafrac[0]->array(mfi);
                   , auto const& areafrac_arr_1 = areafrac[1]->array(mfi);
                   , auto const& areafrac_arr_2 = areafrac[2]->array(mfi);)
      pc_fill_sv_ebg(
        tbox, Ncut, vfrac_arr, bndrycent_arr,
        AMREX_D_DECL(areafrac_arr_0, areafrac_arr_1, areafrac_arr_2),
        sv_eb_bndry_geom[iLocal].data());

      sv_eb_bndry_grad_stencil[iLocal].resize(Ncut);

      // Fill in boundary gradient for cut cells in this grown tile
      const amrex::Real dx = geom.CellSize()[0];
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
      const int sv_eb_bndry_geom_size = sv_eb_bndry_geom[iLocal].size();
      thrust::sort(
        thrust::device, sv_eb_bndry_geom[iLocal].data(),
        sv_eb_bndry_geom[iLocal].data() + sv_eb_bndry_geom_size,
        EBBndryGeomCmp());
#else
      sort<amrex::Gpu::DeviceVector<EBBndryGeom>>(sv_eb_bndry_geom[iLocal]);
#endif

      if (bgs == 0) {
        pc_fill_bndry_grad_stencil_quadratic(
          tbox, dx, Ncut, sv_eb_bndry_geom[iLocal].data(), Ncut,
          sv_eb_bndry_grad_stencil[iLocal].data());
      } else if (bgs == 1) {
        pc_fill_bndry_grad_stencil_ls(
          tbox, dx, Ncut, sv_eb_bndry_geom[iLocal].data(), Ncut,
          flags.array(mfi), sv_eb_bndry_grad_stencil[iLocal].data());
      } else {
        amrex::Print()
          << "Unknown or unspecified boundary gradient stencil type:" << bgs
          << std::endl;
        amrex::Abort();
      }

      sv_eb_flux[iLocal].define(sv_eb_bndry_grad_stencil[iLocal], NVAR);
      sv_eb_bcval[iLocal].define(sv_eb_bndry_grad_stencil[iLocal], QVAR);

      if (eb_isothermal && (diffuse_temp || diffuse_enth)) {
        sv_eb_bcval[iLocal].setVal(eb_boundary_T, QTEMP);
      }
      if (eb_noslip && diffuse_vel) {
        sv_eb_bcval[iLocal].setVal(0, QU, AMREX_SPACEDIM);
      }

    } else {
      amrex::Print() << "unknown (or multivalued) fab type" << std::endl;
      amrex::Abort();
    }
  }

  // Second pass over dirs and fabs to fill flux interpolation stencils
  amrex::Box fbox[AMREX_SPACEDIM];

  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    flux_interp_stencil[dir].resize(vfrac.local_size());

    fbox[dir] = amrex::bdryLo(
      amrex::Box(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(0, 0, 0))),
      dir, 1);

    for (int dir1 = 0; dir1 < AMREX_SPACEDIM; ++dir1) {
      if (dir1 != dir) {
        fbox[dir].grow(dir1, 1);
      }
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(vfrac, false); mfi.isValid(); ++mfi) {
      const amrex::Box tbox = mfi.growntilebox(numGrow());
      const auto& flagfab = flags[mfi];
      amrex::FabType typ = flagfab.getType(tbox);
      int iLocal = mfi.LocalIndex();

      if (typ == amrex::FabType::regular || typ == amrex::FabType::covered) {
      } else if (typ == amrex::FabType::singlevalued) {
        const auto afrac_arr = (*areafrac[dir])[mfi].array();
        const auto facecent_arr = (*facecent[dir])[mfi].array();

        // This used to be an std::set for cut_faces (it ensured
        // sorting and uniqueness)
        EBBndryGeom* d_sv_eb_bndry_geom = sv_eb_bndry_geom[iLocal].data();
        const int Nall_cut_faces = amrex::Reduce::Sum<int>(
          sv_eb_bndry_geom[iLocal].size(),
          [=] AMREX_GPU_DEVICE(int i) noexcept -> int {
            int r = 0;
            const amrex::IntVect& iv = d_sv_eb_bndry_geom[i].iv;
            for (int iside = 0; iside <= 1; iside++) {
              const amrex::IntVect iv_face = iv + iside * amrex::BASISV(dir);
              if (afrac_arr(iv_face) < 1.0) {
                r++;
              }
            }
            return r;
          });

        amrex::Gpu::DeviceVector<amrex::IntVect> v_all_cut_faces(
          Nall_cut_faces);
        amrex::IntVect* all_cut_faces = v_all_cut_faces.data();

        const int sv_eb_bndry_geom_size = sv_eb_bndry_geom[iLocal].size();
        // Serial loop on the GPU
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*dummy*/) {
          int cnt = 0;
          for (int i = 0; i < sv_eb_bndry_geom_size; i++) {
            const amrex::IntVect& iv = d_sv_eb_bndry_geom[i].iv;
            for (int iside = 0; iside <= 1; iside++) {
              const amrex::IntVect iv_face = iv + iside * amrex::BASISV(dir);
              if (afrac_arr(iv_face) < 1.0) {
                all_cut_faces[cnt] = iv_face;
                cnt++;
              }
            }
          }
        });

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
        const int v_all_cut_faces_size = v_all_cut_faces.size();
        thrust::sort(
          thrust::device, v_all_cut_faces.data(),
          v_all_cut_faces.data() + v_all_cut_faces_size);
        amrex::IntVect* unique_result_end = thrust::unique(
          thrust::device, v_all_cut_faces.data(),
          v_all_cut_faces.data() + v_all_cut_faces_size,
          thrust::equal_to<amrex::IntVect>());
        const int count_result =
          thrust::distance(v_all_cut_faces.data(), unique_result_end);
        amrex::Gpu::DeviceVector<amrex::IntVect> v_cut_faces(count_result);
        amrex::IntVect* d_all_cut_faces = v_all_cut_faces.data();
        amrex::IntVect* d_cut_faces = v_cut_faces.data();
        amrex::ParallelFor(
          v_cut_faces.size(), [=] AMREX_GPU_DEVICE(int i) noexcept {
            d_cut_faces[i] = d_all_cut_faces[i];
          });
#else
        sort<amrex::Gpu::DeviceVector<amrex::IntVect>>(v_all_cut_faces);
        amrex::Gpu::DeviceVector<amrex::IntVect> v_cut_faces =
          unique<amrex::Gpu::DeviceVector<amrex::IntVect>>(v_all_cut_faces);
#endif

        const int Nsten = v_cut_faces.size();
        if (Nsten > 0) {
          flux_interp_stencil[dir][iLocal].resize(Nsten);

          amrex::IntVect* cut_faces = v_cut_faces.data();
          auto* d_flux_interp_stencil = flux_interp_stencil[dir][iLocal].data();
          amrex::ParallelFor(
            v_cut_faces.size(), [=] AMREX_GPU_DEVICE(int i) noexcept {
              d_flux_interp_stencil[i].iv = cut_faces[i];
            });

          pc_fill_flux_interp_stencil(
            tbox, fbox[dir], Nsten, facecent_arr, afrac_arr,
            flux_interp_stencil[dir][iLocal].data());
        }
      } else {
        amrex::Abort("multi-valued flux interp stencil to be implemented");
      }
    }
  }
}

void
PeleC::define_body_state()
{
  BL_PROFILE("PeleC::define_body_state()");

  if (!eb_in_domain) {
    return;
  }

  // Scan over data and find a point in the fluid to use to
  // set computable values in cells outside the domain
  if (!body_state_set) {
    bool foundPt = false;
    const amrex::MultiFab& S = get_new_data(State_Type);
    auto const& fact =
      dynamic_cast<amrex::EBFArrayBoxFactory const&>(Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    for (amrex::MFIter mfi(S, false); mfi.isValid() && !foundPt; ++mfi) {
      const amrex::Box vbox = mfi.validbox();
      auto const& farr = S.const_array(mfi);
      auto const& flag_arr = flags.const_array(mfi);

      const auto lo = amrex::lbound(vbox);
      const auto hi = amrex::ubound(vbox);
      amrex::Gpu::DeviceVector<amrex::Real> local_body_state(NVAR, -1);
      amrex::Real* p_body_state = local_body_state.begin();
      amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*dummy*/) noexcept {
        bool found = false;
        for (int i = lo.x; i <= hi.x && !found; ++i) {
          for (int j = lo.y; j <= hi.y && !found; ++j) {
            for (int k = lo.z; k <= hi.z && !found; ++k) {
              const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
              if (flag_arr(iv).isRegular()) {
                found = true;
                for (int n = 0; n < NVAR; ++n) {
                  p_body_state[n] = farr(iv, n);
                }
              }
            }
          }
        }
      });
      amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, local_body_state.begin(),
        local_body_state.end(), body_state.begin());
      foundPt = body_state[0] != -1;
    }

    // Find proc with lowest rank to find valid point, use that for all
    amrex::Vector<int> found(amrex::ParallelDescriptor::NProcs(), 0);
    found[amrex::ParallelDescriptor::MyProc()] = (int)foundPt;
    amrex::ParallelDescriptor::ReduceIntSum(&(found[0]), found.size());
    int body_rank = -1;
    for (int i = 0; i < found.size(); ++i) {
      if (found[i] == 1) {
        body_rank = i;
      }
    }
    AMREX_ASSERT(body_rank >= 0);
    amrex::ParallelDescriptor::Bcast(
      &(body_state[0]), body_state.size(), body_rank); // NOLINT
    body_state_set = true;
  }
}

void
PeleC::set_body_state(amrex::MultiFab& S)
{
  BL_PROFILE("PeleC::set_body_state()");

  if (!eb_in_domain) {
    return;
  }

  if (!body_state_set) {
    define_body_state();
  }

  auto const& fact = dynamic_cast<amrex::EBFArrayBoxFactory const&>(Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();

  auto const& sarrs = S.arrays();
  auto const& flagarrs = flags.const_arrays();
  auto const captured_body_state = body_state;
  const amrex::IntVect ngs(0);
  amrex::ParallelFor(
    S, ngs, NVAR,
    [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) noexcept {
      pc_set_body_state(
        i, j, k, n, flagarrs[nbx], captured_body_state, sarrs[nbx]);
    });
  amrex::Gpu::synchronize();
}

void
PeleC::zero_in_body(amrex::MultiFab& S) const
{
  BL_PROFILE("PeleC::zero_in_body()");

  if (!eb_in_domain) {
    return;
  }

  amrex::GpuArray<amrex::Real, NVAR> zeros = {0.0};
  auto const& fact = dynamic_cast<amrex::EBFArrayBoxFactory const&>(Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();

  auto const& sarrs = S.arrays();
  auto const& flagarrs = flags.const_arrays();
  const amrex::IntVect ngs(0);
  amrex::ParallelFor(
    S, ngs, NVAR,
    [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k, int n) noexcept {
      pc_set_body_state(i, j, k, n, flagarrs[nbx], zeros, sarrs[nbx]);
    });
  amrex::Gpu::synchronize();
}

// Sets up implicit function using EB2 infrastructure
void
initialize_EB2(
  const amrex::Geometry& geom,
  const int eb_max_level,
  const int max_level,
  const amrex::Vector<amrex::IntVect>& ref_ratio,
  const amrex::IntVect& max_grid_size)
{
  BL_PROFILE("PeleC::initialize_EB2()");

  amrex::Print() << "Initializing EB2" << std::endl;
  amrex::ParmParse ppeb2("eb2");

  std::string geom_type("all_regular");
  ppeb2.query("geom_type", geom_type);

  int max_coarsening_level = 0;
  for (int lev = 0; lev < max_level; ++lev) {
    max_coarsening_level +=
      (ref_ratio[lev] == 2 ? 1
                           : 2); // Since EB always coarsening by factor of 2
  }

  // Custom types defined here - all_regular, plane, sphere, etc, will get
  // picked up by default (see AMReX_EB2.cpp around L100 )
  amrex::Vector<std::string> amrex_defaults(
    {"all_regular", "box", "cylinder", "plane", "sphere", "torus", "parser"});
  if (geom_type == "moving_plane") {
    amrex::RealArray point;
    point[0] = 0.5;
    point[1] = 0.0;
    point[2] = 0.0;

    amrex::RealArray normal;
    normal[0] = -1.0;
    normal[1] = 0.0;
    normal[2] = 0.0;

    amrex::EB2::PlaneIF pf(point, normal);

    // amrex::EB2::GeometryShop<amrex::EB2::PlaneIF> gshop(pf);

    amrex::EB2::BoxIF pipe(
      {AMREX_D_DECL(-1.0, 0.25, -1.)}, {AMREX_D_DECL(1.5, 0.5, 1.)}, false);
    auto gshop = amrex::EB2::makeShop(pipe);

    amrex::EB2::Build(gshop, geom, max_coarsening_level, max_coarsening_level);
  } else if (geom_type == "pins") {
      // FIXME : UPDATE MAX COARSENING LEVEL FOR THIS CASE
      //setting some constants
      //we can only do a maximum of 2 pins (change if needed)
      const int max_pin=2;
      
      //number of user defined pins
      int num_pin;
      
      amrex::ParmParse pp("pins");
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allpin(max_pin);

      //initalize pins with some dummy values
      //that fall outside of the domain
      //
      const amrex::Real *problo,*probhi;
      amrex::Real maxlen;

      problo=geom.ProbLo();
      probhi=geom.ProbHi();

      maxlen=std::max(std::max(geom.ProbLength(0),geom.ProbLength(1)),geom.ProbLength(2));

      //setting pins to be way outside the domain initially
      for(int ipin=0;ipin<max_pin;ipin++)
      {
         allpin[ipin][0] = problo[0]-100.0*maxlen;
         allpin[ipin][1] = problo[1]-100.0*maxlen;
         allpin[ipin][2] = problo[2]-100.0*maxlen;
      }
            
      //get user defined number of pins
      pp.get("num_pin", num_pin);

      amrex::Vector <std::unique_ptr<amrex::EB2::PinIF>> impfunc_pins(max_pin);
      for(int ipin = 0; ipin < num_pin; ipin++)
      {
          amrex::Array<amrex::Real,AMREX_SPACEDIM> tip{0.0,0.0,0.0};

          std::string  tipstr = "pin_" + convertIntGG(ipin) + "_tip"; 
          std::string  cstr = "pin_" + convertIntGG(ipin) + "_c"; 
          std::string  dirstr = "pin_" + convertIntGG(ipin) + "_dir"; 
          amrex::Vector<amrex::Real> vectip;
          amrex::Real  pinc;
          int  pindir;
          pp.getarr(tipstr.c_str(), vectip,  0, AMREX_SPACEDIM);
          pp.get(cstr.c_str(), pinc);
          pp.get(dirstr.c_str(), pindir);
          for(int idir = 0; idir < AMREX_SPACEDIM; idir++)
          {
              tip[idir] = vectip[idir] ;
          }
          allpin[ipin] = tip;

          impfunc_pins[ipin] = std::unique_ptr<amrex::EB2::PinIF>
                              (new amrex::EB2::PinIF(pinc, allpin[ipin], pindir, false));
      }

      auto allpin_IF = amrex::EB2::makeUnion(*impfunc_pins[0],*impfunc_pins[1]);

      auto gshop = amrex::EB2::makeShop(allpin_IF);
      amrex::EB2::Build(gshop, geom, max_level, max_level);
      // amrex::EB2::Build(gshop, geom, max_level+1, max_level+30);
  } else if (geom_type == "boxes") {
      //setting some constants
      //we can only do a maximum of 2 boxes (change if needed)
      const int max_box=2;
      
      //number of user defined boxes
      int num_box;
      
      amrex::ParmParse pp("boxes");
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allboxlo(max_box);
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allboxhi(max_box);

      //initalize boxes with some dummy values
      //that fall outside of the domain
      //
      const amrex::Real *problo,*probhi;
      amrex::Real maxlen;

      problo=geom.ProbLo();
      probhi=geom.ProbHi();

      maxlen=std::max(std::max(geom.ProbLength(0),geom.ProbLength(1)),geom.ProbLength(2));

      //setting boxes to be way outside the domain initially
      for(int ibox=0;ibox<max_box;ibox++)
      {
         allboxlo[ibox][0] = problo[0]-100.0*maxlen;
         allboxlo[ibox][1] = problo[1]-100.0*maxlen;
         allboxlo[ibox][2] = problo[2]-100.0*maxlen;

         allboxhi[ibox][0] = problo[0]-100.0*maxlen;
         allboxhi[ibox][1] = problo[1]-100.0*maxlen;
         allboxhi[ibox][2] = problo[2]-100.0*maxlen;
      }
            
      //get user defined number of boxes
      pp.get("num_boxes", num_box);

      amrex::Vector <std::unique_ptr<amrex::EB2::BoxIF>> impfunc_boxes(max_box);
      for(int ibox = 0; ibox < num_box; ibox++)
      {
          amrex::Array<amrex::Real,AMREX_SPACEDIM> boxlo{0.0,0.0,0.0};
          amrex::Array<amrex::Real,AMREX_SPACEDIM> boxhi{0.0,0.0,0.0};

          std::string  boxlostr = "box_" + convertIntGG(ibox) + "_lo"; 
          std::string  boxhistr = "box_" + convertIntGG(ibox) + "_hi"; 
          amrex::Vector<amrex::Real> vecboxlo;
          amrex::Vector<amrex::Real> vecboxhi;
          pp.getarr(boxlostr.c_str(), vecboxlo,  0, AMREX_SPACEDIM);
          pp.getarr(boxhistr.c_str(), vecboxhi,  0, AMREX_SPACEDIM);
          for(int idir = 0; idir < AMREX_SPACEDIM; idir++)
          {
              boxlo[idir] = vecboxlo[idir] ;
              boxhi[idir] = vecboxhi[idir] ;
          }
          allboxlo[ibox] = boxlo;
          allboxhi[ibox] = boxhi;

          impfunc_boxes[ibox] = std::unique_ptr<amrex::EB2::BoxIF>
                              (new amrex::EB2::BoxIF(allboxlo[ibox], allboxhi[ibox], false));
      }

      auto allbox_IF = amrex::EB2::makeUnion(*impfunc_boxes[0],*impfunc_boxes[1]);

      auto gshop = amrex::EB2::makeShop(allbox_IF);
      // amrex::EB2::Build(gshop, geom, max_level, max_level);
      amrex::EB2::Build(gshop, geom, max_level+3, max_level+30);
      // amrex::EB2::Build(gshop, geom, max_level+3, max_level+30);
  } else if (geom_type == "sandia_pins") {
      const int max_pin=2;
      
      //number of user defined pins
      int num_pin;
      
      amrex::ParmParse pp("sandia_pins");
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allbase(max_pin);
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allspherecent(max_pin);

      //initalize pins with some dummy values
      //that fall outside of the domain
      //
      const amrex::Real *problo,*probhi;
      amrex::Real maxlen;

      problo=geom.ProbLo();
      probhi=geom.ProbHi();

      maxlen=std::max(std::max(geom.ProbLength(0),geom.ProbLength(1)),geom.ProbLength(2));

      //setting pins to be way outside the domain initially
      for(int ipin=0;ipin<max_pin;ipin++)
      {
         allbase[ipin][0] = problo[0]-100.0*maxlen;
         allbase[ipin][1] = problo[1]-100.0*maxlen;
         allbase[ipin][2] = problo[2]-100.0*maxlen;
      }
            
      //get user defined number of pins
      pp.get("num_pin", num_pin);

      amrex::Vector <std::unique_ptr<amrex::EB2::TruncatedConeIF>> impfunc_cones(max_pin);
      amrex::Vector <std::unique_ptr<amrex::EB2::SphereIF>> impfunc_spheres(max_pin);
      for(int ipin = 0; ipin < num_pin; ipin++)
      {
          amrex::Array<amrex::Real,AMREX_SPACEDIM> base{0.0,0.0,0.0};
          amrex::Array<amrex::Real,AMREX_SPACEDIM> sphere_cent{0.0,0.0,0.0};

          std::string  basestr = "pin_" + convertIntGG(ipin) + "_base"; 
          std::string  r1str = "pin_" + convertIntGG(ipin) + "_radius1"; 
          std::string  r2str = "pin_" + convertIntGG(ipin) + "_radius2"; 
          std::string  hstr = "pin_" + convertIntGG(ipin) + "_height"; 
          std::string  dirstr = "pin_" + convertIntGG(ipin) + "_dir"; 
          amrex::Vector<amrex::Real> vecbase;
          amrex::Real  pinr1;
          amrex::Real  pinr2;
          amrex::Real  pinh;
          int  pindir;
          pp.getarr(basestr.c_str(), vecbase,  0, AMREX_SPACEDIM);
          pp.get(r1str.c_str(), pinr1);
          pp.get(r2str.c_str(), pinr2);
          pp.get(hstr.c_str(), pinh);
          pp.get(dirstr.c_str(), pindir);
          for(int idir = 0; idir < AMREX_SPACEDIM; idir++)
          {
              base[idir] = vecbase[idir] ;
              sphere_cent[idir] = vecbase[idir];
              if(idir == pindir) sphere_cent[idir] += pinh;
          }
          allbase[ipin] = base;
          allspherecent[ipin] = sphere_cent;

          impfunc_cones[ipin] = std::unique_ptr<amrex::EB2::TruncatedConeIF>
                              (new amrex::EB2::TruncatedConeIF(pinr1, pinr2, pinh, allbase[ipin], pindir, false));
          impfunc_spheres[ipin] = std::unique_ptr<amrex::EB2::SphereIF>
                              (new amrex::EB2::SphereIF(pinr2, allspherecent[ipin], false));
      }

      auto allpin_IF = amrex::EB2::makeUnion(*impfunc_cones[0],*impfunc_cones[1],*impfunc_spheres[0],*impfunc_spheres[1]);

      auto gshop = amrex::EB2::makeShop(allpin_IF);
      amrex::EB2::Build(gshop, geom, max_level, max_level);
      // amrex::EB2::Build(gshop, geom, max_level+2, max_level+30);
  } else if (geom_type == "cylinder_pins") {
      const int max_pin=2;
      
      //number of user defined pins
      int num_pin;
      
      amrex::ParmParse pp("cylinder_pins");
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allcylcent(max_pin);
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allspherecent(max_pin);

      //initalize pins with some dummy values
      //that fall outside of the domain
      //
      const amrex::Real *problo,*probhi;
      amrex::Real maxlen;

      problo=geom.ProbLo();
      probhi=geom.ProbHi();

      maxlen=std::max(std::max(geom.ProbLength(0),geom.ProbLength(1)),geom.ProbLength(2));

      //setting pins to be way outside the domain initially
      for(int ipin=0;ipin<max_pin;ipin++)
      {
         allcylcent[ipin][0] = problo[0]-100.0*maxlen;
         allcylcent[ipin][1] = problo[1]-100.0*maxlen;
         allcylcent[ipin][2] = problo[2]-100.0*maxlen;
      }
            
      //get user defined number of pins
      pp.get("num_pin", num_pin);

      amrex::Vector <std::unique_ptr<amrex::EB2::CylinderIF>> impfunc_cylinders(max_pin);
      amrex::Vector <std::unique_ptr<amrex::EB2::SphereIF>> impfunc_spheres(max_pin);
      for(int ipin = 0; ipin < num_pin; ipin++)
      {
          amrex::Array<amrex::Real,AMREX_SPACEDIM> cyl_cent{0.0,0.0,0.0};
          amrex::Array<amrex::Real,AMREX_SPACEDIM> sphere_cent{0.0,0.0,0.0};

          std::string  centstr = "pin_" + convertIntGG(ipin) + "_center"; 
          std::string  rstr = "pin_" + convertIntGG(ipin) + "_radius"; 
          std::string  hstr = "pin_" + convertIntGG(ipin) + "_height"; 
          std::string  dirstr = "pin_" + convertIntGG(ipin) + "_dir"; 
          std::string  signstr = "pin_" + convertIntGG(ipin) + "_sign"; 
          amrex::Vector<amrex::Real> veccent;
          amrex::Real  pinr;
          amrex::Real  pinh;
          int  pindir;
          int  pinsign;
          pp.getarr(centstr.c_str(), veccent,  0, AMREX_SPACEDIM);
          pp.get(rstr.c_str(), pinr);
          pp.get(hstr.c_str(), pinh);
          pp.get(dirstr.c_str(), pindir);
          pp.get(signstr.c_str(), pinsign);
          for(int idir = 0; idir < AMREX_SPACEDIM; idir++)
          {
              cyl_cent[idir] = veccent[idir] ;
              sphere_cent[idir] = veccent[idir];
              if(idir == pindir) sphere_cent[idir] += pinsign * pinh / 2.0;
          }
          allcylcent[ipin] = cyl_cent;
          allspherecent[ipin] = sphere_cent;

          impfunc_cylinders[ipin] = std::unique_ptr<amrex::EB2::CylinderIF>
                              (new amrex::EB2::CylinderIF(pinr, pinh, pindir, allcylcent[ipin], false));
          impfunc_spheres[ipin] = std::unique_ptr<amrex::EB2::SphereIF>
                              (new amrex::EB2::SphereIF(pinr, allspherecent[ipin], false));
      }

      auto allpin_IF = amrex::EB2::makeUnion(*impfunc_cylinders[0],*impfunc_cylinders[1],*impfunc_spheres[0],*impfunc_spheres[1]);

      auto gshop = amrex::EB2::makeShop(allpin_IF);
      amrex::EB2::Build(gshop, geom, max_level, max_level);
      // amrex::EB2::Build(gshop, geom, 1, 1);
      // amrex::EB2::Build(gshop, geom, max_level+1, max_level+30);
  } else if (geom_type == "cylinder_pins_spline") {
    amrex::EB2::SplineIF bottomPin;
    amrex::EB2::SplineIF topPin;
    amrex::RealVect p;

    // Create bottom pin
    std::vector<amrex::RealVect> bsplpts;
    cylinderpin_bottom_profile(bsplpts);
    bottomPin.addSplineElement(bsplpts);

    std::vector<amrex::RealVect> blnpts;

    p = amrex::RealVect(D_DECL(0.245, 0.095, 0.25));
    blnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, 0.05, 0.25));
    blnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, -0.01, 0.25));
    blnpts.push_back(p);
    bottomPin.addLineElement(blnpts);
    blnpts.clear();

    std::vector<amrex::RealVect> tlnpts;

    p = amrex::RealVect(D_DECL(0.245, 5.05, 0.25));
    tlnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, 0.455, 0.25));
    tlnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, 0.405, 0.25));
    tlnpts.push_back(p);
    topPin.addLineElement(tlnpts);
    tlnpts.clear();

    // // Create top pins
    std::vector<amrex::RealVect> tsplpts;
    cylinderpin_top_profile(tsplpts);
    topPin.addSplineElement(tsplpts);

    // amrex::RealArray translateArray;
    // translateArray[0] = 0.0;
    // translateArray[1] = 0.3;
    // translateArray[2] = 0.0;


    auto revolveBottomPin  = amrex::EB2::lathe(bottomPin);
    auto revolveTopPin  = amrex::EB2::lathe(topPin);
    // auto rotatePin = amrex::EB2::rotate(revolveBottomPin, constants::PI(), 1);
    // auto translatePin = amrex::EB2::translate(revolveBottomPin, translateArray);
  

    // auto revolveTopPin  = amrex::EB2::lathe(topPin);
    auto TopBottomPins = amrex::EB2::makeUnion(revolveBottomPin, revolveTopPin);
    // auto TopBottomPins = amrex::EB2::makeUnion(revolveBottomPin);
    auto gshop = amrex::EB2::makeShop(TopBottomPins);
    amrex::EB2::Build(gshop, geom, max_level, max_level);
  } else if (geom_type == "cylinder_pins_tall_spline") {
    amrex::EB2::SplineIF bottomPin;
    amrex::EB2::SplineIF topPin;
    amrex::RealVect p;

    // Create bottom pin
    std::vector<amrex::RealVect> bsplpts;
    cylinderpintall_bottom_profile(bsplpts);
    bottomPin.addSplineElement(bsplpts);

    std::vector<amrex::RealVect> blnpts;

    p = amrex::RealVect(D_DECL(0.245, 0.345, 0.25));
    blnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, 0.15, 0.25));
    blnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, -0.1, 0.25));
    blnpts.push_back(p);
    bottomPin.addLineElement(blnpts);
    blnpts.clear();

    std::vector<amrex::RealVect> tlnpts;

    p = amrex::RealVect(D_DECL(0.245, 1.1, 0.25));
    tlnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, 0.85, 0.25));
    tlnpts.push_back(p);
    p = amrex::RealVect(D_DECL(0.245, 0.655, 0.25));
    tlnpts.push_back(p);
    topPin.addLineElement(tlnpts);
    tlnpts.clear();

    // // Create top pins
    std::vector<amrex::RealVect> tsplpts;
    cylinderpintall_top_profile(tsplpts);
    topPin.addSplineElement(tsplpts);

    // amrex::RealArray translateArray;
    // translateArray[0] = 0.0;
    // translateArray[1] = 0.3;
    // translateArray[2] = 0.0;


    auto revolveBottomPin  = amrex::EB2::lathe(bottomPin);
    auto revolveTopPin  = amrex::EB2::lathe(topPin);
    // auto rotatePin = amrex::EB2::rotate(revolveBottomPin, constants::PI(), 1);
    // auto translatePin = amrex::EB2::translate(revolveBottomPin, translateArray);
  

    // auto revolveTopPin  = amrex::EB2::lathe(topPin);
    auto TopBottomPins = amrex::EB2::makeUnion(revolveBottomPin, revolveTopPin);
    // auto TopBottomPins = amrex::EB2::makeUnion(revolveBottomPin);
    auto gshop = amrex::EB2::makeShop(TopBottomPins);
    amrex::EB2::Build(gshop, geom, max_level, max_level);
  } else if (geom_type == "rboxes") {
      //setting some constants
      //we can only do a maximum of 2 boxes (change if needed)
      const int max_box=2;
      
      //number of user defined boxes
      int num_box;
      amrex::Real box_rot_angle = 0.0;
      
      amrex::ParmParse pp("rboxes");
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allboxlo(max_box);
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allboxhi(max_box);

      //initalize boxes with some dummy values
      //that fall outside of the domain
      //
      const amrex::Real *problo,*probhi;
      amrex::Real maxlen;

      problo=geom.ProbLo();
      probhi=geom.ProbHi();

      maxlen=std::max(std::max(geom.ProbLength(0),geom.ProbLength(1)),geom.ProbLength(2));

      //setting boxes to be way outside the domain initially
      for(int ibox=0;ibox<max_box;ibox++)
      {
         allboxlo[ibox][0] = problo[0]-100.0*maxlen;
         allboxlo[ibox][1] = problo[1]-100.0*maxlen;
         allboxlo[ibox][2] = problo[2]-100.0*maxlen;

         allboxhi[ibox][0] = problo[0]-100.0*maxlen;
         allboxhi[ibox][1] = problo[1]-100.0*maxlen;
         allboxhi[ibox][2] = problo[2]-100.0*maxlen;
      }
            
      //get user defined number of boxes
      pp.get("num_boxes", num_box);
      pp.get("box_rot_angle", box_rot_angle);

      amrex::Vector <std::unique_ptr<amrex::EB2::RotationIF<amrex::EB2::BoxIF>>> impfunc_boxes(max_box);
      for(int ibox = 0; ibox < num_box; ibox++)
      {
          amrex::Array<amrex::Real,AMREX_SPACEDIM> boxlo{0.0,0.0,0.0};
          amrex::Array<amrex::Real,AMREX_SPACEDIM> boxhi{0.0,0.0,0.0};

          std::string  boxlostr = "box_" + convertIntGG(ibox) + "_lo"; 
          std::string  boxhistr = "box_" + convertIntGG(ibox) + "_hi"; 
          amrex::Vector<amrex::Real> vecboxlo;
          amrex::Vector<amrex::Real> vecboxhi;
          pp.getarr(boxlostr.c_str(), vecboxlo,  0, AMREX_SPACEDIM);
          pp.getarr(boxhistr.c_str(), vecboxhi,  0, AMREX_SPACEDIM);
          for(int idir = 0; idir < AMREX_SPACEDIM; idir++)
          {
              boxlo[idir] = vecboxlo[idir] ;
              boxhi[idir] = vecboxhi[idir] ;
          }
          allboxlo[ibox] = boxlo;
          allboxhi[ibox] = boxhi;

          amrex::EB2::BoxIF box0(allboxlo[ibox], allboxhi[ibox], false);

          impfunc_boxes[ibox] = std::make_unique<amrex::EB2::RotationIF<amrex::EB2::BoxIF>> (box0, box_rot_angle, 0);
      }

      auto allbox_IF = amrex::EB2::makeUnion(*impfunc_boxes[0],*impfunc_boxes[1]);

      auto gshop = amrex::EB2::makeShop(allbox_IF);
      amrex::EB2::Build(gshop, geom, max_level, max_level);
  } else if (geom_type == "spheres") {
      //setting some constants
      //we can only do a maximum of 2 spheres (change if needed)
      const int max_sphere=2;
      
      //number of user defined spheres
      int num_sphere;
      
      amrex::ParmParse pp("spheres");
      amrex::Vector<amrex::Array<amrex::Real,AMREX_SPACEDIM>> allspherecent(max_sphere);

      //initalize spheres with some dummy values
      //that fall outside of the domain
      //
      const amrex::Real *problo,*probhi;
      amrex::Real maxlen;

      problo=geom.ProbLo();
      probhi=geom.ProbHi();

      maxlen=std::max(std::max(geom.ProbLength(0),geom.ProbLength(1)),geom.ProbLength(2));

      //setting spheres to be way outside the domain initially
      for(int isphere=0;isphere<max_sphere;isphere++)
      {
         allspherecent[isphere][0] = problo[0]-100.0*maxlen;
         allspherecent[isphere][1] = problo[1]-100.0*maxlen;
         allspherecent[isphere][2] = problo[2]-100.0*maxlen;
      }
            
      //get user defined number of spheres
      pp.get("num_spheres", num_sphere);

      amrex::Vector <std::unique_ptr<amrex::EB2::SphereIF>> impfunc_spheres(max_sphere);
      for(int isphere = 0; isphere < num_sphere; isphere++)
      {
          amrex::Array<amrex::Real,AMREX_SPACEDIM> spherecent{0.0,0.0,0.0};
          amrex::Real  sphereR;
          bool  fluidInside;

          std::string  spherecentstr = "sphere_" + convertIntGG(isphere) + "_cent"; 
          std::string  rstr = "sphere_" + convertIntGG(isphere) + "_r"; 
          std::string  insidestr = "sphere_" + convertIntGG(isphere) + "_inside"; 
          amrex::Vector<amrex::Real> vecspherecent;
          pp.getarr(spherecentstr.c_str(), vecspherecent,  0, AMREX_SPACEDIM);
          pp.get(rstr.c_str(), sphereR);
          pp.get(insidestr.c_str(), fluidInside);
          for(int idir = 0; idir < AMREX_SPACEDIM; idir++)
          {
              spherecent[idir] = vecspherecent[idir] ;
          }
          allspherecent[isphere] = spherecent;

          impfunc_spheres[isphere] = std::unique_ptr<amrex::EB2::SphereIF>
                              (new amrex::EB2::SphereIF(sphereR, allspherecent[isphere], fluidInside));
      }

      auto allsphere_IF = amrex::EB2::makeUnion(*impfunc_spheres[0],*impfunc_spheres[1]);

      auto gshop = amrex::EB2::makeShop(allsphere_IF);
      amrex::EB2::Build(gshop, geom, max_level, max_level);
  } else if (geom_type == "quarter-circle") {

    amrex::Real r_inner = 1.0;
    amrex::Real r_outer = 2.0;
    ppeb2.query("r_inner", r_inner);
    ppeb2.query("r_outer", r_outer);

    amrex::EB2::CylinderIF inner(
      r_inner, 10, 2, {AMREX_D_DECL(0, 0, 0)}, false);
    amrex::EB2::CylinderIF outer(r_outer, 10, 2, {AMREX_D_DECL(0, 0, 0)}, true);

    auto polys = amrex::EB2::makeUnion(inner, outer);
    auto gshop = amrex::EB2::makeShop(polys);
    amrex::EB2::Build(
      gshop, geom, max_coarsening_level, max_coarsening_level, 4, false);
  } else if (geom_type == "converging-nozzle") {
#ifdef ConvergingNozzle
    EBConvergingNozzle(geom, max_level);
#else
    amrex::Abort("converging-nozzle geom_type not supported");
#endif
  } else if (!(std::find(amrex_defaults.begin(), amrex_defaults.end(), geom_type) !=
        amrex_defaults.end())) {
    std::unique_ptr<pele::pelec::Geometry> geometry(
      pele::pelec::Geometry::create(geom_type));
    geometry->build(geom, max_coarsening_level);
  } else {
    amrex::EB2::Build(geom, max_level, max_level);
  }

  // Add finer level, might be inconsistent with the coarser level created
  // above.
  if (geom_type != "chkfile") {
    amrex::EB2::addFineLevels(max_level - eb_max_level);
  }

  bool write_chk_geom = false;
  ppeb2.query("write_chk_geom", write_chk_geom);
  if (write_chk_geom) {
    const auto& is = amrex::EB2::IndexSpace::top();
    const auto& eb_level = is.getLevel(geom);
    std::string chkfile = "chk_geom";
    ppeb2.query("chkfile", chkfile);

    eb_level.write_to_chkpt_file(
      chkfile, amrex::EB2::ExtendDomainFace(), max_grid_size[0]);
  }
}

void
PeleC::initialize_signed_distance()
{
  BL_PROFILE("PeleC::initialize_signed_distance()");
  if (level == 0) {
    const auto& ebfactory =
      dynamic_cast<amrex::EBFArrayBoxFactory const&>(Factory());
    signed_dist_0.define(grids, dmap, 1, 1, amrex::MFInfo(), ebfactory);

    // Estimate the maximum distance we need in terms of level 0 dx:
    auto extentFactor = static_cast<amrex::Real>(parent->nErrorBuf(0));
    for (int ilev = 1; ilev <= parent->maxLevel(); ++ilev) {
      extentFactor += static_cast<amrex::Real>(parent->nErrorBuf(ilev)) /
                      std::pow(
                        static_cast<amrex::Real>(parent->refRatio(ilev - 1)[0]),
                        static_cast<amrex::Real>(ilev));
    }
    extentFactor *= tagging_parm->detag_eb_factor;

    amrex::MultiFab signDist(
      convert(grids, amrex::IntVect::TheUnitVector()), dmap, 1, 1,
      amrex::MFInfo(), ebfactory);
    amrex::FillSignedDistance(signDist, true);

    const auto& sd_ccs = signed_dist_0.arrays();
    const auto& sd_nds = signDist.const_arrays();
    const amrex::IntVect ngs(signed_dist_0.nGrow());
    amrex::ParallelFor(
      signed_dist_0, ngs,
      [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
        const auto& sd_cc = sd_ccs[nbx];
        const auto& sd_nd = sd_nds[nbx];
        const amrex::Real fac = AMREX_D_PICK(0.5, 0.25, 0.125);
        sd_cc(i, j, k) = AMREX_D_TERM(
          sd_nd(i, j, k) + sd_nd(i + 1, j, k),
          +sd_nd(i, j + 1, k) + sd_nd(i + 1, j + 1, k),
          +sd_nd(i, j, k + 1) + sd_nd(i + 1, j, k + 1) +
            sd_nd(i, j + 1, k + 1) + sd_nd(i + 1, j + 1, k + 1));
        sd_cc(i, j, k) *= fac;
      });
    amrex::Gpu::synchronize();

    signed_dist_0.FillBoundary(parent->Geom(0).periodicity());
    extend_signed_distance(&signed_dist_0, extentFactor);
  }
}

void
PeleC::eb_distance(const int lev, amrex::MultiFab& signDistLev)
{
  BL_PROFILE("PeleC::eb_distance()");
  if (lev == 0) {
    amrex::MultiFab::Copy(signDistLev, signed_dist_0, 0, 0, 1, 0);
    return;
  }

  // A pair of MF to hold crse & fine dist data
  amrex::Array<std::unique_ptr<amrex::MultiFab>, 2> MFpair;

  // dummy bcs
  amrex::Vector<amrex::BCRec> bcrec_dummy(1);
  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    bcrec_dummy[0].setLo(dir, INT_DIR);
    bcrec_dummy[0].setHi(dir, INT_DIR);
  }

  // Interpolate on successive levels up to lev
  const auto& pc_0lev = getLevel(0);
  for (int ilev = 1; ilev <= lev; ++ilev) {

    // Use MF EB interp
    auto& interpolater = amrex::eb_mf_lincc_interp;

    // Get signDist on coarsen fineBA
    const auto& pc_ilev = getLevel(ilev);
    const auto& grids_ilev = pc_ilev.grids;
    const auto& dmap_ilev = pc_ilev.DistributionMap();
    amrex::BoxArray coarsenBA(grids_ilev.size());
    for (int j = 0, N = coarsenBA.size(); j < N; ++j) {
      coarsenBA.set(
        j, interpolater.CoarseBox(grids_ilev[j], parent->refRatio(ilev - 1)));
    }
    amrex::MultiFab coarsenSignDist(coarsenBA, dmap_ilev, 1, 0);
    coarsenSignDist.setVal(0.0);
    const amrex::MultiFab* crseSignDist =
      (ilev == 1) ? &pc_0lev.signed_dist_0 : MFpair[0].get();
    coarsenSignDist.ParallelCopy(*crseSignDist, 0, 0, 1);

    // Interpolate on current ilev
    amrex::MultiFab* currentSignDist;
    if (ilev < lev) {
      const auto& ebfactory =
        dynamic_cast<amrex::EBFArrayBoxFactory const&>(pc_ilev.Factory());
      MFpair[1] = std::make_unique<amrex::MultiFab>(
        grids_ilev, dmap_ilev, 1, 0, amrex::MFInfo(), ebfactory);
    }
    currentSignDist = (ilev == lev) ? &signDistLev : MFpair[1].get();

    interpolater.interp(
      coarsenSignDist, 0, *currentSignDist, 0, 1, amrex::IntVect(0),
      parent->Geom(ilev - 1), parent->Geom(ilev), parent->Geom(ilev).Domain(),
      parent->refRatio(ilev - 1), {bcrec_dummy}, 0);

    // Swap MFpair
    if (ilev < lev) {
      std::swap(MFpair[0], MFpair[1]);
    }
  }
}

// Extend the cell-centered based signed distance function
void
PeleC::extend_signed_distance(
  amrex::MultiFab* signDist, amrex::Real extendFactor)
{
  // This is a not-so-pretty piece of code that'll take AMReX cell-averaged
  // signed distance and propagates it manually up to the point where we need to
  // have it for derefining.
  BL_PROFILE("PeleC::extend_signed_distance()");
  const auto geomdata = parent->Geom(0).data();
  amrex::Real maxSignedDist = signDist->max(0);
  const auto& ebfactory =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(signDist->Factory());
  const auto& flags = ebfactory.getMultiEBCellFlagFab();
  int nGrowFac = flags.nGrow() + 1;

  // First set the region far away at the max value we need
  auto const& sd_ccs = signDist->arrays();
  const amrex::IntVect ngs(signDist->nGrow());
  amrex::ParallelFor(
    *signDist, ngs,
    [=] AMREX_GPU_DEVICE(int nbx, int i, int j, int k) noexcept {
      const auto& sd_cc = sd_ccs[nbx];
      if (sd_cc(i, j, k) >= maxSignedDist - 1e-12) {
        const amrex::Real* dx = geomdata.CellSize();
        sd_cc(i, j, k) = nGrowFac * dx[0] * extendFactor;
      }
    });
  amrex::Gpu::synchronize();

  // Iteratively compute the distance function in boxes, propagating accross
  // boxes using ghost cells If needed, increase the number of loop to extend
  // the reach of the distance function
  const int nMaxLoop = 4;
  for (int dloop = 1; dloop <= nMaxLoop; dloop++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*signDist, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const amrex::Box& gbx = grow(bx, 1);
      if (flags[mfi].getType(gbx) == amrex::FabType::covered) {
        continue;
      }
      auto const& sd_cc = signDist->array(mfi);
      ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const auto glo = amrex::lbound(gbx);
        const auto ghi = amrex::ubound(gbx);
        const amrex::Real* dx = geomdata.CellSize();
        const amrex::Real extendedDist = dx[0] * extendFactor;
        if (sd_cc(i, j, k) >= maxSignedDist - 1e-12) {
          amrex::Real closestEBDist = 1e12;
          for (int kk = glo.z; kk <= ghi.z; ++kk) {
            for (int jj = glo.y; jj <= ghi.y; ++jj) {
              for (int ii = glo.x; ii <= ghi.x; ++ii) {
                if ((i != ii) || (j != jj) || (k != kk)) {
                  if (sd_cc(ii, jj, kk) > 0.0) {
                    const amrex::Real distToCell = std::sqrt(AMREX_D_TERM(
                      ((i - ii) * dx[0] * (i - ii) * dx[0]),
                      +((j - jj) * dx[1] * (j - jj) * dx[1]),
                      +((k - kk) * dx[2] * (k - kk) * dx[2])));
                    const amrex::Real distToEB = distToCell + sd_cc(ii, jj, kk);
                    if (distToEB < closestEBDist) {
                      closestEBDist = distToEB;
                    }
                  }
                }
              }
            }
          }
          if (closestEBDist < 1e10) {
            sd_cc(i, j, k) = closestEBDist;
          } else {
            sd_cc(i, j, k) = extendedDist;
          }
        }
      });
    }
    signDist->FillBoundary(parent->Geom(0).periodicity());
  }
}

void
PeleC::InitialRedistribution(
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec> bcs,
  amrex::MultiFab& S_new)
{
  BL_PROFILE("PeleC::InitialRedistribution()");

  // Don't redistribute if there is no EB or if the redistribution type is
  // anything other than StateRedist
  if ((!eb_in_domain) || (redistribution_type != "StateRedist")) {
    return;
  }

  if (verbose != 0) {
    amrex::Print() << "Doing initial redistribution... " << std::endl;
  }

  // Initial data are set at new time step
  amrex::MultiFab tmp(
    grids, dmap, S_new.nComp(), numGrow(), amrex::MFInfo(), Factory());

  amrex::MultiFab::Copy(tmp, S_new, 0, 0, S_new.nComp(), S_new.nGrow());
  FillPatch(*this, tmp, numGrow(), time, State_Type, 0, S_new.nComp());
  EB_set_covered(tmp, 0.0);

  amrex::Gpu::DeviceVector<amrex::BCRec> d_bcs(bcs.size());
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, bcs.begin(), bcs.end(), d_bcs.begin());

  for (amrex::MFIter mfi(S_new, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.validbox();

    auto const& fact =
      dynamic_cast<amrex::EBFArrayBoxFactory const&>(S_new.Factory());

    auto const& flags = fact.getMultiEBCellFlagFab()[mfi];
    amrex::Array4<const amrex::EBCellFlag> const& flag_arr =
      flags.const_array();

    if (
      (flags.getType(amrex::grow(bx, 1)) != amrex::FabType::covered) &&
      (flags.getType(amrex::grow(bx, 1)) != amrex::FabType::regular)) {
      amrex::Array4<const amrex::Real> AMREX_D_DECL(fcx, fcy, fcz), ccc,
        AMREX_D_DECL(apx, apy, apz);

      AMREX_D_TERM(fcx = facecent[0]->const_array(mfi);
                   , fcy = facecent[1]->const_array(mfi);
                   , fcz = facecent[2]->const_array(mfi););

      ccc = fact.getCentroid().const_array(mfi);

      AMREX_D_TERM(apx = areafrac[0]->const_array(mfi);
                   , apy = areafrac[1]->const_array(mfi);
                   , apz = areafrac[2]->const_array(mfi););

      const auto& sarr = S_new.array(mfi);
      const auto& tarr = tmp.array(mfi);
      Redistribution::ApplyToInitialData(
        bx, NVAR, sarr, tarr, flag_arr, AMREX_D_DECL(apx, apy, apz),
        vfrac.const_array(mfi), AMREX_D_DECL(fcx, fcy, fcz), ccc,
        d_bcs.dataPtr(), geom, redistribution_type, eb_srd_max_order);

      // Make sure rho is same as sum rhoY after redistribution
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          amrex::Real drhoYsum = 0.0;
          for (int n = 0; n < NUM_SPECIES; n++) {
            drhoYsum += sarr(i, j, k, UFS + n) - tarr(i, j, k, UFS + n);
          }
          sarr(i, j, k, URHO) = tarr(i, j, k, URHO) + drhoYsum;
        });
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          pc_check_initial_species(i, j, k, sarr);
        });
    }
  }
  set_body_state(S_new);
}
