module hyp_advection_module 

  use amrex_ebcellflag_module, only : get_neighbor_cells
  use pelec_eb_stencil_types_module, only : eb_bndry_geom
  use riemann_util_module, only : riemann_md_singlepoint, riemann_md_vec
  use prob_params_module, only: dim

  implicit none 
  private 
  public pc_hyp_mol_flux
  contains 

  !> Computes fluxes for hyperbolic conservative update.
  !> @brief 
  !> Uses MOL formulation
  !! @param[inout] flux1  flux in X direction on X edges
  !> @param[in] q        (const)  input state, primitives
  !> @param[in] flatn    (const)  flattening parameter
  !> @param[in] src      (const)  source
  !> @param[in] nx       (const)  number of cells in X direction
  !> @param[in] ny       (const)  number of cells in Y direction
  !> @param[in] nz       (const)  number of cells in Z direction
  !> @param[in] dx       (const)  grid spacing in X direction
  !> @param[in] dy       (const)  grid spacing in Y direction
  !> @param[in] dz       (const)  grid spacing in Z direction
  !> @param[in] dt       (const)  time stepsize
  !> @param[inout] flux1    (modify) flux in X direction on X edges
  !> @param[inout] flux2    (modify) flux in Y direction on Y edges
  !> @param[inout] flux3    (modify) flux in Z direction on Z edges
  subroutine pc_hyp_mol_flux(lo, hi, &
                     domlo, domhi, &
                     q, qd_lo, qd_hi, &
                     qaux, qa_lo, qa_hi, &
                     Ax,  Axlo,  Axhi,&
                     flux1, fd1_lo, fd1_hi, &
                     Ay,  Aylo,  Ayhi,&
                     flux2, fd2_lo, fd2_hi, &
                     Az,  Azlo,  Azhi,&
                     flux3, fd3_lo, fd3_hi, &
                     flatn, fltd_lo, fltd_hi, &
                     V, Vlo, Vhi, &
                     D, Dlo, Dhi,&
                     flag, fglo, fghi, &
                     ebg, Nebg, ebflux, nebflux, &
                     bcMask, blo, bhi, &
                     h) &
                     bind(C,name="pc_hyp_mol_flux")

    use meth_params_module, only : plm_iorder, QVAR, NVAR, QPRES, QRHO, QU, QV, QW, &
                                   QFS, QC, QCSML, NQAUX, nadv, &
                                   URHO, UMX, UMY, UMZ, UEDEN, UEINT, UFS, UTEMP, UFX, UFA, &
                                   small_dens, small_pres
    use slope_module, only : slopex, slopey, slopez
    use actual_network, only : naux
    use eos_module, only : eos_rp1
    use chemistry_module, only: Ru

    implicit none

    integer, parameter  :: nspec_2=9
    integer :: vis, vie, vic ! Loop bounds for vector blocking
    integer :: vi, vii ! Loop indicies for unrolled loops over 

    integer, intent(in) ::      qd_lo(3),   qd_hi(3)
    integer, intent(in) ::      qa_lo(3),   qa_hi(3)
    integer, intent(in) ::         lo(3),      hi(3)
    integer, intent(in) ::      domlo(3),   domhi(3)
    integer, intent(in) ::       Axlo(3),    Axhi(3)
    integer, intent(in) ::     fd1_lo(3),  fd1_hi(3)
    integer, intent(in) ::       Aylo(3),    Ayhi(3)
    integer, intent(in) ::     fd2_lo(3),  fd2_hi(3)
    integer, intent(in) ::       Azlo(3),    Azhi(3)
    integer, intent(in) ::     fd3_lo(3),  fd3_hi(3)
    integer, intent(in) ::    fltd_lo(3), fltd_hi(3)
    integer, intent(in) ::        Vlo(3),     Vhi(3)
    integer, intent(in) ::        Dlo(3),     Dhi(3)
    integer, intent(in) ::        blo(3),     bhi(3)
    double precision, intent(in) :: h(3)

#ifdef PELEC_USE_EB
    integer, intent(in) ::  fglo(3),    fghi(3)
    integer, intent(in) :: flag(fglo(1):fghi(1),fglo(2):fghi(2),fglo(3):fghi(3))

    integer, intent(in) :: nebflux
    double precision, intent(inout) ::   ebflux(0:nebflux-1,1:NVAR)
    integer,            intent(in   ) :: Nebg
    type(eb_bndry_geom),intent(in   ) :: ebg(0:Nebg-1)    
    double precision :: eb_norm(3), full_area
#endif
    double precision, intent(in) ::     q(  qd_lo(1):  qd_hi(1),  qd_lo(2):  qd_hi(2),  qd_lo(3):  qd_hi(3),QVAR)  !> State
    double precision, intent(in) ::  qaux(  qa_lo(1):  qa_hi(1),  qa_lo(2):  qa_hi(2),  qa_lo(3):  qa_hi(3),NQAUX) !> Auxiliary state
    double precision, intent(in) :: flatn(fltd_lo(1):fltd_hi(1),fltd_lo(2):fltd_hi(2),fltd_lo(3):fltd_hi(3))

    double precision, intent(in   ) ::    Ax(  Axlo(1):  Axhi(1),  Axlo(2):  Axhi(2),  Axlo(3):  Axhi(3))
    double precision, intent(inout) :: flux1(fd1_lo(1):fd1_hi(1),fd1_lo(2):fd1_hi(2),fd1_lo(3):fd1_hi(3),NVAR)
    double precision, intent(in   ) ::    Ay(  Aylo(1):  Ayhi(1),  Aylo(2):  Ayhi(2),  Aylo(3):  Ayhi(3))
    double precision, intent(inout) :: flux2(fd2_lo(1):fd2_hi(1),fd2_lo(2):fd2_hi(2),fd2_lo(3):fd2_hi(3),NVAR)
    double precision, intent(in   ) ::    Az(  Azlo(1):  Azhi(1),  Azlo(2):  Azhi(2),  Azlo(3):  Azhi(3))
    double precision, intent(inout) :: flux3(fd3_lo(1):fd3_hi(1),fd3_lo(2):fd3_hi(2),fd3_lo(3):fd3_hi(3),NVAR)
    double precision, intent(inout) ::     V(   Vlo(1):   Vhi(1),   Vlo(2):   Vhi(2),   Vlo(3):   Vhi(3))
    double precision, intent(inout) ::     D(   Dlo(1):   Dhi(1),   Dlo(2):   Dhi(2),   Dlo(3):   Dhi(3),NVAR)
    integer,          intent(inout) ::bcMask(   blo(1):   bhi(1),   blo(2):   bhi(2),   blo(3):   bhi(3),3)

    integer :: i, j, k, n, nsp, L, ivar
    integer :: qt_lo(3), qt_hi(3)
    integer :: ilo1, ilo2, ihi1, ihi2, ilo3, ihi3

    ! Left and right state arrays (edge centered, cell centered)
    double precision, pointer :: dqx(:,:,:,:), dqy(:,:,:,:), dqz(:,:,:,:)

    ! Other left and right state arrays
    double precision :: qtempl(1:5+nspec_2)
    double precision :: qtempr(1:5+nspec_2)
    double precision :: rhoe_l
    double precision :: rhoe_r
    double precision :: cspeed
    double precision :: gamc_l
    double precision :: gamc_r
    double precision :: cavg
    double precision :: csmall

    ! Riemann solve work arrays
    double precision:: u_gd, v_gd, w_gd, &
         p_gd, game_gd, re_gd, &
         r_gd, ustar
    double precision :: flux_tmp(NVAR)
    integer, parameter :: idir = 1
    integer :: nextra
    integer, parameter :: coord_type = 0
    integer, parameter :: bc_test_val = 1
    
    double precision :: eos_state_rho
    double precision :: eos_state_p
    double precision :: eos_state_massfrac(nspec_2)
    double precision :: eos_state_gam1
    double precision :: eos_state_e
    double precision :: eos_state_cs

    integer, parameter :: R_RHO = 1
    integer, parameter :: R_UN  = 2
    integer, parameter :: R_UT1 = 3
    integer, parameter :: R_UT2 = 4
    integer, parameter :: R_P   = 5
    integer, parameter :: R_Y   = 6

    !   concept is to advance cells lo to hi
    !   need fluxes on the boundary
    !   if tile is eb need to expand by 2 cells in each directions
    !   would like to do this tile by tile

    nextra = 3
    ilo1=lo(1)-nextra
    ilo2=lo(2)-nextra
    ilo3=lo(3)-nextra
    ihi1=hi(1)+nextra
    ihi2=hi(2)+nextra
    ihi3=hi(3)+nextra

    !Here we have removed most of the code into a minimal example to expose how
    !the PGI 18.10 compiler gives nondeterministic results when calculating
    !flux1 here (using completely short-circuted nonphysical values, but they
    !still shouldn't be nondeterministic we feel)

    !$acc update device(nvar)
    !$acc enter data create(d) copyin(flux1,flux2,flux3) copyin(v,ax,q)

    !$acc parallel loop gang vector collapse(3) private(flux_tmp) default(present)
    do k = ilo3, ihi3
       do j = ilo2, ihi2
          do i = ilo1+1, ihi1
             do ivar = 1, NVAR
                flux_tmp(ivar) = 0.d0
             enddo
             flux_tmp(1) = q(i,j,k,1)
             do ivar = 1, NVAR
                flux1(i,j,k,ivar) = flux1(i,j,k,ivar) + flux_tmp(ivar) * ax(i,j,k)
             enddo
          enddo
       enddo
    enddo
    !$acc end parallel
 
    !$acc parallel loop gang vector collapse(4) default(present)
    do ivar=1,NVAR
       do k = ilo3+1, ihi3-1
          do j = ilo2+1, ihi2-1
             do i = ilo1+1, ihi1-1
                d(i,j,k,ivar) = - (flux1(i+1,j,k,ivar) - flux1(i,j,k,ivar) &
                                +  flux2(i,j+1,k,ivar) - flux2(i,j,k,ivar) &
                                +  flux3(i,j,k+1,ivar) - flux3(i,j,k,ivar)) / v(i,j,k)
             enddo
          enddo
       enddo
    enddo
    !$acc end parallel

    !$acc exit data copyout(flux1,d) delete(v,ax,q)

  end subroutine pc_hyp_mol_flux
end module hyp_advection_module
