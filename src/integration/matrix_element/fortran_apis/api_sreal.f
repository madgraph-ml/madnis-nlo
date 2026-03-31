      subroutine init_api(n_ext, alpha_s)

      implicit none
      integer n_ext
      real*8 alpha_s
      double precision c(0:1),gamma(0:1),gammap(0:1),gamma_ph,gammap_ph
      common/fks_colors/c,gamma,gammap,gamma_ph,gammap_ph
      double precision c_used, gamma_used, gammap_used
      double precision xicut_used,eikIreg
      
      common /cxicut_used/xicut_used
      include 'nexternal.inc'
      include 'q_es.inc'
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      include 'coupl.inc'
      include 'fks_powers.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      LOGICAL UPDATELOOP
      COMMON /TO_UPDATELOOP/UPDATELOOP
      double precision Pi, Zeta3, CA, CF, TF
      
      parameter (CA=3.d0)
      parameter (Pi=3.1415926535897932385d0)
      parameter (CF=4/3.d0)
      parameter (TF=1/2.d0)

      
      call setpara('param_card.dat')
      UPDATELOOP = .true.
      QES2 = 8302.4899239999995d0
      mu_r = sqrt(8302.4899239999995d0)
      G = 1.2229388294466237d0
      call update_as_param()
      UPDATELOOP = .false.
      n_ext = nexternal
      
      alpha_s = G**2/(4.D0*3.1415926535897931D0)
C      xicut_used = xicut
      shat = 1000000.0d0
      sqrtshat = 1000.0d0
      c(0)=CA
      c(1)=CF
      gamma(0)=( 11d0*CA-2d0*Nf )/6d0
      gamma(1)=CF*3d0/2d0
      gammap(0)=( 67d0/9d0 - 2d0*PI**2/3d0 )*CA - 23d0/18d0*Nf
      gammap(1)=( 13/2d0 - 2d0*PI**2/3d0 )*CF

      end subroutine

      subroutine call_sreal_me(count, pp, p_born_sampl, p_soft, p_coll,p_soft_counters, fks_sectors,
     # xi_i_fks_inp,y_ij_fks_inp, deltac, xic, me, me_s, me_co, me_sc)
      use FKSParams
      implicit none
      integer n_ext
      include 'nexternal.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      LOGICAL KEEP_ORDER(NSQAMPSO), FIRSTTIME
      INCLUDE 'orders.inc'
      DATA KEEP_ORDER / NSQAMPSO*.TRUE. /
      DATA FIRSTTIME / .TRUE. /
      real*8 pp(0:3, 1:nexternal, 0:count-1), p_born_sampl(0:3, 1:nexternal-1, 0:count-1), p_soft(0:3, 1:nexternal, 0:count-1)
      real*8 p_soft_counters(0:3, 0:count-1)
      real *8 p_coll(0:3, 1:nexternal, 0:count-1)
      real*8 me(0:count-1), me_s(0:count-1), me_co(0:count-1), me_sc(0:count-1)
      INTEGER fks_sectors(0:count-1)
      real*8 xi_i_fks_inp(0:count-1)
      real*8 y_ij_fks_inp(0:count-1)
      double precision deltac, xic
      integer position, position_ifks
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      double precision xi_i_fks, y_ij_fks, nrma, nrmb, sclr
      integer count, medim, precdim
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      logical split_type_used(1:2)
      common/to_split_type_used/split_type_used
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      double precision p_born(0:3,nexternal-1)
      common/pborn/    p_born
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision s_s,fks_Sij
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      double precision p_born_coll(0:3,nexternal-1)
      common/pborn_coll/p_born_coll
      double precision dummy_wgt

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision iden_comp
      common /c_iden_comp/iden_comp
      double precision    delta_used
      common /cdelta_used/delta_used
      double precision    xicut_used
      common /cxicut_used/xicut_used
      double precision rescaler
      double precision p_reco(0:3)
      double precision phi_mom, phi_daught
      double complex xij_aor
      common/cxij_aor/xij_aor
      double complex ximag
      parameter (ximag=(0d0,1d0))
      logical firsttime_pdf
      data firsttime_pdf /.true./


      double precision th_m,cth_m,sth_m,phi_m,cphi_m,sphi_m
      double precision th_d,cth_d,sth_d,phi_d,cphi_d,sphi_d
      double precision qin(1:3), support(0:3)
      double precision phi_mother_fks

      include 'fks_info.inc'
      include  'fks_powers.inc'

      
      integer sym
      external fks_Sij


      
      n_ext = nexternal
      shat = 1000000.0d0
      sqrtshat = 1000.0d0

      call FKSParamReader('FKS_params.dat',.TRUE.,.FALSE.)
      do k=0,count-1
        CALCULATEDBORN = .FALSE.
        NFKSPROCESS = fks_sectors(k)
        call fks_inc_chooser
        call leshouche_inc_chooser
        call setfksfactor(.false.)

        delta_used = deltac
        xicut_used = xic

        p_born = p_born_sampl(:,:,k)
        me(k) = 0.0d0
        xi_i_fks = xi_i_fks_inp(k)
        position = FKS_J_D(fks_sectors(k))
        position_ifks = FKS_I_D(fks_sectors(k))
        p_i_fks_cnt(:,0) = p_soft_counters(:, k)
        rescaler = (p_coll(0,position,k)+p_coll(0,position_ifks,k))/sqrtshat*2
        p_i_fks_cnt(:,2) = (p_coll(:,position,k)+p_coll(:,position_ifks,k))/rescaler
        p_i_fks_cnt(:,1) = p_i_fks_cnt(:,2)
        split_type_used(1) = .TRUE.
        split_type_used(2) = .FALSE.
        y_ij_fks = y_ij_fks_inp(k)
        delta_used = deltac
        xicut_used = xic

        call sreal(pp(:,:,k),xi_i_fks,y_ij_fks, dummy_wgt)
        s_s = fks_Sij(pp(:,:,k),i_fks,j_fks,xi_i_fks,y_ij_fks)
        me(k) = dummy_wgt*s_s

        me_s(k) = 0.0d0
        me_co(k) = 0.0d0
        me_sc(k) = 0.0d0
        
        call fks_inc_chooser
        delta_used = deltac
        xicut_used = xic
        if (y_ij_fks.gt.(1-delta_used)) then
            call getangles(p_born(:,position),th_m,cth_m,sth_m,phi_m,cphi_m,sphi_m)
C           first trial
            support = pp(:,position,k)/pp(0,position,k)*500.0d0
            qin = support(1:3)
            support(1) = cth_m*cphi_m*qin(1)+cth_m*sphi_m*qin(2)-sth_m*qin(3)
            support(2) = -sphi_m*qin(1)+cphi_m*qin(2)
            support(3) = sth_m*cphi_m*qin(1)+sth_m*sphi_m*qin(2)+cth_m*qin(3)
            call getangles(support,th_d,cth_d,sth_d,phi_d,cphi_d,sphi_d)
            phi_daught = phi_d
            phi_mother_fks = phi_m
            xij_aor=-exp( 2*ximag*(phi_mother_fks+phi_daught) )
            NFKSPROCESS = fks_sectors(k)
            call fks_inc_chooser
            call leshouche_inc_chooser
            firsttime_pdf = .true.
            call setfksfactor(.false.)

            delta_used = deltac
            xicut_used = xic

            p_born = p_born_sampl(:,:,k)
            p1_cnt(:,:,1) = p_coll(:,:,k)
            p_born_coll = p_born
            dummy_wgt = 0.0d0
            call sreal(p1_cnt(:,:,1),xi_i_fks,1.0d0, dummy_wgt)
            s_s = fks_Sij(p1_cnt(:,:,1),i_fks,j_fks,xi_i_fks,1.0d0)
            me_co(k) = dummy_wgt*s_s
            if (s_s.le.0d0) then
                me_co(k) = 0.0d0
            endif
        endif
        delta_used = deltac
        xicut_used = xic
        if ((xi_i_fks.lt.(xicut_used))) then
            NFKSPROCESS = fks_sectors(k)
            call fks_inc_chooser
            CALCULATEDBORN = .FALSE.
            call sborn(p_born,dummy_wgt)
            call setfksfactor(.false.)

            delta_used = deltac
            xicut_used = xic
            
            p_born = p_born_sampl(:,:,k)
            p1_cnt(:,:,0) = p_soft(:,:,k)
            p_born_coll = p_born
            call sreal(p1_cnt(:,:,0),0.0d0,y_ij_fks, dummy_wgt)
            s_s = fks_Sij(p1_cnt(:,:,0),i_fks,j_fks,0.0d0,y_ij_fks)
            me_s(k) = dummy_wgt*s_s 
            

            if (s_s.le.0d0) then
                me_s(k) = 0.0d0
            endif
        endif
        delta_used = deltac
        xicut_used = xic
        if ((y_ij_fks.gt.(1-delta_used)).and.(xi_i_fks.lt.xicut_used)) then
            NFKSPROCESS = fks_sectors(k)
            call fks_inc_chooser
            CALCULATEDBORN = .FALSE.
            call sborn(p_born,dummy_wgt)
            call setfksfactor(.false.)

            delta_used = deltac
            xicut_used = xic
            
            p_born = p_born_sampl(:,:,k)
            p1_cnt(:,:,2) = p_soft(:,:,k)
            p_born_coll = p_born
            call sreal(p1_cnt(:,:,2),0.0d0,1.0d0, dummy_wgt)
            s_s = fks_Sij(p1_cnt(:,:,2),i_fks,j_fks,0.0d0,1.0d0)
            me_sc(k) = dummy_wgt*s_s
            
            if (s_s.le.0d0) then
                me_sc(k) = 0.0d0
            endif
        endif

      enddo
      end subroutine
