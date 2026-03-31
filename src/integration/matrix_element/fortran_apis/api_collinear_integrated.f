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
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      double precision Pi, Zeta3, CA, CF, TF
      parameter (CA=3.d0)
      parameter (Pi=3.1415926535897932385d0)
      parameter (CF=4/3.d0)
      parameter (TF=1/2.d0)
      
      call setpara('param_card.dat')
      n_ext = nexternal-1
      QES2 = 8302.4899239999995d0
      mu_r = sqrt(8302.4899239999995d0)
      G = 1.2229388294466237d0
      call update_as_param()
      alpha_s = G**2/(4.0D0*3.1415926535897931D0)
      xicut_used = 0.5d0
      QES2 = 8302.4899239999995d0
      shat = 1000000.0d0
      sqrtshat = 1000.0d0
      c(0)=CA
      c(1)=CF
      gamma(0)=( 11d0*CA-2d0*Nf )/6d0
      gamma(1)=CF*3d0/2d0
      gammap(0)=( 67d0/9d0 - 2d0*PI**2/3d0 )*CA - 23d0/18d0*Nf
      gammap(1)=( 13/2d0 - 2d0*PI**2/3d0 )*CF

      end subroutine

      subroutine call_collinear_integrated_counterterm(count, p, deltac,xic, me)
      implicit none
      integer n_ext
      real*8 alpha_s
      double precision c(0:1),gamma(0:1),gammap(0:1),gamma_ph,gammap_ph
      double precision deltac, xic
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
      double precision Q
      include 'coupl.inc'
      double precision Ej
      integer aj
      integer count, medim, precdim
      real*8 p(0:3, nexternal-1, 0:count-1)
      real*8 me(0:count-1)
      double precision aso2pi
      include 'fks_powers.inc'
      double precision Pi, Zeta3, CA, CF, TF

      
C      include 'veto_xsec.inc'
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
     

      integer i, j, k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      g = 1.2229388294466237d0   
      NFKSPROCESS=1
      aso2pi=g**2/(8*3.1415926535897931D0**2)
      do k=0,count-1
        call fks_inc_chooser
        CALCULATEDBORN = .FALSE.
        Q = 0.0d0
        me(k) = 0.0d0
        SAVEMOM(:,1) = p(0,:,k)
        SAVEMOM(:,2) = p(3,:,k)
        call SBORN(p(:,:,k), me(k))
        do i=1 ,nexternal-1
c set the various color factors according to the 
c type of the leg
          if (particle_type(i).eq.8) then
            aj=0
          else if(abs(particle_type(i)).eq.3) then
            aj=1
          else
            aj=-1
          end if
          
          Ej=p(0,i,k)
          
C     set colour factors
        
C Disclaimer: the variable deltaO is set in a .inc file, deltaO=1
          if (aj.eq.-1) cycle
          c_used = c(aj)
          gamma_used = gamma(aj)
          gammap_used = gammap(aj)
          xicut_used=xic
          if (i.gt.nincoming) then 
C Q terms for final state parton
                   Q = Q+gammap_used
     &                    -dlog(shat*deltac/2d0/QES2)*( gamma_used-
     &                    2d0*c_used*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &                    +2d0*c_used*( dlog(2d0*Ej/sqrtshat)**2
     &                    -dlog(xicut_used)**2 )
     &                    -2d0*gamma_used*dlog(2d0*Ej/sqrtshat)         
  
          endif
        enddo
        me(k) = me(k) * Q * aso2pi

      enddo
      end subroutine
