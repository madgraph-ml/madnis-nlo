      subroutine init_api(n_ext, alpha_s)
      implicit none
      integer n_ext
      double precision alpha_s
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision xicut_used,eikIreg
      common /cxicut_used/xicut_used
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      LOGICAL NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      COMMON /C_NEED_LINKS/NEED_COLOR_LINKS, NEED_CHARGE_LINKS

      include 'nexternal.inc'
      include 'coupl.inc'
      include 'nFKSconfigs.inc'
      include 'q_es.inc'

      call setpara('param_card.dat')
      xicut_used = 0.5d0
      QES2 = 8302.4899239999995d0
      shat = 1000000.0d0
C     THIS IS HARDCODED!!!!!!
      n_ext = nexternal-1
      QES2 = 8302.4899239999995d0
      mu_r = sqrt(8302.4899239999995d0)
      G = 1.2229388294466237d0
      call update_as_param()
      alpha_s = G**2/(4.D0*3.14159265358979323845D0)

      end subroutine





      subroutine call_soft_integrated_counterterm(count, p, xic, wgt_out)
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'nFKSconfigs.inc'
      include 'fks_powers.inc'
      include 'q_es.inc'
      integer count, medim, precdim
      integer n_ext
      double precision p(0:3,1:nexternal-1,0:count-1), wgt, wgt_out(0:count-1)
      double precision xic
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat  
      double precision xicut_used,eikIreg
      common /cxicut_used/xicut_used
      INTEGER M,N,i
      integer idk
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      LOGICAL NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      COMMON /C_NEED_LINKS/NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      double precision Pi
      LOGICAL FIRSTTIME
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      
      parameter (Pi=3.1415926535897932385d0)
      
      NEED_COLOR_LINKS = .TRUE.
      NEED_CHARGE_LINKS = .FALSE.
      G = 1.2229388294466237d0
      
      
      n_ext = nexternal-1
      do i=0,count-1
            NFKSPROCESS = 1
            call fks_inc_chooser
            wgt_out(i) = 0.0d0
            CALCULATEDBORN = .FALSE.
            FIRSTTIME= .FALSE.
            call SBORN(p(:,:,i),wgt)
            wgt=0.0d0
            xicut_used = xic
            do m=3,n_ext
                  do n=m,n_ext
                        eikIreg = 0.0d0
                        if (m.eq.n) cycle
                        call SBORN_SF(p(:,:,i), m, n, wgt)
                        if (wgt.ne.0.0d0) then
                              call eikonal_Ireg(p(:,:,i),m,n,xicut_used,eikIreg)
                              wgt_out(i) = wgt_out(i)+wgt*eikIreg                     
                        endif
                  enddo
            enddo
            wgt_out(i) = -wgt_out(i)*2/(8*pi**2)
      enddo
      end subroutine
