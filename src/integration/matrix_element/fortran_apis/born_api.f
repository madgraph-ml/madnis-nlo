      subroutine init_api(n_ext, alpha_s)
      implicit none
      integer n_ext
      real*8 alpha_s

      include 'nexternal.inc'
      include 'coupl.inc'

      call setpara('param_card.dat')
      n_ext = nexternal
      mu_r = sqrt(8302.4899239999995d0)
      call update_as_param()
      alpha_s = G**2/(4.D0*3.1415926535897931D0)

      end subroutine

      subroutine call_matrix_element(count, p, me)
      implicit none
      include 'nexternal.inc'

      integer count, medim, precdim
      real*8 p(0:3, nexternal, 0:count-1)
      real*8 me(0:count-1)

      integer i, j, k

      do i=0,count-1
        call smatrix(p(:,:,i), me(i))
      enddo

      end subroutine
