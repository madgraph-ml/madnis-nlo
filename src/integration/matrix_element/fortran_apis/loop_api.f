      subroutine init_api(matelem_array_dim, nsquaredso_loop, n_ext, alpha_s)
      implicit none
      integer matelem_array_dim, nsquaredso_loop, n_ext
      real*8 alpha_s

      include 'nexternal.inc'
      include 'coupl.inc'
      call setpara('param_card.dat')
      mu_r = sqrt(8302.4899239999995d0)
      call update_as_param()
      n_ext = nexternal
      alpha_s = G**2/(4.D0*3.1415926535897931D0)

      call ml5_0_get_answer_dimension(matelem_array_dim)
      call ml5_0_get_nsqso_loop(nsquaredso_loop)
      call setpara('param_card.dat')

      end subroutine

      subroutine call_matrix_element(count, medim, precdim, p, me, prec, rcode)
      implicit none
      include 'nexternal.inc'

      integer count, medim, precdim, n_ext
      real*8 p(0:3, nexternal, 0:count-1)
      real*8 me(0:3, 0:medim, 0:count-1)
      real*8 prec(0:precdim, 0:count-1)
      integer rcode(0:count-1)
      integer i, j, k

      do i=0,count-1
        call ml5_0_sloopmatrix_thres(
     $    p(:,:,i), me(:,:,i), -1.0D0, prec(:,i), rcode(i))
      enddo

      end subroutine
