import ctypes
import os
import sys
import numpy as np
import numpy.ctypeslib as npct
import torch

int_ptr = ctypes.POINTER(ctypes.c_int)
double_ptr = ctypes.POINTER(ctypes.c_double)
double_arr_3d = npct.ndpointer(dtype=np.float64, ndim=3, flags="FORTRAN")
double_arr_2d = npct.ndpointer(dtype=np.float64, ndim=2, flags="FORTRAN")
double_arr = npct.ndpointer(dtype=np.float64, ndim=1, flags="FORTRAN")
int_arr = npct.ndpointer(dtype=np.int32, ndim=1, flags="FORTRAN")


def silence_please():
    # Always use the real stdout FD
    real_stdout = sys.__stdout__

    if not hasattr(real_stdout, "fileno"):
        # Extremely rare, but fail safely
        def unsilence():
            pass

        return unsilence

    fd = real_stdout.fileno()

    def _redirect_stdout(to):
        os.dup2(to.fileno(), fd)

    # Save a copy of the original stdout FD
    saved_fd = os.dup(fd)

    # Redirect to /dev/null
    devnull = open(os.devnull, "w")
    _redirect_stdout(devnull)

    def unsilence():
        os.dup2(saved_fd, fd)
        os.close(saved_fd)
        devnull.close()

    return unsilence


class BornMatrixElement:
    def __init__(self, api_path: str, verbose: bool = False):
        self.verbose = verbose

        self.lib = ctypes.cdll.LoadLibrary(api_path)
        self.lib.init_api_.argtypes = [int_ptr, double_ptr]
        self.lib.init_api_.restype = None
        self.lib.call_matrix_element_.argtypes = [int_ptr, double_arr_3d, double_arr]
        self.lib.call_matrix_element_.restype = None

        self.api_dir = os.path.dirname(api_path)

        cwd = os.getcwd()
        os.chdir(self.api_dir)
        mom_dim = ctypes.c_int()
        alpha_s = ctypes.c_double()
        self.lib.init_api_(ctypes.byref(mom_dim), ctypes.byref(alpha_s))
        os.chdir(cwd)
        self.alpha_s = alpha_s.value
        self.n_external = mom_dim.value
        self.first_call = True

    def __call__(self, momenta: torch.Tensor) -> torch.Tensor:
        shape = momenta.shape
        if len(shape) != 3 or shape[1] != self.n_external or shape[2] != 4:
            raise ValueError(
                f"Expected shape (*, {self.n_external}, 4) for momenta, "
                f"got {tuple(shape)}"
            )
        mom = np.asfortranarray(momenta.permute(2, 1, 0).cpu().numpy(), dtype=np.float64)
        count = len(momenta)
        me = np.zeros(count, dtype=np.float64, order="F")
        if self.first_call:
            cwd = os.getcwd()
            os.chdir(self.api_dir)
            if not self.verbose:
                unsilence = silence_please()

        self.lib.call_matrix_element_(ctypes.byref(ctypes.c_int(count)), mom, me)
        if self.first_call:
            os.chdir(cwd)
            self.first_call = False
            if not self.verbose:
                unsilence()
        # we ignore prec and rcode for now
        # return only born and virtual
        return torch.from_numpy(me).to(momenta.device)


class LoopMatrixElement:
    def __init__(self, api_path: str, verbose: bool = False):
        self.verbose = verbose

        self.lib = ctypes.cdll.LoadLibrary(api_path)
        self.lib.init_api_.argtypes = [int_ptr, int_ptr, int_ptr, double_ptr]
        self.lib.init_api_.restype = None
        self.lib.call_matrix_element_.argtypes = [
            int_ptr,
            int_ptr,
            int_ptr,
            double_arr_3d,
            double_arr_3d,
            double_arr_2d,
            int_arr,
        ]
        self.lib.call_matrix_element_.restype = None

        self.api_dir = os.path.dirname(api_path)

        cwd = os.getcwd()
        os.chdir(self.api_dir)
        self.me_dim = ctypes.c_int()
        self.prec_dim = ctypes.c_int()
        mom_dim = ctypes.c_int()
        alpha_s = ctypes.c_double()
        self.lib.init_api_(
            ctypes.byref(self.me_dim),
            ctypes.byref(self.prec_dim),
            ctypes.byref(mom_dim),
            ctypes.byref(alpha_s),
        )
        os.chdir(cwd)
        self.alpha_s = alpha_s.value
        self.n_external = mom_dim.value
        self.first_call = True

    def __call__(self, momenta: torch.Tensor) -> torch.Tensor:
        shape = momenta.shape
        if len(shape) != 3 or shape[1] != self.n_external or shape[2] != 4:
            raise ValueError(
                f"Expected shape (*, {self.n_external}, 4) for momenta, "
                f"got {tuple(shape)}"
            )
        mom = np.asfortranarray(momenta.permute(2, 1, 0).cpu().numpy(), dtype=np.float64)
        count = len(momenta)
        me = np.zeros((4, self.me_dim.value + 1, count), dtype=np.float64, order="F")
        prec = np.zeros((self.prec_dim.value + 1, count), dtype=np.float64, order="F")
        rcode = np.zeros(count, dtype=np.int32)
        if self.first_call:
            cwd = os.getcwd()
            os.chdir(self.api_dir)
            if not self.verbose:
                unsilence = silence_please()

        self.lib.call_matrix_element_(
            ctypes.byref(ctypes.c_int(count)),
            ctypes.byref(self.me_dim),
            ctypes.byref(self.prec_dim),
            mom,
            me,
            prec,
            rcode,
        )

        if self.first_call:
            os.chdir(cwd)
            self.first_call = False
            if not self.verbose:
                unsilence()
        # we ignore prec and rcode for now
        # return only born and virtual
        return torch.from_numpy(me[:2, 0, :]).T.to(momenta.device)


class SoftIntegratedCounterterm:
    def __init__(self, api_path: str, verbose: bool = False):
        self.verbose = verbose

        self.lib = ctypes.cdll.LoadLibrary(api_path)
        self.lib.init_api_.argtypes = [int_ptr, double_ptr]
        self.lib.init_api_.restype = None
        self.lib.call_soft_integrated_counterterm_.argtypes = [
            int_ptr,
            double_arr_3d,
            double_ptr,
            double_arr,
        ]
        self.lib.call_soft_integrated_counterterm_.restype = None

        self.api_dir = os.path.dirname(api_path)

        cwd = os.getcwd()
        os.chdir(self.api_dir)
        mom_dim = ctypes.c_int()
        alpha_s = ctypes.c_double()
        self.lib.init_api_(ctypes.byref(mom_dim), ctypes.byref(alpha_s))
        os.chdir(cwd)
        self.alpha_s = alpha_s.value
        self.n_external = mom_dim.value
        self.first_call = True

    def __call__(self, momenta: torch.Tensor, xi_cut: float = 0.5) -> torch.Tensor:
        shape = momenta.shape
        if len(shape) != 3 or shape[1] != self.n_external or shape[2] != 4:
            raise ValueError(
                f"Expected shape (*, {self.n_external}, 4) for momenta, "
                f"got {tuple(shape)}"
            )
        mom = np.asfortranarray(momenta.permute(2, 1, 0).cpu().numpy(), dtype=np.float64)
        count = len(momenta)
        me = np.zeros(count, dtype=np.float64, order="F")
        if self.first_call:
            cwd = os.getcwd()
            os.chdir(self.api_dir)
            if not self.verbose:
                unsilence = silence_please()

        self.lib.call_soft_integrated_counterterm_(
            ctypes.byref(ctypes.c_int(count)),
            mom,
            ctypes.byref(ctypes.c_double(xi_cut)),
            me,
        )
        if self.first_call:
            os.chdir(cwd)
            self.first_call = False
            if not self.verbose:
                unsilence()
        # we ignore prec and rcode for now
        # return only born and virtual
        return torch.from_numpy(me).to(momenta.device)


class CollinearIntegratedCounterterm:
    def __init__(self, api_path: str, verbose: bool = False):
        self.verbose = verbose

        self.lib = ctypes.cdll.LoadLibrary(api_path)
        self.lib.init_api_.argtypes = [int_ptr, double_ptr]
        self.lib.init_api_.restype = None
        self.lib.call_collinear_integrated_counterterm_.argtypes = [
            int_ptr,
            double_arr_3d,
            double_ptr,
            double_ptr,
            double_arr,
        ]
        self.lib.call_collinear_integrated_counterterm_.restype = None

        self.api_dir = os.path.dirname(api_path)

        cwd = os.getcwd()
        os.chdir(self.api_dir)
        mom_dim = ctypes.c_int()
        alpha_s = ctypes.c_double()
        self.lib.init_api_(ctypes.byref(mom_dim), ctypes.byref(alpha_s))
        os.chdir(cwd)
        self.alpha_s = alpha_s.value
        self.n_external = mom_dim.value
        self.first_call = True

    def __call__(
        self, momenta: torch.Tensor, xi_cut: float = 0.5, delta_cut: float = 1.0
    ) -> torch.Tensor:
        shape = momenta.shape
        if len(shape) != 3 or shape[1] != self.n_external or shape[2] != 4:
            raise ValueError(
                f"Expected shape (*, {self.n_external}, 4) for momenta, "
                f"got {tuple(shape)}"
            )
        mom = np.asfortranarray(momenta.permute(2, 1, 0).cpu().numpy(), dtype=np.float64)
        count = len(momenta)
        me = np.zeros(count, dtype=np.float64, order="F")
        if self.first_call:
            cwd = os.getcwd()
            os.chdir(self.api_dir)
            if not self.verbose:
                unsilence = silence_please()
        self.lib.call_collinear_integrated_counterterm_(
            ctypes.byref(ctypes.c_int(count)),
            mom,
            ctypes.byref(ctypes.c_double(delta_cut)),
            ctypes.byref(ctypes.c_double(xi_cut)),
            me,
        )
        if self.first_call:
            os.chdir(cwd)
            self.first_call = False
            if not self.verbose:
                unsilence()
        # we ignore prec and rcode for now
        # return only born and virtual
        return torch.from_numpy(me).to(momenta.device)


class RealEmission:
    def __init__(self, api_path: str, verbose: bool = True):
        self.verbose = verbose

        self.lib = ctypes.cdll.LoadLibrary(api_path)
        self.lib.init_api_.argtypes = [int_ptr, double_ptr]
        self.lib.init_api_.restype = None
        self.lib.call_sreal_me_.argtypes = [
            int_ptr,
            double_arr_3d,
            double_arr_3d,
            double_arr_3d,
            double_arr_3d,
            double_arr_2d,
            int_arr,
            double_arr,
            double_arr,
            double_ptr,
            double_ptr,
            double_arr,
            double_arr,
            double_arr,
            double_arr,
        ]
        self.lib.call_sreal_me_.restype = None

        self.api_dir = os.path.dirname(api_path)

        cwd = os.getcwd()
        os.chdir(self.api_dir)
        mom_dim = ctypes.c_int()
        alpha_s = ctypes.c_double()
        self.lib.init_api_(ctypes.byref(mom_dim), ctypes.byref(alpha_s))
        os.chdir(cwd)
        self.alpha_s = alpha_s.value
        self.n_external = mom_dim.value
        self.first_call = True

    def __call__(
        self,
        momenta: torch.Tensor,
        momenta_born: torch.Tensor,
        momenta_soft: torch.Tensor,
        momenta_coll: torch.Tensor,
        momenta_soft_ct: torch.Tensor,
        fks_sector: torch.Tensor,
        xi: torch.Tensor,
        y: torch.Tensor,
        xi_cut: float = 0.5,
        delta_cut: float = 1.0,
    ) -> torch.Tensor:
        shape = momenta.shape
        if len(shape) != 3 or shape[1] != self.n_external or shape[2] != 4:
            raise ValueError(
                f"Expected shape (*, {self.n_external}, 4) for momenta, "
                f"got {tuple(shape)}"
            )
        mom = np.asfortranarray(momenta.permute(2, 1, 0).cpu().numpy(), dtype=np.float64)
        mom_b = np.asfortranarray(
            momenta_born.permute(2, 1, 0).cpu().numpy(), dtype=np.float64
        )
        mom_s = np.asfortranarray(
            momenta_soft.permute(2, 1, 0).cpu().numpy(), dtype=np.float64
        )
        mom_c = np.asfortranarray(
            momenta_coll.permute(2, 1, 0).cpu().numpy(), dtype=np.float64
        )
        mom_soft_ct = np.asfortranarray(
            momenta_soft_ct.permute(1, 0).cpu().numpy(), dtype=np.float64
        )
        fks = np.asfortranarray(fks_sector.cpu().numpy(), dtype=np.int32)
        xi_i_fks = np.asfortranarray(xi.cpu().numpy(), dtype=np.float64)
        y_ij_fks = np.asfortranarray(y.cpu().numpy(), dtype=np.float64)
        count = len(momenta)
        me = np.zeros(count, dtype=np.float64, order="F")
        me_s = np.zeros(count, dtype=np.float64, order="F")
        me_c = np.zeros(count, dtype=np.float64, order="F")
        me_sc = np.zeros(count, dtype=np.float64, order="F")

        if self.first_call:
            cwd = os.getcwd()
            os.chdir(self.api_dir)
            if not self.verbose:
                unsilence = silence_please()

        self.lib.call_sreal_me_(
            ctypes.byref(ctypes.c_int(count)),
            mom,
            mom_b,
            mom_s,
            mom_c,
            mom_soft_ct,
            fks,
            xi_i_fks,
            y_ij_fks,
            ctypes.byref(ctypes.c_double(delta_cut)),
            ctypes.byref(ctypes.c_double(xi_cut)),
            me,
            me_s,
            me_c,
            me_sc,
        )
        if self.first_call:
            os.chdir(cwd)
            self.first_call = False
            if not self.verbose:
                unsilence()
        # we ignore prec and rcode for now
        # return only born and virtual
        return (
            torch.from_numpy(me).to(momenta.device),
            torch.from_numpy(me_s).to(momenta.device),
            torch.from_numpy(me_c).to(momenta.device),
            torch.from_numpy(me_sc).to(momenta.device),
        )
