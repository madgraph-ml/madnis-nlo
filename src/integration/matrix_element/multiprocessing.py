import os
import torch
from multiprocessing import Pool


def _init_pool_unary(api_cls, api_args, api_kwargs):
    global _api_instance
    _api_instance = api_cls(*api_args, **api_kwargs)


def _run_api_unary(args):
    global _api_instance
    x_batch, kwargs = args
    return _api_instance(x_batch, **kwargs)


def _init_pool_real(api_cls, api_args, api_kwargs):
    global _real_api_instance
    _real_api_instance = api_cls(*api_args, **api_kwargs)


def _run_api_real(args):
    global _real_api_instance
    return _real_api_instance(*args)


class ThreadedUnaryAPI:
    def __init__(self, api_cls, *api_args, n_processes=None, **api_kwargs):
        self.n_processes = n_processes or os.cpu_count()
        self._api_local = api_cls(*api_args, **api_kwargs)  # local fallback
        self.pool = Pool(
            processes=self.n_processes,
            initializer=_init_pool_unary,
            initargs=(api_cls, api_args, api_kwargs),
        )

    def __call__(self, x: torch.Tensor, **kwargs):
        B = x.shape[0]
        if B == 0 or B < self.n_processes:
            return self._api_local(x, **kwargs)

        n_chunks = min(B, self.n_processes)
        batches = x.chunk(n_chunks, dim=0)
        results = self.pool.map(_run_api_unary, [(b, kwargs) for b in batches])
        # results is a list of tensors or tuples of tensors
        if isinstance(results[0], torch.Tensor):
            return torch.cat(results, dim=0)
        else:
            # assume tuple of tensors
            zipped = list(zip(*results))
            return tuple(torch.cat(r, dim=0) for r in zipped)

    def close(self):
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ThreadedRealEmissionAPI:
    def __init__(self, api_cls, *api_args, n_processes=None, **api_kwargs):
        self.n_processes = n_processes or os.cpu_count()
        self._api_local = api_cls(*api_args, **api_kwargs)
        self.pool = Pool(
            processes=self.n_processes,
            initializer=_init_pool_real,
            initargs=(api_cls, api_args, api_kwargs),
        )

    def __call__(self, p_np1, p_n, p_soft, p_coll, k_ct, fks_sec, xi, y,
                 xi_cut=0.5, delta_cut=1.0):
        B = p_np1.shape[0]
        if B == 0:
            return self._api_local(p_np1, p_n, p_soft, p_coll, k_ct, fks_sec, xi, y,
                                   xi_cut, delta_cut)

        if B < self.n_processes:
            return self._api_local(p_np1, p_n, p_soft, p_coll, k_ct, fks_sec, xi, y,
                                   xi_cut, delta_cut)

        n_chunks = min(B, self.n_processes)

        args_batches = list(
            zip(
                p_np1.chunk(n_chunks, dim=0),
                p_n.chunk(n_chunks, dim=0),
                p_soft.chunk(n_chunks, dim=0),
                p_coll.chunk(n_chunks, dim=0),
                k_ct.chunk(n_chunks, dim=0),
                fks_sec.chunk(n_chunks, dim=0),
                xi.chunk(n_chunks, dim=0),
                y.chunk(n_chunks, dim=0),
                [xi_cut] * n_chunks,
                [delta_cut] * n_chunks,
            )
        )

        results = self.pool.map(_run_api_real, args_batches)
        zipped = list(zip(*results))
        return tuple(torch.cat(r, dim=0) for r in zipped)

    def close(self):
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        # best-effort only; don't raise during shutdown
        try:
            self.close()
        except Exception:
            pass
