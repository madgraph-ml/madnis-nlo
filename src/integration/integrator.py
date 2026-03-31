import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import functools

from src.integration.integrand import NLOCrossSection, NLOSurrogateCrossSection
from documenter import Documenter
from madnis.integrator import (
    Integrator,
    kl_divergence,
    rkl_divergence,
    stratified_variance,
    VegasPreTraining,
    stratified_variance_softclip,
)
from src.utils.utils import clipped_stratified_variance
from src.integration.phase_space.multichannel import MadnisPhaseSpace
from src.utils.physics import tensors_to_numpy
from src.integration.plots import Plots, plot_train_metrics, plot_fks_sectors


def get_start_time():
    return time.time(), time.process_time()


def print_run_time(start):
    start_time, start_cpu_time = start
    train_time = time.time() - start_time
    train_cpu_time = time.process_time() - start_cpu_time
    print(
        f"--- Run time: {str(timedelta(seconds=round(train_time, 2) + 1e-5))[:-4]} wall time, "
        f"{str(timedelta(seconds=round(train_cpu_time, 2) + 1e-5))[:-4]} cpu time ---"
    )


class TimedIntegrand(nn.Module):
    """Wrapper around an integrand that accumulates wall time spent in forward()."""

    def __init__(self, integrand):
        super().__init__()
        self._integrand = integrand
        self.total_time = 0.0

    def forward(self, *args, **kwargs):
        t0 = time.time()
        result = self._integrand(*args, **kwargs)
        self.total_time += time.time() - t0
        return result

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._integrand, name)


def print_metrics(name: str, metrics, integrand_time: float = None):
    print(f"Integrand: {name}")
    print(f"  Result:       {metrics.integral:.8f} +- {metrics.error:.8f}")
    print(f"  Rel. error:   {metrics.rel_error * 100:.4f}%")
    print(f"  RSD:          {metrics.rel_stddev:.2f}")
    if getattr(metrics, "rel_stddev_after_cuts", None) is not None:
        print(f"  RSD (accepted):   {metrics.rel_stddev_after_cuts:.2f}")
    print(f"  RSD opt.:     {metrics.rel_stddev_opt:.2f}")
    print(f"  Count:        {metrics.count}")
    if getattr(metrics, "cut_efficiency", None) is not None:
        print(f"  Cut eff.:     {metrics.cut_efficiency * 100:.2f}%")
    if integrand_time is not None:
        print(f"  Integrand t:  {str(timedelta(seconds=round(integrand_time, 2) + 1e-5))[:-4]}")
    if len(metrics.channel_integrals) <= 1:
        print()
        return

    print("  Channels:")
    for i, (mean, err, rsd, count) in enumerate(
        zip(
            metrics.channel_integrals,
            metrics.channel_errors,
            metrics.channel_rel_stddevs,
            metrics.channel_counts,
        )
    ):
        print(f"    Channel {i}: I={mean:.8f} +- {err:.8f}, RSD={rsd:.2f}, N={count}")


class ParxIntegrator:
    def __init__(self, params: dict, device: torch.device, doc: Documenter):
        self.params = params
        self.device = device
        self.doc = doc
        self.cross_sections = {
            name: eval(cs_params["class"])(
                params["e_cm"], cs_params, device
            )
            for name, cs_params in params["cross_sections"].items()
        }
        self.madnis_phase_space = MadnisPhaseSpace(
            params, is_nlo="nlo" in self.cross_sections or "nlo_surr" in self.cross_sections
        )
        self.seed = params.get("seed", None)
        print(f"seed: {self.seed}")
        self.integrands = {
            name: self.madnis_phase_space.build_integrand(cross_section)
            for name, cross_section in self.cross_sections.items()
        }
        print("Built integrands:", list(self.integrands.keys()))
        if params["loss"] == "stratified_variance_softclip":
            loss = functools.partial(
                stratified_variance_softclip,
                threshold=params.get("loss_softclip_threshold", 30.0),
            )
        else:
            loss = {
                "stratified_variance": stratified_variance,
                "clipped_stratified_variance": clipped_stratified_variance,
                "kl_divergence": kl_divergence,
                "rkl_divergence": rkl_divergence,
            }[params["loss"]]

        def build_scheduler(optimizer):
            if params["lr_scheduler"] == "exponential":
                decay_rate = params["lr_decay"] ** (
                    1 / max(params["train_iterations"], 1)
                )
                return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
            elif params["lr_scheduler"] == "inverse":
                inv_func = lambda step: 1 / (
                    1 + params["lr_decay"] * step / params["train_iterations"]
                )
                return torch.optim.lr_scheduler.LambdaLR(optimizer, inv_func)
            elif params["lr_scheduler"] == "onecycle":
                print("Using OneCycle LR scheduler")
                return torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(params["lr_max"]),
                    total_steps=params["train_iterations"],
                    div_factor=100.0,
                )
            elif params["lr_scheduler"] == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=params["train_iterations"]
                )
            else:
                return None

        self.integrator = Integrator(
            integrand=self.integrands[params["integrand"]],
            flow_kwargs=dict(
                permutations=params["permutations"],
                blocks=params["blocks"],
                layers=params["layers"],
                units=params["units"],
                bins=params["spline_bins"],
                activation=nn.LeakyReLU,
            ),
            cwnet_kwargs=dict(
                layers=params["cwnet_layers"],
                units=params["cwnet_units"],
            ),
            loss=loss,
            batch_size=params["batch_size_offset"],
            batch_size_per_channel=params["batch_size_per_channel"],
            learning_rate=params["lr"],
            scheduler=build_scheduler,
            drop_zero_integrands=params["drop_zero_integrands"],
            batch_size_threshold=params["batch_size_threshold"],
            uniform_channel_ratio=params["uniform_channel_ratio"],
            integration_history_length=params["integration_history_length"],
            buffer_capacity=params.get("buffer_capacity", 0),
            minimum_buffer_size=params.get("minimum_buffer_size", 0),
            buffered_steps=params.get("buffered_steps", 0),
            channel_dropping_threshold=params["channel_dropping_threshold"],
            channel_dropping_interval=params["channel_dropping_interval"],
            channel_grouping_mode="uniform",
            freeze_cwnet_iteration=int(
                params["train_iterations"] * (1 - params["fixed_cwnet_fraction"])
            ),
            device=device,
        )

        vegas_mode = params.get("vegas", False)
        pretrain_vegas = params.get("vegas_pretraining", False)

        if vegas_mode or pretrain_vegas:
            self.vegas = VegasPreTraining(
                self.integrator,
                bins=params["vegas_bins"],
                damping=params["vegas_damping"],
            )
            self.train_flow = not vegas_mode
        else:
            self.vegas = None
            self.train_flow = True

        self.fade_in = self.params.get("fade_in", False)
        self.train_iterations = self.params["train_iterations"]
        self.saturation_time = self.params.get("saturation_time", 0.8)
        self.gamma_delay = self.params.get("gamma_delay", 4.0)

    def train(self):
        if self.vegas is not None:
            vegas_losses = []
            vegas_train_results = []
            print(f"Running VEGAS {'pre-' if self.train_flow else ''}training")

            def vegas_callback(status):
                vegas_losses.append(status.variance)
                batch = status.step + 1
                vegas_loss = np.mean(vegas_losses)
                info = [f"[VEGAS] Batch {batch:6d}: loss={vegas_loss:.6f}"]
                batch_results = {"batch": batch, "loss": vegas_loss}
                vegas_train_results.append(batch_results)
                print(", ".join(info))
                vegas_losses.clear()

            start_time = get_start_time()
            self.vegas.train(
                [self.params["vegas_batch_size"]] * self.params["vegas_iterations"],
                callback=vegas_callback,
            )
            self.vegas_train_results = vegas_train_results
            self.vegas.initialize_integrator()
            print_run_time(start_time)
            print()

        if not self.train_flow:
            return

        online_losses = []
        buffered_losses = []
        train_results = []
        if self.params.get("fade_in", False):
            print("Using real-contribution fade-in during training")

        def smootherstep(t, T, gamma=6.0):
            if t <= 0:
                return 0.0
            if t >= T:
                return 1.0
            x = t / T
            x = x**gamma
            out = 0.0
            if x**3 * (x * (x * 6 - 15) + 10) > 0.5:
                out = x**3 * (x * (x * 6 - 15) + 10)
            return out

        def callback(status):
            if status.buffered:
                buffered_losses.append(status.loss)
            else:
                online_losses.append(status.loss)
            batch = status.step + 1
            # I want this to be done at every step for smootheness
            is_nlo = "nlo" in self.cross_sections
            if is_nlo:
                if self.fade_in:
                    self.cross_sections["nlo"].real_weight = smootherstep(
                        batch,
                        self.train_iterations * self.saturation_time,
                        self.gamma_delay,
                    )
                else:
                    self.cross_sections["nlo"].real_weight = 1.0
            if batch % self.params["log_interval"] != 0:
                return
            online_loss = np.mean(online_losses)
            if is_nlo and self.fade_in:
                info = [
                    f"[MadNIS] Batch {batch:6d}: loss={online_loss:.6f}, real_weight={self.cross_sections['nlo'].real_weight:.6f}"
                ]
            else:
                info = [f"[MadNIS] Batch {batch:6d}: loss={online_loss:.6f}"]
            batch_results = {"batch": batch, "online_loss": online_loss}
            if len(buffered_losses) > 0:
                buffered_loss = np.mean(buffered_losses)
                info.append(f"buf={buffered_loss:.6f}")
                batch_results["buffered_loss"] = buffered_loss
            if status.learning_rate is not None:
                info.append(f"lr={status.learning_rate:.4e}")
                batch_results["learning_rate"] = status.learning_rate
            train_results.append(batch_results)
            print(", ".join(info))
            online_losses.clear()
            buffered_losses.clear()

        start_time = get_start_time()
        self.integrator.train(self.params["train_iterations"], callback)
        self.train_results = train_results
        print_run_time(start_time)
        print()

    def _sample_r_channels(self, n_samples: int, integrator=None):
        integrator = self.integrator if integrator is None else integrator
        with torch.no_grad():
            batch = integrator.sample(
                n_samples,
                batch_size=100_000,
                channel_weight_mode="variance",
                evaluate_integrand=False,
            )
        return batch.x, batch.channels, batch.weights, batch.q_sample

    def generate_lo_dataset(
        self,
        n_samples: int,
        cs_eval_name: str,
        for_plotting: bool = False,
        from_vegas: bool = False,
    ):
        cs_eval = self.cross_sections[cs_eval_name]
        start_time = get_start_time()
        r, channels, _, _ = self._sample_r_channels(
            n_samples=n_samples, integrator=self.vegas if from_vegas else None
        )
        p_born, weights_born, _ = self.madnis_phase_space._lo_ps_from_r(r, channels)
        chunk_size = 100
        total = p_born.shape[0]
        chunks = [p_born[i : i + chunk_size] for i in range(0, total, chunk_size)]
        cs_eval_result = torch.cat(
            [cs_eval.matrix_element(chunk) for chunk in chunks], dim=0
        )
        print(f"Evaluated {total} events in {time.time() - start_time[0]:.2f} seconds")
        return {
            "p_n": p_born,
            "w_n": weights_born,
            "cs": cs_eval_result,
        }

    def generate_nlo_dataset(
        self,
        n_samples: int,
        cs_eval_name: str,
        for_plotting: bool = False,
        from_vegas: bool = False,
        batch_size: int = 100_000,
        save: bool = False,
    ):
        cs_eval = self.cross_sections[cs_eval_name]
        start_time = get_start_time()
        if self.params.get("fade_in", False):
            self.cross_sections[cs_eval_name].real_weight = 1.0

        first = True
        out_results_dict: dict[str] = {}
        list_accumulate_keys: set[str] = set()
        self.madnis_phase_space.clustered_jets_n = None
        self.madnis_phase_space.clustered_jets_np1 = None

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            sl = slice(start, end)
            size = end - start

            r, channels, weights, flow_weights = self._sample_r_channels(
                n_samples=size, integrator=self.vegas if from_vegas else None
            )

            (
                rad_vars,
                p_n,
                p_np1,
                p_soft,
                p_coll,
                k_ct,
                fks_sec,
            ), _ = self.madnis_phase_space._nlo_ps_from_r(
                r,
                channels,
                prev_clustered_jets_n=self.madnis_phase_space.clustered_jets_n,
                prev_clustered_jets_np1=self.madnis_phase_space.clustered_jets_np1,
            )
            nlo_inputs = [
                rad_vars,
                p_n,
                p_np1,
                p_soft,
                p_coll,
                k_ct,
                fks_sec,
            ]
            total = p_n[0].shape[0]
            if not for_plotting:
                cs_eval_result = cs_eval.matrix_element(
                    nlo_inputs, pieces=True
                )  # this is a dict containing all matrix element pieces (no cross sections)
                cs_eval_result_dict = cs_eval_result
            else:
                # if for plotting only throw away directly from the final dictionary all the kinematics; keep only the weights
                cs_eval_result_dict = cs_eval.matrix_element(
                    nlo_inputs,
                    for_plotting=True,
                    madnis_weights=flow_weights,
                )  # this is a dict containing all matrix element pieces (no cross sections), in particular it is already concatenated
                cs_eval_result = cs_eval_result_dict.get(
                    "total_weight", cs_eval_result_dict
                )

            if first:
                # allocate full-size containers
                for k, v in cs_eval_result_dict.items():
                    if torch.is_tensor(v):
                        out_results_dict[k] = torch.empty(
                            (n_samples,) + v.shape[1:],
                            device=v.device,
                            dtype=v.dtype,
                        )
                    else:
                        out_results_dict[k] = []
                        list_accumulate_keys.add(k)
                first = False

            for k, v in cs_eval_result_dict.items():
                if k in list_accumulate_keys:
                    out_results_dict[k].append(v)
                else:
                    out_results_dict[k][sl] = v
            print(
                f"Evaluated {total} events in {time.time() - start_time[0]:.2f} seconds"
            )
            if "ref_cross_section" in self.params:
                cs_ref_name = self.params["ref_cross_section"]
                cs_ref = self.cross_sections[cs_ref_name]
                self.integrator.integrand = self.madnis_phase_space.build_integrand(
                    cs_ref
                )
                cs_eval_result_ref = cs_ref(nlo_inputs)
                if not self.seed:
                    print(
                        "No seed specified, weights for reference cross section may differ!"
                    )
                else:
                    torch.manual_seed(self.seed)
                _, _, weights_ref, _ = self._sample_r_channels(n_samples)
            if save:
                base_dict = {
                    "rad_vars": rad_vars,
                    "p_n": p_n,
                    "p_np1": p_np1,
                    "p_soft": p_soft,
                    "p_coll": p_coll,
                    "p_soft_ct": k_ct,
                    "fks_sec": fks_sec,
                    "weights": weights,
                    "cs": cs_eval_result,
                }

        if "ref_cross_section" in self.params and save:
            base_dict.update(
                {
                    "weights_compare": weights_ref,
                    "cs_compare": cs_eval_result_ref,
                }
            )
            return base_dict
        else:
            for k in list_accumulate_keys:
                batches = out_results_dict[k]
                if batches and isinstance(batches[0], tuple):
                    out_results_dict[k] = tuple(
                        torch.cat([b[i] for b in batches], dim=0)
                        for i in range(len(batches[0]))
                    )
                else:
                    out_results_dict[k] = torch.cat(batches, dim=0)
            return out_results_dict

    def generate_dataset(
        self, save: bool = True, n_samples: int = None, from_vegas: bool = False
    ):
        cs_eval_name = self.params["eval_cross_section"]
        n_samples = self.params["n_samples"] if n_samples is None else n_samples
        print(
            f"Generating dataset with {n_samples} samples for cross section '{cs_eval_name}'"
        )
        if save:
            if "nlo" in cs_eval_name or "nlo_surr" in cs_eval_name:
                data = self.generate_nlo_dataset(n_samples, cs_eval_name, save=save)
                name = "nlo_dataset.npy"
            else:
                data = self.generate_lo_dataset(n_samples, cs_eval_name)
                name = "lo_dataset.npy"
            np.save(self.doc.add_file(name), data)
            print(f"Saved {name}")
        else:
            if "nlo" in cs_eval_name or "nlo_surr" in cs_eval_name:
                data = self.generate_nlo_dataset(
                    n_samples,
                    cs_eval_name,
                    for_plotting=True,
                    from_vegas=from_vegas,
                    save=save,
                )
            else:
                data = self.generate_lo_dataset(
                    n_samples,
                    cs_eval_name,
                    for_plotting=True,
                    from_vegas=from_vegas,
                )
            return data

    def integrate(self):
        n_samples = self.params["n_samples"]
        cs_eval_name = self.params["eval_cross_section"]
        if self.params.get("fade_in", False):
            self.cross_sections[cs_eval_name].real_weight = 1.0
        cs_eval = self.cross_sections[cs_eval_name]
        raw_integrand = self.madnis_phase_space.build_integrand(cs_eval)
        timed = TimedIntegrand(raw_integrand)
        self.integrator.integrand = timed
        start_time = get_start_time()
        if self.train_flow:
            BS = int(100e3)
            print(
                f"Calling integrand '{cs_eval_name}' for {n_samples} samples. Expect {n_samples//BS} integrand batchings of size {BS} below."
            )
            metrics = self.integrator.integration_metrics(n_samples, batch_size=BS)
        else:
            self.vegas.integrand = timed
            metrics = self.vegas.integration_metrics(n_samples, batch_size=100_000)
        print_run_time(start_time)
        print_metrics(cs_eval_name, metrics, integrand_time=timed.total_time)

        if "ref_cross_section" in self.params:
            cs_ref_name = self.params["ref_cross_section"]
            cs_ref = self.cross_sections[cs_ref_name]
            raw_integrand = self.madnis_phase_space.build_integrand(cs_ref)
            timed = TimedIntegrand(raw_integrand)
            self.integrator.integrand = timed
            print(f"Calling integrand '{cs_ref_name}'")
            start_time = get_start_time()
            if self.train_flow:
                metrics = self.integrator.integration_metrics(
                    n_samples, batch_size=100_000
                )
            else:
                self.vegas.integrand = timed
                metrics = self.vegas.integration_metrics(n_samples, batch_size=100_000)
            print_run_time(start_time)
            print_metrics(cs_ref_name, metrics, integrand_time=timed.total_time)

    def make_plots(self):
        params = self.params["plots"]
        print("Plotting...")
        if self.vegas and not self.train_flow:
            weights_dict = self.generate_dataset(
                save=False, n_samples=params["n_samples"], from_vegas=True
            )
        else:
            weights_dict = self.generate_dataset(
                save=False, n_samples=params["n_samples"]
            )
        jets_n = tensors_to_numpy(self.madnis_phase_space.clustered_jets_n)
        if hasattr(self.madnis_phase_space, "clustered_jets_np1"):
            jets_np1 = tensors_to_numpy(self.madnis_phase_space.clustered_jets_np1)
        else:
            jets_np1 = None
        pair_list = [
            ["true", "none"],
        ]
        print(f"    Plotting losses")
        plot_train_metrics(
            file=self.doc.add_file("train_metrics.pdf", add_run_name=False),
            train_results=getattr(self, "train_results", None),
            vegas_train_results=getattr(self, "vegas_train_results", None),
            data_file=self.doc.add_file("train_metrics.pkl", add_run_name=False),
            logy_losses=True,
        )
        print(f"    Plotting FKS sectors")
        plot_fks_sectors(
            fks_sectors=weights_dict.get("fks_sector", None),
            file=self.doc.add_file("fks_sectors.pdf", add_run_name=False),
            lines_file=self.doc.add_file("fks_sectors.pkl", add_run_name=False),
            weights=np.abs(weights_dict['real_subtracted_weight']),
        )
        if params.get("surrogates", False):
            # check that 'surrogate' exists in at least one weights dict key, otherwise skip surrogate plots
            if not any("surrogate" in key for key in weights_dict.keys()):
                print(
                    f"[WARNING] Surrogate plots requested but no surrogate weights found, skipping surrogate plots."
                )
            else:
                pair_list.append(["V_surrogate", "virtual"])
                pair_list.append(["VR_surrogate", "virtual_and_real"])
        sorted_jets_n = None
        sorted_jets_np1 = None
        for combination in pair_list:
            name = combination[0]
            surr_to_use = combination[1]
            plots = Plots(
                jets_n=jets_n,
                jets_np1=jets_np1,
                weights_dict=weights_dict,
                n_particles=len(self.madnis_phase_space.outgoing_pids),
                surrogates_to_use=surr_to_use,
                model_name="VEGAS" if (not self.train_flow and self.vegas) else None,
                debug=False,
                sorted_jets_n=sorted_jets_n,
                sorted_jets_np1=sorted_jets_np1,
            )
            sorted_jets_n = plots.jets_n_PtPhiEtaM
            sorted_jets_np1 = plots.jets_np1_PtPhiEtaM

            save_samples = params.get("save_samples", False)
            if save_samples:
                print(f"    Saving samples")
            sample_kw = lambda tag: dict(
                file=self.doc.add_file(f"{tag}_{name}.pdf", add_run_name=False),
                lines_file=self.doc.add_file(f"{tag}_{name}.pkl", add_run_name=False),
                sample_file=self.doc.add_file(
                    f"samples_{tag}_{name}.pkl", add_run_name=False
                )
                if save_samples
                else None,
            )
            plots.plot_jet_pt(**sample_kw("pt"))
            plots.plot_jet_eta(**sample_kw("eta"))
            plots.plot_jet_dphi(**sample_kw("dphi"))
            plots.plot_jet_deta(**sample_kw("deta"))
            plots.plot_jet_dR(**sample_kw("dR"))
            plots.plot_jet_mjj(**sample_kw("mjj"))
