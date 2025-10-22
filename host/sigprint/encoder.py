from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OmegaState:
    """
    Ω-field state representation (single window)

    psi: per-channel amplitude summary (mean envelope)
    lambda_: per-channel phase summary (circular mean angle, radians)
    coherence: global spatial ∫ψ² dV proxy (mean of A_norm**2)
    plv: mean pairwise phase-locking value across channels
    entropy: normalized phase entropy across all samples/channels (0..1)
    timestamp: unix epoch seconds
    """

    psi: Dict[str, float]
    lambda_: Dict[str, float]
    coherence: float
    plv: float
    entropy: float
    timestamp: float


class SigprintEncoder:
    """
    Encodes Ω-field state from EEG into a 20-digit SIGPRINT code.

    Design choices:
      - Lock-in demodulation at center_hz using quadrature (I/Q) and a moving-average LPF
      - Envelope A ≈ ψ, instantaneous phase φ ≈ Λ
      - Robust per-channel normalization (divide by median envelope)
      - Spatial coherence and phase synchrony from envelope/phase networks
      - Deterministic quantization into 20 digits for a compact signature

    Dependencies: numpy only (no SciPy required)
    """

    def __init__(
        self,
        fs: float,
        center_hz: float = 8.0,
        lpf_seconds: float = 0.5,
        window_seconds: Optional[float] = None,
    ) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        if lpf_seconds <= 0:
            raise ValueError("lpf_seconds must be positive")
        self.fs = float(fs)
        self.center_hz = float(center_hz)
        self.lpf_seconds = float(lpf_seconds)
        self.window_seconds = window_seconds

    # ----------------------- Core signal operations ----------------------- #

    def _common_avg_ref(self, x: np.ndarray) -> np.ndarray:
        """Apply common average reference across channels."""
        return x - x.mean(axis=0, keepdims=True)

    def _lock_in(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lock-in demodulation around `center_hz`.

        Inputs:
          x: [n_channels, n_samples] real-valued EEG

        Returns:
          A: [n_channels, n_samples] envelope (>=0)
          phi: [n_channels, n_samples] instantaneous phase (radians)
        """
        n_channels, n_samples = x.shape
        t = np.arange(n_samples, dtype=np.float64) / self.fs
        omega = 2.0 * math.pi * self.center_hz
        cos_ref = np.cos(omega * t)
        sin_ref = np.sin(omega * t)

        # Mix to baseband
        i_mix = x * cos_ref  # [C,T]
        q_mix = x * sin_ref

        # Low-pass via moving average
        M = max(1, int(round(self.lpf_seconds * self.fs)))
        if M > n_samples:
            M = n_samples
        kernel = np.ones(M, dtype=np.float64) / float(M)

        # Convolve along time per channel
        # 'same' to maintain shape; group delay exists but irrelevant for window stats
        i_lp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, i_mix)
        q_lp = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, q_mix)

        # Envelope and phase
        A = np.hypot(i_lp, q_lp)
        phi = np.arctan2(q_lp, i_lp)
        return A, phi

    @staticmethod
    def _circular_mean_phase(phi: np.ndarray) -> float:
        """Return circular mean angle of a 1D array of phases (radians)."""
        z = np.exp(1j * phi)
        m = np.mean(z)
        return float(np.angle(m))

    @staticmethod
    def _pairwise_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.triu_indices(n, k=1)

    @staticmethod
    def _quantize(value: float, lo: float, hi: float) -> int:
        y = (np.clip(value, lo, hi) - lo) / (hi - lo + 1e-12)
        d = int(np.floor(10 * float(y) + 1e-9))
        return min(max(d, 0), 9)

    # ----------------------- Public encode API ----------------------- #

    def encode(
        self,
        data: np.ndarray,
        ch_names: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[str, OmegaState, Dict[str, float]]:
        """
        Encode a single window of EEG as a 20-digit SIGPRINT code with Ω semantics.

        Inputs:
          data: ndarray [n_channels, n_samples]
          ch_names: optional names per channel (len == n_channels)
          timestamp: unix epoch seconds; if None, uses time.time()

        Returns:
          (code_str, omega_state, features_dict)
        """
        if data.ndim != 2:
            raise ValueError("data must be 2D [channels, samples]")
        n_channels, n_samples = data.shape
        if n_channels < 1 or n_samples < 2:
            raise ValueError("data has insufficient channels or samples")
        if ch_names is None:
            ch_names = [f"C{i+1}" for i in range(n_channels)]
        if len(ch_names) != n_channels:
            raise ValueError("len(ch_names) must equal n_channels")

        # Preprocess: remove DC per channel, common average reference
        x = data - data.mean(axis=1, keepdims=True)
        x = self._common_avg_ref(x)

        # Lock-in
        A, phi = self._lock_in(x)

        # Per-channel envelope normalization (robust) to reduce gain sensitivity
        med = np.median(A, axis=1, keepdims=True) + 1e-12
        A_norm = A / med

        # Omega per-channel summaries
        psi_map: Dict[str, float] = {}
        lambda_map: Dict[str, float] = {}
        for idx, name in enumerate(ch_names):
            psi_map[name] = float(np.mean(A_norm[idx]))
            lambda_map[name] = self._circular_mean_phase(phi[idx])

        # Global metrics
        coherence = float(np.mean(A_norm ** 2))

        # Pairwise PLV across channels
        ii, jj = self._pairwise_indices(n_channels)
        if ii.size:
            dphi = np.unwrap(phi[ii], axis=1) - np.unwrap(phi[jj], axis=1)
            plv_pairs = np.abs(np.mean(np.exp(1j * dphi), axis=1))
            plv_mean = float(np.mean(plv_pairs))
            plv_median = float(np.median(plv_pairs))
            plv_max = float(np.max(plv_pairs))
        else:
            plv_pairs = np.array([], dtype=np.float64)
            plv_mean = plv_median = plv_max = 0.0

        # Kuramoto order parameter over time
        R_t = np.abs(np.mean(np.exp(1j * phi), axis=0))
        R_mean = float(np.mean(R_t))
        R_p90 = float(np.percentile(R_t, 90))
        R_std = float(np.std(R_t))

        # Circular variance across all channels/samples
        circ_var = float(1.0 - np.abs(np.mean(np.exp(1j * phi))))

        # Phase distribution entropy (normalized 0..1)
        hist, _ = np.histogram(phi, bins=36, range=(-np.pi, np.pi), density=True)
        p = hist / (np.sum(hist) + 1e-12)
        ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12)
        phase_uniformity = float(1.0 - ent)  # 0 uniform → 1 peaked

        # Envelope network correlations
        Ac = (A_norm - A_norm.mean(axis=1, keepdims=True)) / (A_norm.std(axis=1, keepdims=True) + 1e-12)
        corr = (Ac @ Ac.T) / (Ac.shape[1] - 1)
        corr = np.clip(corr, -1.0, 1.0)
        off = corr[~np.eye(n_channels, dtype=bool)]
        r_mean = float(off.mean()) if off.size else 0.0
        r_median = float(np.median(off)) if off.size else 0.0
        r_max = float(np.max(off)) if off.size else 0.0
        try:
            w = np.linalg.eigvalsh(corr)
            lambda1_norm = float(np.max(w) / max(n_channels, 1))
        except np.linalg.LinAlgError:
            lambda1_norm = 0.0

        # Envelope stats
        a = A_norm.reshape(-1)
        a_mean = float(np.mean(a))
        a_std = float(np.std(a))
        a_p10 = float(np.percentile(a, 10))
        a_median = float(np.median(a))
        a_p90 = float(np.percentile(a, 90))

        # Temporal variability of mean envelope across channels
        mean_env_t = A_norm.mean(axis=0)
        mean_env_t_std = float(np.std(mean_env_t))

        # Quantization into 20-digit SIGPRINT
        feats = [
            a_mean, a_std, a_p10, a_median, a_p90,  # 1–5
            r_mean, r_median, r_max, lambda1_norm,  # 6–9
            plv_mean, plv_median, plv_max,          # 10–12
            R_mean, R_p90, circ_var,                # 13–15
            float(np.mean(np.cos(phi))),            # 16 mean cos φ
            float(np.mean(np.sin(phi))),            # 17 mean sin φ
            phase_uniformity,                       # 18 1 - phase entropy
            R_std, mean_env_t_std                   # 19–20
        ]

        ranges = [
            (0.5, 1.5), (0.0, 0.5), (0.3, 1.0), (0.6, 1.4), (0.8, 2.0),  # 1–5
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.2, 1.0),              # 6–9
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),                          # 10–12
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),                          # 13–15
            (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0),                        # 16–18
            (0.0, 0.5), (0.0, 0.5),                                      # 19–20
        ]

        digits = [self._quantize(v, lo, hi) for v, (lo, hi) in zip(feats, ranges)]
        code = ''.join(str(d) for d in digits)

        features = {
            # Envelope stats
            "a_mean": a_mean, "a_std": a_std, "a_p10": a_p10, "a_median": a_median, "a_p90": a_p90,
            # Envelope corr network
            "r_mean": r_mean, "r_median": r_median, "r_max": r_max, "lambda1_norm": lambda1_norm,
            # Phase synchrony
            "plv_mean": plv_mean, "plv_median": plv_median, "plv_max": plv_max,
            # Kuramoto
            "R_mean": R_mean, "R_p90": R_p90, "R_std": R_std,
            # Phase distribution
            "circular_variance": circ_var, "phase_uniformity": phase_uniformity,
            "mean_cos_phi": float(np.mean(np.cos(phi))),
            "mean_sin_phi": float(np.mean(np.sin(phi))),
            # Temporal envelope variability
            "mean_env_t_std": mean_env_t_std,
            # Ω-field summaries
            "coherence": coherence,
        }

        omega = OmegaState(
            psi=psi_map,
            lambda_=lambda_map,
            coherence=coherence,
            plv=plv_mean,
            entropy=float(1.0 - phase_uniformity),  # normalized entropy 0..1
            timestamp=float(time.time() if timestamp is None else timestamp),
        )

        return code, omega, features
