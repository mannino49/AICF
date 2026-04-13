"""
Dynamical Layer — ψ_D(t)
=========================

Implements the whole-brain complexity factor as defined in
formalism_repair_v1.md Section 4.

Three sub-quantities capture distinct aspects of neural dynamics:

    h(t)         — entropy rate (irreducible unpredictability)
    C_D(t)       — dynamical complexity (excess entropy / statistical complexity)
    A_modal(t)   — modal agility (transition entropy of mode-switching)

Combined (after z-scoring and logistic sigmoid normalization):

    ψ_D(t) = w_h · σ(z_h) + w_C · σ(z_C) + w_agil · σ(z_A)

Each sub-quantity is estimated over a sliding window in time.

Reference
---------
Hancock, P., Kee, S., Rosas, F., Girn, M., Kotler, S., Mannino, M., &
    Huskey, R. (2025). Dynamical complexity signatures of flow.
formalism_repair_v1.md §4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # logistic sigmoid


class DynamicalLayer:
    """Compute the dynamical complexity factor ψ_D(t).

    Parameters
    ----------
    w_h : float, optional
        Weight on entropy rate h(t).  Default 1/3.
    w_C : float, optional
        Weight on dynamical complexity C_D(t).  Default 1/3.
    w_agil : float, optional
        Weight on modal agility A_modal(t).  Default 1/3.
        All three weights must sum to 1.

    Raises
    ------
    ValueError
        If weights are negative or do not sum to 1.

    Notes
    -----
    Implements formalism_repair_v1.md §4.
    """

    def __init__(
        self,
        w_h: float = 1.0 / 3.0,
        w_C: float = 1.0 / 3.0,
        w_agil: float = 1.0 / 3.0,
    ) -> None:
        weights = np.array([w_h, w_C, w_agil], dtype=float)
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"w_h + w_C + w_agil must equal 1; got {weights.sum():.6f}"
            )
        self.w_h = float(w_h)
        self.w_C = float(w_C)
        self.w_agil = float(w_agil)

    # ------------------------------------------------------------------
    # Sub-quantity estimators
    # ------------------------------------------------------------------

    def compute_entropy_rate(
        self,
        time_series: NDArray[np.float64],
        window_size: int,
    ) -> NDArray[np.float64]:
        """Estimate the entropy rate over a sliding window.

        The entropy rate h(t) measures the irreducible unpredictability
        of the neural time series (formalism §4.1a):

            h(t) = lim_{n→∞} H(X_n | X_{n-1}, ..., X_1)

        **Practical estimator**: we approximate h using the Lempel-Ziv (LZ76)
        complexity normalized by sequence length, which provides a computable
        upper bound on entropy rate for finite time series.  For a binary
        sequence of length n, LZ76 complexity c(n) relates to entropy rate as:

            h_LZ ≈ c(n) · log(n) / n

        For continuous (multi-channel) data, each sample is binarized by
        comparison to the median across the window before LZ computation.

        Parameters
        ----------
        time_series : 2-D array of float, shape (n_channels, n_timepoints)
            Multivariate neural time series.
        window_size : int
            Number of time points per sliding window.

        Returns
        -------
        1-D array of float
            Entropy rate estimate at each window position (length =
            n_timepoints − window_size + 1).
        """
        time_series = np.asarray(time_series, dtype=float)
        if time_series.ndim == 1:
            time_series = time_series[np.newaxis, :]
        if time_series.ndim != 2:
            raise ValueError(
                f"time_series must be 2-D (n_channels × n_timepoints); "
                f"got shape {time_series.shape}"
            )
        n_channels, n_tp = time_series.shape
        if window_size < 4:
            raise ValueError(f"window_size must be ≥ 4; got {window_size}")
        if window_size > n_tp:
            raise ValueError(
                f"window_size ({window_size}) exceeds n_timepoints ({n_tp})"
            )

        n_windows = n_tp - window_size + 1
        h_estimates = np.empty(n_windows)

        for w in range(n_windows):
            segment = time_series[:, w : w + window_size]  # (n_ch, win)
            # Binarize each channel: 1 if above channel median, else 0
            medians = np.median(segment, axis=1, keepdims=True)
            binary = (segment > medians).astype(int)  # (n_ch, win)
            # Flatten channels: concatenate into a single binary sequence
            flat = binary.ravel()
            h_estimates[w] = self._lempel_ziv_entropy_rate(flat)

        return h_estimates

    def compute_dynamical_complexity(
        self,
        time_series: NDArray[np.float64],
        window_size: int,
    ) -> NDArray[np.float64]:
        """Estimate dynamical complexity (excess entropy) over sliding windows.

        The dynamical complexity C_D(t) is the mutual information between the
        past and future of the process (formalism §4.1b):

            C_D(t) = I(X_past; X_future)

        **Practical estimator**: for each window we split the segment into
        equal-length past and future halves, binarize, compute joint and
        marginal entropies, and estimate I = H_past + H_future − H_joint.

        Parameters
        ----------
        time_series : 2-D array of float, shape (n_channels, n_timepoints)
            Multivariate neural time series.
        window_size : int
            Number of time points per sliding window (must be even so that
            equal past/future halves can be formed).

        Returns
        -------
        1-D array of float
            Dynamical complexity estimate at each window position (nats).
        """
        time_series = np.asarray(time_series, dtype=float)
        if time_series.ndim == 1:
            time_series = time_series[np.newaxis, :]
        if time_series.ndim != 2:
            raise ValueError(
                f"time_series must be 2-D; got shape {time_series.shape}"
            )
        n_channels, n_tp = time_series.shape
        if window_size < 4:
            raise ValueError(f"window_size must be ≥ 4; got {window_size}")
        if window_size % 2 != 0:
            raise ValueError(
                f"window_size must be even for equal past/future split; "
                f"got {window_size}"
            )
        if window_size > n_tp:
            raise ValueError(
                f"window_size ({window_size}) exceeds n_timepoints ({n_tp})"
            )

        half = window_size // 2
        n_windows = n_tp - window_size + 1
        C_estimates = np.empty(n_windows)

        for w in range(n_windows):
            segment = time_series[:, w : w + window_size]
            median = np.median(segment, axis=1, keepdims=True)
            binary = (segment > median).astype(int)

            past = binary[:, :half].ravel()    # concatenated past half
            future = binary[:, half:].ravel()  # concatenated future half

            H_past = self._empirical_entropy(past)
            H_future = self._empirical_entropy(future)

            # Joint entropy via concatenation (approximation)
            joint_codes = past * 2 + future  # crude pairing
            H_joint = self._empirical_entropy(joint_codes)

            MI = max(0.0, H_past + H_future - H_joint)  # bits, clamp at 0
            C_estimates[w] = MI

        return C_estimates

    def compute_modal_agility(
        self,
        state_sequence: NDArray[np.int_],
        window_size: int,
    ) -> NDArray[np.float64]:
        """Estimate modal agility (transition entropy) over sliding windows.

        Modal agility A_modal(t) is the entropy of the empirical transition
        matrix within each window (formalism §4.1c):

            A_modal = −Σ_{j,k} T_{jk} · log(T_{jk})

        where T_{jk} is the empirical probability of transitioning from
        state j to state k.  Higher values → more fluid, less stereotyped
        mode-switching.

        Parameters
        ----------
        state_sequence : 1-D array of int
            Sequence of discrete states (modes) over time.  States are
            assumed to be non-negative integers 0, 1, ..., n_states-1.
        window_size : int
            Number of time points per sliding window.

        Returns
        -------
        1-D array of float
            Modal agility estimate at each window position (nats).
        """
        states = np.asarray(state_sequence, dtype=int)
        if states.ndim != 1:
            raise ValueError(
                f"state_sequence must be 1-D; got shape {states.shape}"
            )
        if window_size < 2:
            raise ValueError(f"window_size must be ≥ 2; got {window_size}")
        if window_size > len(states):
            raise ValueError(
                f"window_size ({window_size}) exceeds sequence length ({len(states)})"
            )

        n_tp = len(states)
        n_windows = n_tp - window_size + 1
        n_states = states.max() + 1
        agility = np.empty(n_windows)

        for w in range(n_windows):
            window = states[w : w + window_size]
            T = self._empirical_transition_matrix(window, n_states)
            # Entropy of the transition matrix (treating T as a distribution)
            flat = T.ravel()
            mask = flat > 0
            agility[w] = float(-np.sum(flat[mask] * np.log(flat[mask])))

        return agility

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(
        raw_values: NDArray[np.float64],
        mean: float | None = None,
        std: float | None = None,
    ) -> NDArray[np.float64]:
        """Z-score then apply logistic sigmoid to map to (0, 1).

        Implements formalism §4.2:
            σ(z_x(t)) where z_x = (x − μ_x) / σ_x

        Session-level statistics (mean, std) can be provided for
        consistency across windows; if None, computed from raw_values.

        Parameters
        ----------
        raw_values : 1-D array of float
            Raw sub-quantity values across time.
        mean : float, optional
            Pre-computed session mean.  If None, computed from raw_values.
        std : float, optional
            Pre-computed session standard deviation.  If None, computed
            from raw_values.

        Returns
        -------
        1-D array of float in (0, 1)
            Normalized values.
        """
        raw = np.asarray(raw_values, dtype=float)
        if mean is None:
            mean = float(raw.mean())
        if std is None:
            std = float(raw.std())
        if std < 1e-12:
            # Constant signal: all values map to sigmoid(0) = 0.5
            return np.full_like(raw, 0.5)
        z = (raw - mean) / std
        return expit(z).astype(float)

    # ------------------------------------------------------------------
    # Top-level factor
    # ------------------------------------------------------------------

    def compute(
        self,
        h_raw: float | NDArray[np.float64],
        C_raw: float | NDArray[np.float64],
        agility_raw: float | NDArray[np.float64],
        session_stats: dict | None = None,
    ) -> float | NDArray[np.float64]:
        """Compute ψ_D(t) from raw sub-quantity values.

        ψ_D = w_h · σ(z_h) + w_C · σ(z_C) + w_agil · σ(z_A)

        Parameters
        ----------
        h_raw : float or 1-D array
            Raw entropy rate estimate(s).
        C_raw : float or 1-D array
            Raw dynamical complexity estimate(s).
        agility_raw : float or 1-D array
            Raw modal agility estimate(s).
        session_stats : dict, optional
            Pre-computed normalization statistics with keys
            'h_mean', 'h_std', 'C_mean', 'C_std', 'agil_mean', 'agil_std'.
            If None, statistics are computed from the inputs.

        Returns
        -------
        float or 1-D array in [0, 1]
            ψ_D value(s).
        """
        scalar_input = np.isscalar(h_raw)
        h = np.atleast_1d(np.asarray(h_raw, dtype=float))
        C = np.atleast_1d(np.asarray(C_raw, dtype=float))
        agil = np.atleast_1d(np.asarray(agility_raw, dtype=float))

        s = session_stats or {}
        h_norm = self.normalize(h, s.get("h_mean"), s.get("h_std"))
        C_norm = self.normalize(C, s.get("C_mean"), s.get("C_std"))
        agil_norm = self.normalize(agil, s.get("agil_mean"), s.get("agil_std"))

        psi_D = self.w_h * h_norm + self.w_C * C_norm + self.w_agil * agil_norm
        # Clip to handle floating-point drift (weights already sum to 1 and
        # each σ output ∈ (0, 1), so sum is always in (0, 1) analytically)
        psi_D = np.clip(psi_D, 0.0, 1.0)

        return float(psi_D[0]) if scalar_input else psi_D

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lempel_ziv_entropy_rate(binary_sequence: NDArray[np.int_]) -> float:
        """LZ76 complexity-based entropy rate estimate (nats/symbol).

        Normalizes LZ76 complexity c(n) as: h_LZ = c(n) · log(n) / n

        This is an upper bound on the true entropy rate for i.i.d. sources
        and converges asymptotically for stationary ergodic sources.
        """
        seq = binary_sequence.ravel()
        n = len(seq)
        if n < 2:
            return 0.0

        # LZ76 algorithm: count number of distinct substrings
        complexity = 1
        i = 0
        k = 1
        k_max = 1
        while True:
            if seq[i + k - 1] == seq[complexity - 1]:
                k += 1
                if i + k - 1 > n - 1:
                    complexity += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == complexity:
                    complexity += 1
                    if complexity + k_max - 1 > n - 1:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1

        # LZ76 normalization
        c = float(complexity)
        h_lz = c * np.log(n) / n
        return float(h_lz)

    @staticmethod
    def _empirical_entropy(sequence: NDArray[np.int_]) -> float:
        """Shannon entropy of an empirical sequence (nats)."""
        seq = np.asarray(sequence)
        _, counts = np.unique(seq, return_counts=True)
        probs = counts / counts.sum()
        mask = probs > 0
        return float(-np.sum(probs[mask] * np.log(probs[mask])))

    @staticmethod
    def _empirical_transition_matrix(
        sequence: NDArray[np.int_], n_states: int
    ) -> NDArray[np.float64]:
        """Compute the row-normalized empirical transition matrix.

        T[j, k] = P(next state = k | current state = j)

        Rows with zero counts are set to a uniform distribution to avoid
        division by zero.
        """
        T = np.zeros((n_states, n_states), dtype=float)
        for t in range(len(sequence) - 1):
            j = sequence[t]
            k = sequence[t + 1]
            T[j, k] += 1.0

        row_sums = T.sum(axis=1, keepdims=True)
        # Rows with no transitions: set to uniform (handle degenerate windows)
        zero_rows = (row_sums.ravel() == 0)
        T[zero_rows] = 1.0 / n_states
        row_sums[zero_rows] = 1.0
        T = T / row_sums
        return T
