from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # optional SciPy for nicer low-pass
    from scipy import signal as _sp_signal  # type: ignore
except Exception:  # pragma: no cover - optional
    _sp_signal = None

from .encoder import OmegaState


def _moving_average_kernel(fs: float, cutoff_hz: float) -> np.ndarray:
    """Return a moving-average kernel length approximating a cutoff.

    MA 3dB cutoff ~ 0.443 * Fs / M → M ~ 0.443 * Fs / fc
    Ensure at least 3 taps.
    """
    M = max(3, int(round(0.443 * fs / max(cutoff_hz, 1e-6))))
    kernel = np.ones(M, dtype=np.float64) / float(M)
    return kernel


class StructuredSigprintEncoder:
    """
    20-digit structured SIGPRINT encoder with Ω-field semantics.

    Layout (total 20 digits):
      1–4   Phase topology (front-back, left-right) 2 digits each
      5–8   Amplitude distribution (frontal%, left%) 2 digits each
      9–12  Coherence (global 0-99) + PLV (0-99)
      13–18 Reserved: entropy (0-99), stylus stage (00..06), reserved 00
      19–20 Checksum of payload (SHA-256 mod 100)

    Operates on 1-second epochs per initialization.
    Uses lock-in demodulation at `lockin_freq` to estimate ψ (amplitude) and Λ (phase).
    """

    def __init__(
        self,
        channel_names: List[str],
        sample_rate: float = 250.0,
        lockin_freq: float = 8.0,
        gate_threshold: int = 8,
        amp_gain: float = 1.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if lockin_freq <= 0:
            raise ValueError("lockin_freq must be positive")
        if not channel_names:
            raise ValueError("channel_names required")

        self.channel_names = list(channel_names)
        self.sample_rate = float(sample_rate)
        self.lockin_freq = float(lockin_freq)
        self.gate_threshold = int(gate_threshold)
        self.amp_gain = float(amp_gain)

        # Precompute reference tones for a 1-second epoch
        self.epoch_length = int(round(self.sample_rate))
        t = np.arange(self.epoch_length, dtype=np.float64) / self.sample_rate
        w = 2.0 * np.pi * self.lockin_freq
        self.ref_sin = np.sin(w * t)
        self.ref_cos = np.cos(w * t)

        # Low-pass for I/Q (target ~2 Hz)
        self._use_scipy = _sp_signal is not None
        if self._use_scipy:
            nyq = self.sample_rate / 2.0
            cutoff = 2.0 / nyq
            self._lp_b, self._lp_a = _sp_signal.butter(4, cutoff, btype="low")
        else:
            self._lp_kernel = _moving_average_kernel(self.sample_rate, cutoff_hz=2.0)

        # State tracking
        self.prev_signature: Optional[str] = None
        self.prev_omega: Optional[OmegaState] = None
        self.gate_count = 0
        self.loop_count = 0

    # --------------------------- Public API --------------------------- #

    def process_epoch(
        self,
        eeg_epoch: Dict[str, np.ndarray],
        stylus_context: Optional[Dict] = None,
    ) -> Tuple[str, OmegaState]:
        """
        Process a 1-second epoch (dict of channel -> samples) and return (SIGPRINT, Ω-state).
        The input signal per channel is trimmed/padded to 1 second at `sample_rate`.
        """
        omega_state = self._compute_omega_state(eeg_epoch)

        # Encode 20-digit payload
        parts: List[str] = []
        parts.append(self._encode_phase_topology(omega_state))  # 4 digits
        parts.append(self._encode_amplitude_distribution(omega_state))  # 4 digits
        parts.append(self._encode_coherence(omega_state))  # 4 digits
        parts.append(self._encode_reserved(omega_state, stylus_context))  # 6 digits
        payload = ''.join(parts)
        checksum = self._compute_checksum(payload)  # 2 digits
        sig = payload + checksum

        # Transition detection
        if self.prev_signature is not None:
            transition_type = self._detect_transition(sig)
            setattr(omega_state, "transition", transition_type)
        else:
            setattr(omega_state, "transition", "INIT")

        # Update state
        self.prev_signature = sig
        self.prev_omega = omega_state
        return sig, omega_state

    # ------------------------ Ω-field extraction ---------------------- #

    def _compute_omega_state(self, eeg_epoch: Dict[str, np.ndarray]) -> OmegaState:
        psi: Dict[str, float] = {}
        lambda_map: Dict[str, float] = {}

        for ch in self.channel_names:
            sig = np.asarray(eeg_epoch.get(ch, np.zeros(self.epoch_length, dtype=np.float64)), dtype=np.float64)

            # Trim/pad to epoch length
            if sig.size != self.epoch_length:
                if sig.size > self.epoch_length:
                    sig = sig[: self.epoch_length]
                else:
                    sig = np.pad(sig, (0, self.epoch_length - sig.size))

            # Lock-in mix
            I_raw = sig * self.ref_cos
            Q_raw = sig * self.ref_sin

            if self._use_scipy:
                # Zero-phase low-pass; use last sample as DC estimate
                I = _sp_signal.filtfilt(self._lp_b, self._lp_a, I_raw)
                Q = _sp_signal.filtfilt(self._lp_b, self._lp_a, Q_raw)
                I_dc = float(I[-1])
                Q_dc = float(Q[-1])
            else:
                # Moving-average low-pass
                k = self._lp_kernel
                I_lp = np.convolve(I_raw, k, mode="same")
                Q_lp = np.convolve(Q_raw, k, mode="same")
                I_dc = float(I_lp[-1])
                Q_dc = float(Q_lp[-1])

            amplitude = float(np.hypot(I_dc, Q_dc)) * self.amp_gain
            phase_rad = float(np.arctan2(Q_dc, I_dc))

            psi[ch] = amplitude
            lambda_map[ch] = phase_rad  # store in radians

        coherence = self._compute_global_coherence(psi)
        plv = self._compute_phase_locking_value(lambda_map)
        entropy = self._compute_entropy(psi)

        omega = OmegaState(
            psi=psi,
            lambda_=lambda_map,
            coherence=coherence,
            plv=plv,
            entropy=entropy,
            timestamp=time.time(),
        )
        return omega

    @staticmethod
    def _compute_global_coherence(psi: Dict[str, float]) -> float:
        amps = list(psi.values())
        if not amps:
            return 0.0
        total_power = float(sum(a * a for a in amps))
        if total_power <= 0:
            return 0.0
        # Log-style normalizer on 0..100
        val = 100.0 * (1.0 - float(np.exp(-total_power / max(len(amps), 1))))
        return max(0.0, min(100.0, val))

    @staticmethod
    def _compute_phase_locking_value(lambda_map: Dict[str, float]) -> float:
        if not lambda_map:
            return 0.0
        phases = np.array(list(lambda_map.values()), dtype=np.float64)  # radians
        z = np.exp(1j * phases)
        plv = float(np.abs(np.mean(z)))
        return max(0.0, min(1.0, plv))

    @staticmethod
    def _compute_entropy(psi: Dict[str, float]) -> float:
        if not psi:
            return 0.0
        amps = np.array(list(psi.values()), dtype=np.float64)
        total = float(np.sum(amps))
        if total <= 0:
            return 0.0
        p = amps / total
        p = p[p > 0]
        H = float(-np.sum(p * np.log2(p)))
        Hmax = float(np.log2(len(p))) if len(p) > 0 else 1.0
        return 0.0 if Hmax == 0 else float(H / Hmax)

    # ------------------------- Encoders (digits) ---------------------- #

    def _encode_phase_topology(self, omega: OmegaState) -> str:
        # Region groups
        frontal = [ch for ch in self.channel_names if ch.startswith(("Fp", "F"))]
        posterior = [ch for ch in self.channel_names if ch.startswith(("P", "O"))]
        left = [ch for ch in self.channel_names if ch.endswith(("1", "3", "5", "7"))]
        right = [ch for ch in self.channel_names if ch.endswith(("2", "4", "6", "8"))]

        def mean_phase_deg(chs: List[str]) -> float:
            vals = [omega.lambda_.get(ch) for ch in chs if ch in omega.lambda_]
            vals = [v for v in vals if v is not None]
            if not vals:
                return 0.0
            z = np.exp(1j * np.array(vals, dtype=np.float64))
            ang = float(np.degrees(np.angle(np.mean(z))))
            return ang

        phi_front = mean_phase_deg(frontal)
        phi_post = mean_phase_deg(posterior)
        phi_left = mean_phase_deg(left)
        phi_right = mean_phase_deg(right)

        front_back_diff = int(((phi_post - phi_front) % 360.0) / 3.6)  # 0..99
        left_right_diff = int(((phi_right - phi_left) % 360.0) / 3.6)   # 0..99
        return f"{front_back_diff:02d}{left_right_diff:02d}"

    def _encode_amplitude_distribution(self, omega: OmegaState) -> str:
        regions = {
            "frontal": [ch for ch in self.channel_names if ch.startswith(("Fp", "F"))],
            "posterior": [ch for ch in self.channel_names if ch.startswith(("P", "O"))],
            "left": [ch for ch in self.channel_names if ch.endswith(("1", "3", "5", "7"))],
            "right": [ch for ch in self.channel_names if ch.endswith(("2", "4", "6", "8"))],
        }

        power: Dict[str, float] = {}
        for region, chs in regions.items():
            power[region] = float(sum((omega.psi.get(ch, 0.0) ** 2) for ch in chs))

        total = sum(power.values()) or 1.0
        front_ratio = int(100.0 * power["frontal"] / total)
        left_ratio = int(100.0 * power["left"] / total)
        return f"{front_ratio:02d}{left_ratio:02d}"

    @staticmethod
    def _encode_coherence(omega: OmegaState) -> str:
        coh = int(max(0, min(99, int(round(omega.coherence)))))
        plv = int(max(0, min(99, int(round(omega.plv * 100)))))
        return f"{coh:02d}{plv:02d}"

    @staticmethod
    def _encode_reserved(omega: OmegaState, stylus_context: Optional[Dict] = None) -> str:
        ent = int(max(0, min(99, int(round(omega.entropy * 100)))))
        stage = 0
        if stylus_context and isinstance(stylus_context, dict):
            stage = int(stylus_context.get("stage", 0) or 0)
            stage = max(0, min(99, stage))
        return f"{ent:02d}{stage:02d}00"

    @staticmethod
    def _compute_checksum(payload: str) -> str:
        h = hashlib.sha256(payload.encode()).hexdigest()
        cs = int(h, 16) % 100
        return f"{cs:02d}"

    # ------------------------- Transition logic ---------------------- #

    def _detect_transition(self, current_sig: str) -> str:
        if self.prev_signature is None:
            return "INIT"
        dist = sum((a != b) for a, b in zip(self.prev_signature, current_sig))
        if dist >= self.gate_threshold:
            self.gate_count += 1
            return "GATE"
        else:
            self.loop_count += 1
            return "LOOP"


class VoiceJournal:
    """
    Voice journaling synchronized with SIGPRINT (Order-2 loop).

    Requires `speech_recognition` and a working microphone. Imports are lazy.
    """

    def __init__(self, encoder: StructuredSigprintEncoder, ledger_path: str = "journal_ledger.jsonl"):
        self.encoder = encoder
        self.ledger_path = ledger_path

        try:
            import speech_recognition as sr  # type: ignore
        except Exception as e:  # pragma: no cover - optional
            raise RuntimeError("speech_recognition is required for VoiceJournal") from e

        self._sr = sr
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Calibrate for ambient noise
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ready for voice input.")

    def start_session(self, eeg_stream_func, stylus_stream_func=None) -> None:
        print("\n=== Voice Journaling Session Started ===")
        print("Speak your thoughts. Say 'exit' to end session.\n")

        session_start = time.time()
        entry_count = 0

        while True:
            try:
                with self.microphone as source:
                    print("[Listening...]")
                    audio = self.recognizer.listen(source, timeout=None)

                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"\nYou said: {text}")

                    if "exit" in text.lower() or "stop" in text.lower():
                        print("\nEnding session...")
                        break

                    eeg_epoch = eeg_stream_func()
                    stylus_context = stylus_stream_func() if stylus_stream_func else None
                    sigprint, omega_state = self.encoder.process_epoch(eeg_epoch, stylus_context)

                    entry = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "session_time": time.time() - session_start,
                        "entry_num": entry_count,
                        "sigprint": sigprint,
                        "text": text,
                        "omega_state": {
                            "coherence": round(omega_state.coherence, 2),
                            "plv": round(omega_state.plv, 3),
                            "entropy": round(omega_state.entropy, 3),
                            "transition": getattr(omega_state, "transition", "INIT"),
                        },
                        "stylus": stylus_context,
                    }

                    self._commit_to_ledger(entry)
                    self._display_feedback(entry)
                    entry_count += 1

                except self._sr.UnknownValueError:
                    print("[Could not understand audio]")
                except self._sr.RequestError as e:
                    print(f"[Speech recognition error: {e}]")
            except KeyboardInterrupt:
                print("\n\nSession interrupted by user.")
                break

        print(f"\n=== Session Complete ===")
        print(f"Duration: {time.time() - session_start:.1f} seconds")
        print(f"Entries: {entry_count}")
        print(f"Gates detected: {self.encoder.gate_count}")
        print(f"Loops detected: {self.encoder.loop_count}")

    def _commit_to_ledger(self, entry: dict) -> None:
        # Read last hash
        prev_hash = "0" * 64
        try:
            with open(self.ledger_path, "r") as f:
                lines = f.readlines()
            if lines:
                import json as _json

                last = _json.loads(lines[-1])
                prev_hash = last.get("hash", prev_hash)
        except FileNotFoundError:
            pass

        import json as _json

        data_str = _json.dumps(entry, sort_keys=True)
        combined = prev_hash + data_str
        current_hash = hashlib.sha256(combined.encode()).hexdigest()
        entry["hash"] = current_hash
        entry["prev_hash"] = prev_hash

        with open(self.ledger_path, "a") as f:
            f.write(_json.dumps(entry) + "\n")

    @staticmethod
    def _display_feedback(entry: dict) -> None:
        print(f"\n--- Entry #{entry['entry_num']} ---")
        print(f"Time: {entry['session_time']:.1f}s")
        print(f"SIGPRINT: {entry['sigprint']}")
        print(f"Coherence: {entry['omega_state']['coherence']:.0f}/100")
        print(f"Phase-lock: {entry['omega_state']['plv']:.2f}")
        print(f"State: {entry['omega_state']['transition']}")
        if entry['omega_state']['transition'] == 'GATE':
            print(">>> GATE DETECTED: Significant state change <<<")
        print()


class StylusInterface:
    """Serial interface to RHZ Stylus (optional dependency: pyserial)."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.current_context: Dict = {}
        self._running = False
        self._thread = None

    def connect(self) -> None:
        try:
            import serial  # type: ignore
        except Exception as e:  # pragma: no cover - optional
            raise RuntimeError("pyserial is required for StylusInterface") from e

        self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)
        print(f"Connected to RHZ Stylus on {self.port}")

    def start_stream(self) -> None:
        if self.serial_conn is None:
            self.connect()
        import threading

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.serial_conn is not None:
            self.serial_conn.close()

    def get_context(self) -> Dict:
        return dict(self.current_context)

    def _read_loop(self) -> None:
        assert self.serial_conn is not None
        while self._running:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        import json as _json

                        data = _json.loads(line)
                        self.current_context = data
                    except Exception:
                        # ignore malformed
                        pass
            except Exception as e:
                print(f"Stylus read error: {e}")
                time.sleep(0.1)

    # --- Outgoing commands ---
    def send_command(self, command: Dict) -> None:
        """Send a JSON command to the Stylus (newline-delimited).

        If no serial connection is present, this is a no-op.
        """
        try:
            import json as _json
        except Exception:
            return
        if self.serial_conn is None:
            return
        line = _json.dumps(command) + "\n"
        try:
            self.serial_conn.write(line.encode("utf-8"))
        except Exception as e:
            print(f"Stylus write error: {e}")
