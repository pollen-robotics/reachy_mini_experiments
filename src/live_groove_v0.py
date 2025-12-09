#!/usr/bin/env python3
# coding: utf-8
"""
reachy_rhythm_controller.py - v12.0 (Simplified, no keyboard, no smart correction)

Real-time robot choreography driven by live BPM from the microphone.
- No manual controls (no keyboard listener).
- No "smart correction" phase alignment. The beat clock follows detected BPM directly.
- Auto-advances dance moves every N beats.
- Final analysis plot unchanged (BPM over time + corrected/reference beat clocks with accepted beats).

Dependencies: numpy, librosa, pyaudio, matplotlib, reachy_mini
"""

from __future__ import annotations

import argparse
import collections
import datetime
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from reachy_mini import ReachyMini, utils
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES


# ───────────────────────────────── Config ────────────────────────────────────
@dataclass
class Config:
    # Where to save the final analysis PNG
    save_dir: str = "."

    # Main control loop period (s). Lower for tighter control, higher for lower CPU.
    control_ts: float = 0.01

    # Audio analysis window (s). Longer gives more stable BPM, but slower reactions.
    audio_win: float = 2.0

    # Microphone sample rate (Hz). 44100 is common and well supported.
    audio_rate: int = 44100

    # Mic buffer size per read. Smaller reduces latency; too small increases CPU/underruns.
    audio_chunk_size: int = 2048

    # How many most recent BPM estimates to average for stability.
    bpm_stability_buffer: int = 4

    # Max allowed standard deviation over the stability buffer to consider "Locked".
    # Lower threshold = stricter lock; higher = looser (locks faster).
    bpm_stability_threshold: float = 6.0

    # If BPM becomes Unstable, how many consecutive unstable periods we tolerate before pausing motion.
    unstable_periods_before_stop: int = 4

    # If we haven't seen audio events for this many seconds, consider silence and stop motion.
    silence_tmo: float = 3.0

    # Buffer of recent accepted beat times used by the graph and control.
    beat_buffer_size: int = 20

    # Beat deduplication: beats closer than this fraction of the expected interval are considered duplicates.
    # Increase to remove double triggers; decrease if valid syncopations are dropped.
    min_interval_factor: float = 0.5

    # Auto-advance the dance after this many beats.
    beats_per_sequence: int = 8

    # How often the terminal UI refreshes (Hz).
    ui_update_rate: float = 1.0

    # Neutral pose (position in meters and Euler orientation in radians).
    # Adjust to fit your neutral posture in your setup.
    neutral_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    neutral_eul: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # The following two are kept for completeness but are not used (smart correction removed).
    offset_correction_rate: float = 0.02
    max_phase_correction_per_frame: float = 0.005

    # Derived: number of samples in the rolling audio window.
    audio_buffer_len: int = field(init=False)

    def __post_init__(self):
        self.audio_buffer_len = int(self.audio_rate * self.audio_win)


# ───────────────────────────── Runtime State ─────────────────────────────────
class MusicState:
    def __init__(self):
        self.lock = threading.Lock()
        self.librosa_bpm = 0.0
        self.raw_librosa_bpm = 0.0
        self.last_event_time = 0.0
        self.state = "Init"
        self.beats: collections.deque[float] = collections.deque(maxlen=512)
        self.unstable_period_count = 0


class Choreographer:
    def __init__(self):
        # Registry of available moves (name -> (fn, base_params, meta))
        self.move_names = list(AVAILABLE_MOVES.keys())
        # Default single waveform; moves that support it will consume it.
        self.waveforms = ["sin"]
        self.move_idx = 0
        self.waveform_idx = 0
        self.amplitude_scale = 1.0
        self.beat_counter_for_cycle = 0.0

    def current_move_name(self):
        return self.move_names[self.move_idx]

    def current_waveform(self):
        return self.waveforms[self.waveform_idx]

    def advance(self, beats_this_frame, config: Config):
        self.beat_counter_for_cycle += beats_this_frame
        if self.beat_counter_for_cycle >= config.beats_per_sequence:
            self.move_idx = (self.move_idx + 1) % len(self.move_names)
            self.beat_counter_for_cycle = 0.0


# ───────────────────────────── Worker Threads ────────────────────────────────
def audio_thread(state: MusicState, config: Config, stop_event: threading.Event) -> None:
    """Continuously read microphone audio and update BPM + beat times."""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=config.audio_rate,
        input=True,
        frames_per_buffer=config.audio_chunk_size,
    )

    buf = np.empty(0, dtype=np.float32)
    bpm_hist = collections.deque(maxlen=config.bpm_stability_buffer)

    while not stop_event.is_set():
        try:
            audio_chunk = np.frombuffer(
                stream.read(config.audio_chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )
            buf = np.append(buf, audio_chunk)
        except (IOError, ValueError):
            continue

        if len(buf) < config.audio_buffer_len:
            continue

        tempo, beat_frames = librosa.beat.beat_track(
            y=buf, sr=config.audio_rate, units="frames", tightness=100
        )
        now = time.time()
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) and tempo.size > 0 else tempo)

        with state.lock:
            state.last_event_time = now
            state.raw_librosa_bpm = tempo_val

            if tempo_val > 40:
                bpm_hist.append(tempo_val)
                state.librosa_bpm = float(np.mean(bpm_hist))

            win_dur = len(buf) / config.audio_rate
            abs_times = [
                now - (win_dur - librosa.frames_to_time(f, sr=config.audio_rate))
                for f in beat_frames
            ]
            for t in abs_times:
                if not state.beats or t - state.beats[-1] > 0.05:
                    state.beats.append(t)

            if len(bpm_hist) < config.bpm_stability_buffer:
                state.state = "Gathering"
                state.unstable_period_count = 0
            elif np.std(bpm_hist) < config.bpm_stability_threshold:
                state.state = "Locked"
                state.unstable_period_count = 0
            else:
                state.state = "Unstable"
                state.unstable_period_count += 1

        # Keep a rolling audio buffer of ~1.5 s to limit CPU
        buf = buf[-int(config.audio_rate * 1.5) :]

    stream.stop_stream()
    stream.close()
    pa.terminate()


def ui_thread(data_queue: Queue, config: Config, stop_event: threading.Event):
    """Lightweight terminal UI; refresh rate controlled by config.ui_update_rate."""
    last_ui_print_time, last_data = time.time(), None
    while not stop_event.is_set():
        try:
            while True:
                last_data = data_queue.get_nowait()
        except Empty:
            pass

        now = time.time()
        if not last_data or now - last_ui_print_time < (1.0 / config.ui_update_rate):
            time.sleep(0.1)
            continue
        last_ui_print_time = now

        paused_status = " | PAUSED (Music Unstable)" if last_data["unstable_pause"] else ""

        print(
            "\n" + "─" * 80 + "\n"
            f"🎵 Music State: {last_data['state']:<10} | BPM (Active/Raw): "
            f"{last_data['active_bpm']:.1f}/{last_data['raw_bpm']:.1f}{paused_status}\n"
            f"🕺 Dance State: {last_data['move_name']:<25} | Wave: {last_data['waveform']:<8} | Amp: {last_data['amp_scale']:.1f}x\n"
            f"⚙️  Settings: Auto mode | Beats/sequence: {config.beats_per_sequence}\n"
            + "─" * 80
        )
        sys.stdout.flush()


# ───────────────────────────── Plot (unchanged) ─────────────────────────────
def generate_final_plot(log, config: Config):
    if not log:
        return
    t = np.array([e["t"] for e in log])
    start_time = t[0]
    t -= start_time
    t_beats = np.array([e["t_beats"] for e in log])
    reference_t_beats = np.array([e["reference_t_beats"] for e in log])
    acc_beats = np.array([b for e in log for b in e["accepted_beats"]]) - start_time

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 8), constrained_layout=True)
    ax[0].plot(t, [e["active_bpm"] for e in log], "-", label="Active BPM")
    ax[0].set_ylabel("BPM")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, np.sin(2 * np.pi * t_beats), "-", label="Corrected Beat Clock (sin)")
    ax[1].plot(
        t,
        np.sin(2 * np.pi * reference_t_beats),
        "--",
        label="Reference Beat Clock (Metronome)",
        alpha=0.7,
    )
    ax[1].vlines(
        acc_beats,
        -1,
        1,
        colors="g",
        linestyles="solid",
        label="Accepted Beat",
        alpha=0.8,
    )
    ax[1].set_ylabel("Beat Cycle")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    path = os.path.join(
        config.save_dir, f"reachy_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    )
    fig.savefig(path, dpi=150)
    print(f"\nAnalysis plot saved to {path}")
    plt.show()


# ───────────────────────────── Main Control Loop ────────────────────────────
def main(config: Config) -> None:
    data_queue, stop_event = Queue(), threading.Event()
    music, choreographer = MusicState(), Choreographer()

    threading.Thread(target=audio_thread, args=(music, config, stop_event), daemon=True).start()
    threading.Thread(target=ui_thread, args=(data_queue, config, stop_event), daemon=True).start()

    last_loop = time.time()
    processed_beats, active_bpm = 0, 0.0
    filtered_beat_times = collections.deque(maxlen=config.beat_buffer_size)
    full_log = []
    t_beats, reference_t_beats = 0.0, 0.0

    print("Connecting to Reachy Mini...")
    with ReachyMini() as mini:
        # mini.wake_up()  # Uncomment if your setup uses explicit wake/sleep
        mini.set_target(
            utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
            antennas=np.zeros(2),
        )
        time.sleep(1.0)
        print("\nRobot ready — play music!\n")

        try:
            while True:
                loop_start_time = time.time()
                dt = loop_start_time - last_loop
                last_loop = loop_start_time

                with music.lock:
                    librosa_bpm, raw_bpm = music.librosa_bpm, music.raw_librosa_bpm
                    state, last_event_time = music.state, music.last_event_time
                    unstable_count = music.unstable_period_count
                    new_beats = list(music.beats)[processed_beats:]
                processed_beats += len(new_beats)

                active_bpm = librosa_bpm if time.time() - last_event_time < config.silence_tmo else 0.0

                # Beat filtering and deduplication
                accepted_this_frame = []
                if new_beats and active_bpm > 0:
                    expected_interval = 60.0 / active_bpm
                    min_interval = expected_interval * config.min_interval_factor
                    i = 0
                    while i < len(new_beats):
                        last_beat = (
                            filtered_beat_times[-1]
                            if filtered_beat_times
                            else new_beats[i] - expected_interval
                        )
                        current_beat = new_beats[i]
                        if i + 1 < len(new_beats) and (new_beats[i + 1] - current_beat) < min_interval:
                            c = new_beats[i + 1]
                            e1 = abs((current_beat - last_beat) - expected_interval)
                            e2 = abs((c - last_beat) - expected_interval)
                            accepted_this_frame.append(current_beat if e1 <= e2 else c)
                            i += 2
                        else:
                            if (current_beat - last_beat) > min_interval:
                                accepted_this_frame.append(current_beat)
                            i += 1
                filtered_beat_times.extend(accepted_this_frame)
                last_good_beat = filtered_beat_times[-1] if filtered_beat_times else 0.0  # kept for plot consistency

                # Start/continue criteria
                is_allowed_to_start = state == "Locked"
                is_stable_enough_to_continue = unstable_count < config.unstable_periods_before_stop
                can_dance = active_bpm > 0 and (is_allowed_to_start or (state == "Unstable" and is_stable_enough_to_continue))

                if can_dance:
                    beats_this_frame = dt * (active_bpm / 60.0)
                    reference_t_beats += beats_this_frame
                    t_beats = reference_t_beats  # no smart correction

                    choreographer.advance(beats_this_frame, config)
                    move_name = choreographer.current_move_name()

                    move_fn, base_params, _ = AVAILABLE_MOVES[move_name]
                    params = base_params.copy()

                    # If the move supports waveform, use the default single waveform.
                    if "waveform" in params:
                        params["waveform"] = choreographer.current_waveform()

                    # Keep amplitude as-is (no manual scaling in this simplified version).
                    offsets = move_fn(t_beats, **params)

                    mini.set_target(
                        utils.create_head_pose(
                            *(config.neutral_pos + offsets.position_offset),
                            *(config.neutral_eul + offsets.orientation_offset),
                            degrees=False,
                        ),
                        antennas=offsets.antennas_offset,
                    )
                else:
                    mini.set_target(
                        utils.create_head_pose(*config.neutral_pos, *config.neutral_eul, degrees=False),
                        antennas=np.zeros(2),
                    )

                # UI + log
                ui_data = {
                    "state": state,
                    "active_bpm": active_bpm,
                    "raw_bpm": raw_bpm,
                    "move_name": choreographer.current_move_name(),
                    "waveform": choreographer.current_waveform(),
                    "amp_scale": choreographer.amplitude_scale,
                    "unstable_pause": not is_stable_enough_to_continue and state == "Unstable",
                }
                data_queue.put(ui_data)
                full_log.append(
                    {
                        "t": time.time(),
                        "active_bpm": active_bpm,
                        "t_beats": t_beats,
                        "reference_t_beats": reference_t_beats,
                        "accepted_beats": accepted_this_frame,
                        "last_good_beat": last_good_beat,
                    }
                )

                time.sleep(max(0.0, config.control_ts - (time.time() - loop_start_time)))

        except KeyboardInterrupt:
            print("\nCtrl-C received, shutting down...")
        finally:
            stop_event.set()
            print("Putting robot to sleep and cleaning up...")
            # try:
            #     mini.goto_sleep()
            # except Exception:
            #     pass
            print("Shutdown complete.")
            generate_final_plot(full_log, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reachy Rhythm Controller (simplified, no keyboard, no smart correction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save-dir", default=".", help="Folder for PNG analysis plot")
    args = parser.parse_args()
    cfg = Config(save_dir=args.save_dir)
    main(cfg)
