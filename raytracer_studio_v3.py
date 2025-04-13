
# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import gradio as gr
from scipy.io import wavfile
from scipy.signal import fftconvolve, spectrogram
from pydub import AudioSegment
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw # Ensure ImageDraw is imported
import shutil
import zipfile
import traceback # For more detailed error printing

# NEUE IMPORTS für den Audio Profiler (Hinzugefügt)
import soundfile as sf # NEU: Für robustes Laden im Profiler
import pyloudnorm as pyln # NEU: Für LUFS-Messung
import math # NEU: Für log10 in dBFS-Berechnung
# --- Ende Neue Imports ---

# --- Robust Preset Handling ---
PRESET_DIR = "presets"
LAST_PRESET_FILE = os.path.join(PRESET_DIR, "last_preset.txt")

def ensure_preset_dir():
    """Ensures the preset directory exists."""
    os.makedirs(PRESET_DIR, exist_ok=True)

def save_last_preset(preset_name):
    """Saves the name of the last used preset."""
    ensure_preset_dir()
    try:
        # Ensure preset_name is a string, handle None or empty gracefully
        name_to_save = preset_name if isinstance(preset_name, str) else ""
        with open(LAST_PRESET_FILE, "w") as f:
            f.write(name_to_save)
    except Exception as e:
        print(f"Error saving last preset '{preset_name}': {e}")
        # traceback.print_exc()

def load_last_preset():
    """Loads the name of the last used preset file, checking its existence."""
    ensure_preset_dir()
    try:
        if not os.path.exists(LAST_PRESET_FILE):
            return None
        with open(LAST_PRESET_FILE, "r") as f:
            last_preset = f.read().strip()
        if not last_preset: # Handle empty file content
            return None
        preset_path = os.path.join(PRESET_DIR, last_preset)
        # Check if the referenced file actually exists and is a json file
        if os.path.exists(preset_path) and preset_path.endswith(".json"):
            return last_preset # Return the filename string
        else:
            # If last preset file points to a non-existent/invalid file, clear it
            print(f"Warning: Last preset file '{last_preset}' not found or invalid. Clearing reference.")
            save_last_preset("") # Clear the invalid entry
            return None
    except Exception as e:
        print(f"Error loading last preset file: {e}")
        # traceback.print_exc()
        return None

# --- Material Configuration ---
material_absorption = {
    "Stein": 0.1,
    "Holz": 0.3,
    "Teppich": 0.6,
    "Glas": 0.2
}
DEFAULT_MATERIAL = "Teppich" # Define a default
DEFAULT_HALL_TYPE = "Room" # Default hall type

# --- Audio Processing Functions (Original - potentially remove later) ---
# Note: Kept for potential future use or comparison, but not used in the main split processing flow.
def generate_impulse_response(rate, ir_duration, reflection_count, max_delay, material):
    """Generates a simple impulse response based on parameters."""
    try:
        rate = int(rate)
        ir_duration = float(ir_duration)
        reflection_count = int(reflection_count)
        max_delay = float(max_delay)

        if rate <= 0 or ir_duration <= 0:
            print(f"Warning: Invalid rate ({rate}) or ir_duration ({ir_duration}). Returning minimal IR.")
            return np.array([1.0], dtype=np.float32)

        length = max(1, int(ir_duration * rate)) # Ensure length is at least 1
        ir = np.zeros(length, dtype=np.float32)
        if length == 0: return ir # Should not happen with max(1, ...) but safety first
        ir[0] = 1.0  # Direct sound

        absorption = material_absorption.get(material, material_absorption.get(DEFAULT_MATERIAL, 0.3))

        if max_delay <= 0: # No reflections if max_delay is non-positive
             return ir

        max_delay_samples = int(max_delay * rate)
        if max_delay_samples <= 1: # Need at least 1 sample delay possible for reflections
            return ir # Only direct sound if max_delay is too small

        for _ in range(reflection_count):
            # Ensure delay is between 1 and max_delay_samples, and within IR bounds
            # Use max(1, ...) to avoid issues if max_delay_samples becomes 1
            delay_samples = np.random.randint(1, max(2, max_delay_samples + 1))
            delay_samples = min(delay_samples, length - 1) # Cannot exceed ir length - 1
            if delay_samples <= 0 : continue # Skip if delay becomes non-positive

            attenuation = np.random.uniform(0.1, 0.4) * (1.0 - absorption)
            ir[delay_samples] += attenuation # Add reflection

        return ir

    except ValueError as ve:
         print(f"Error: Invalid numeric input for IR generation: {ve}. Returning minimal IR.")
         return np.array([1.0], dtype=np.float32)
    except Exception as e:
        print(f"Unexpected error in generate_impulse_response: {e}")
        traceback.print_exc()
        return np.array([1.0], dtype=np.float32)

def convolve_audio(data, ir, dry_wet, bass_gain=1.0, treble_gain=1.0, rate=44100):
    """Convolves audio data with IR, applies EQ and Dry/Wet mix."""
    # Input validation and preparation
    if data is None or data.size == 0:
        print("Warning: convolve_audio received empty data.")
        return np.zeros((0, 2), dtype=np.float32) # Return empty stereo array
    if ir is None or ir.size == 0:
        print("Warning: convolve_audio received empty IR. Using direct signal.")
        # Treat as 100% dry signal instead of generating a dummy IR
        output = np.copy(data)
        dry_wet = 0.0 # Force dry if IR is invalid
    elif ir.ndim > 1: # Ensure IR is 1D
        ir = ir.flatten()

    if data.ndim == 1: # Handle mono input -> convert to stereo
        data = np.stack((data, data), axis=1)
    elif data.shape[1] == 1:
         data = np.repeat(data, 2, axis=1)
    elif data.shape[1] != 2: # Ensure stereo by taking first two channels
        print(f"Warning: Input audio has {data.shape[1]} channels. Using first two.")
        data = data[:, :2]

    # Ensure float32 for processing
    data = data.astype(np.float32)
    ir = ir.astype(np.float32) if ir is not None else None

    # --- Convolution (only if IR is valid) ---
    if ir is not None and ir.size > 0:
        try:
            output_left = fftconvolve(data[:, 0], ir, mode='full')
            output_right = fftconvolve(data[:, 1], ir, mode='full')
            # Trim convolution result to original length + IR length - 1
            len_output = data.shape[0] + len(ir) - 1
            output = np.stack((output_left[:len_output], output_right[:len_output]), axis=1)
            # IMPORTANT: Trim WET signal back to original data length *before* mixing
            output = output[:data.shape[0], :]
        except Exception as e:
            print(f"Error during convolution: {e}")
            traceback.print_exc()
            # Fallback: use dry signal if convolution fails
            output = np.copy(data) # Use copy to avoid modifying original data later
            dry_wet = 0.0 # Force dry on error
    else:
        # This case handles the empty/None IR from the start
        output = np.copy(data) # Wet signal is just a copy of dry
        dry_wet = 0.0 # Ensure it's dry

    # --- Dry/Wet Mix ---
    mixed = ((1.0 - dry_wet) * data) + (dry_wet * output)

    # --- EQ (Applied to the mixed signal) ---
    try:
        # Apply EQ only if gains are not 1.0
        if not np.isclose(bass_gain, 1.0) or not np.isclose(treble_gain, 1.0):
            fft_data = np.fft.rfft(mixed, axis=0)
            # Calculate freqs based on actual rate and signal length
            rate_for_freq = rate if rate > 0 else 44100 # Use default if rate invalid
            freqs = np.fft.rfftfreq(mixed.shape[0], d=1.0/rate_for_freq)

            # Define frequency bands (adjust as needed)
            bass_cutoff = 250 # Hz
            treble_cutoff = 4000 # Hz

            # Create masks safely, excluding DC (0 Hz) from bass boost
            bass_mask = (freqs > 1e-6) & (freqs <= bass_cutoff)
            treble_mask = freqs >= treble_cutoff

            fft_data[bass_mask] *= bass_gain
            fft_data[treble_mask] *= treble_gain

            mixed_eq = np.fft.irfft(fft_data, n=mixed.shape[0], axis=0).astype(np.float32)
        else:
            mixed_eq = mixed # Skip FFT/IFFT if no EQ change

    except Exception as e:
        print(f"Error during EQ: {e}")
        traceback.print_exc()
        mixed_eq = mixed # Fallback to non-EQ'd signal on error

    # --- Normalization ---
    max_val = np.max(np.abs(mixed_eq))
    if max_val > 1e-6: # Avoid division by zero/very small numbers
        mixed_norm = mixed_eq / max_val
    elif np.any(mixed_eq): # Check if it's not all zeros before zeroing out
        print("Warning: Signal close to zero after processing, outputting silence.")
        mixed_norm = np.zeros_like(mixed_eq)
    else: # Already all zeros
        mixed_norm = mixed_eq

    return mixed_norm.astype(np.float32)


# --- NEUE Funktion hinzugefügt: Dynamisches Dry/Wet Mischen ---
def dynamic_dry_wet_mix(
    dry_signal: np.ndarray,
    wet_signal: np.ndarray,
    dry_wet: float,
    kill_start: float = 0.5
) -> np.ndarray:
    """
    Mischt Dry- und Wet-Signal dynamisch, wobei der Dry-Anteil ab kill_start linear ausgeblendet wird.

    Parameter:
    - dry_signal: Originalsignal (z.B. stereo ndarray)
    - wet_signal: Effektsignal (z.B. stereo ndarray)
    - dry_wet: Dry/Wet-Verhältnis (0 = 100% Dry, 1 = 100% Wet)
    - kill_start: Ab welchem DryWet-Wert (z.B. 0.5) das Dry-Signal heruntergeblendet wird

    Rückgabe:
    - Gemischtes Signal als ndarray
    """
    try:
        # Ensure signals are float32 for processing
        dry_signal = dry_signal.astype(np.float32)
        wet_signal = wet_signal.astype(np.float32)

        # Input validation for scalars
        dry_wet = np.clip(float(dry_wet), 0.0, 1.0)
        kill_start = np.clip(float(kill_start), 0.0, 1.0)

        # Check if kill_start makes sense (should ideally be less than 1.0)
        if kill_start >= 1.0:
            # If kill_start is 1 or more, the dynamic part never happens, act like standard mix
            dry_mix_factor = 1.0
        else:
            # Dynamischer Dry-Faktor
            dry_mix_factor = 1.0
            if dry_wet >= kill_start:
                # Calculate fade-out slope based on the remaining range (1.0 - kill_start)
                fade_range = 1.0 - kill_start
                # Avoid division by zero if kill_start is exactly 1.0 (handled above, but safety)
                if fade_range < 1e-6:
                    dry_mix_factor = 0.0 # Fade out completely if kill_start is effectively 1.0
                else:
                    # Berechne den Faktor, wie weit wir im Fade-Bereich sind
                    progress_in_fade = (dry_wet - kill_start) / fade_range
                    # Der dry_mix_factor geht von 1 (am kill_start) auf 0 (bei dry_wet=1.0)
                    dry_mix_factor = np.clip(1.0 - progress_in_fade, 0.0, 1.0)


        # Ensure signals have the same length before mixing - crucial!
        min_len = min(dry_signal.shape[0], wet_signal.shape[0])
        dry_signal_matched = dry_signal[:min_len]
        wet_signal_matched = wet_signal[:min_len]

        # Mischung anwenden: dry_level * dry + wet_level * wet
        # dry_level = dry_mix_factor * (1.0 - dry_wet) # Dry level is reduced by dry_wet AND the dynamic factor
        # wet_level = dry_wet                         # Wet level is just controlled by dry_wet
        # print(f"  [DynamicMix] DryWet={dry_wet:.2f}, KillStart={kill_start:.2f}, DryFactor={dry_mix_factor:.2f}, DryLevel={dry_mix_factor * (1.0 - dry_wet):.2f}, WetLevel={dry_wet:.2f}")
        mixed = (dry_mix_factor * (1.0 - dry_wet) * dry_signal_matched) + (dry_wet * wet_signal_matched)

        return mixed.astype(np.float32)

    except Exception as e:
        print(f"Error in dynamic_dry_wet_mix: {e}")
        traceback.print_exc()
        # Fallback auf normales statisches Mischen bei Fehler, ensuring length match
        try:
            dry_wet_fallback = np.clip(float(dry_wet), 0.0, 1.0) # Ensure dry_wet is valid for fallback
            min_len_fallback = min(dry_signal.shape[0], wet_signal.shape[0])
            print("  [DynamicMix] Error occurred, falling back to static mix.")
            return (((1.0 - dry_wet_fallback) * dry_signal[:min_len_fallback]) + (dry_wet_fallback * wet_signal[:min_len_fallback])).astype(np.float32)
        except Exception as fallback_e:
            print(f"Error during fallback mixing in dynamic_dry_wet_mix: {fallback_e}")
            # If even fallback fails, return wet signal or silence
            if wet_signal is not None and wet_signal.size > 0:
                return wet_signal[:min(dry_signal.shape[0] if dry_signal is not None else 0, wet_signal.shape[0])].astype(np.float32) # Try to match length
            elif dry_signal is not None and dry_signal.size > 0:
                 return dry_signal.astype(np.float32) # Or maybe dry as last resort?
            else:
                return np.array([], dtype=np.float32) # Return empty if inputs are bad


# --- NEW Audio Processing Functions (Split IR) ---
def generate_impulse_response_split(
    rate: int,
    ir_duration: float,
    reflection_count: int,
    max_delay: float,
    material: str,
    directionality: float = 0.5,
    early_late_split: float = 0.08 # Zeitanteil Early Reflections, z.B. 80ms
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt zwei Impulsantworten:
    - Early Reflections (gerichteter, basierend auf directionality)
    - Late Reverb (diffuser, als decaying noise)

    Parameter:
    - rate: Samplingrate (z.B. 44100 Hz)
    - ir_duration: Gesamtdauer der IR (z.B. 1.5 Sekunden)
    - reflection_count: Anzahl der *potentiellen* Early-Reflections
    - max_delay: Maximale Verzögerung für *Early Reflections* (z.B. 0.06s)
    - material: Materialname (z.B. "Stein", "Holz")
    - directionality: 0.0 = völlig diffus, 1.0 = maximal gerichtet (mehr/stärkere Early Ref.)
    - early_late_split: Zeitliche Grenze (in Sekunden) zwischen Early und Late (z.B. 0.08s)

    Rückgabe:
    (early_ir: np.ndarray, late_ir: np.ndarray)
    """
    try:
        rate = int(rate)
        ir_duration = float(ir_duration)
        reflection_count = int(reflection_count)
        max_delay = float(max_delay)
        directionality = float(directionality)
        early_late_split_time = float(early_late_split) # Now represents time in seconds

        if rate <= 0 or ir_duration <= 0:
            print(f"Warning: Invalid rate ({rate}) or ir_duration ({ir_duration}). Returning minimal IRs.")
            return np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32)

        length = max(1, int(ir_duration * rate))
        early_ir = np.zeros(length, dtype=np.float32)
        late_ir = np.zeros(length, dtype=np.float32)

        absorption = material_absorption.get(material, material_absorption.get(DEFAULT_MATERIAL, 0.3))

        # Split point in samples, based on time
        split_point_samples = max(1, min(int(early_late_split_time * rate), length - 1))

        # Max delay for early reflections in samples
        max_delay_samples = max(2, int(max_delay * rate))

        # --- Early Reflections (gerichteter) ---
        # Add direct impulse slightly attenuated to early part (optional, maybe better handled by dry signal)
        # early_ir[0] = 0.1 # Reduced direct sound in early IR

        if reflection_count > 0 and split_point_samples > 1:
            for _ in range(reflection_count):
                # Ensure delay is within early reflection part (1 to split_point_samples-1)
                # Limit delay by both max_delay_samples and the split_point_samples
                actual_max_early_delay = min(max_delay_samples, split_point_samples)
                if actual_max_early_delay <= 1: continue # Skip if possible delay range is too small

                delay_samples = np.random.randint(1, max(2, actual_max_early_delay))

                if delay_samples > 0 and delay_samples < split_point_samples:
                    # Richtung beeinflusst Stärke (gerichteter Schall = stärkere Reflexion)
                    base_strength = np.random.uniform(0.3, 0.8) # Base reflection strength
                    strength = base_strength * (1.0 - absorption)
                    # Apply directionality factor: more directional = potentially stronger early reflections
                    # Scale strength directly with directionality
                    strength *= np.clip(directionality, 0.1, 1.0) # Ensure some ER even if directionality is low
                    # Simple decay based on delay time (longer delay = weaker) relative to max early delay
                    strength *= (1.0 - (delay_samples / actual_max_early_delay)**0.7) # Slightly faster decay

                    early_ir[delay_samples] += strength

        # --- Late Reverb (diffus verteilt, decaying noise) ---
        # Start the late reverb smoothly after the early reflections end
        start_late_index = split_point_samples
        late_part_length = length - start_late_index

        if late_part_length > 0:
            # Exponential decay over the late part duration
            # Target decay, e.g., -60dB (amplitude 1e-3) over the late part
            # decay_factor ^ late_part_length = 1e-3 => decay_factor = (1e-3)^(1/late_part_length)
            # Use a slightly faster decay target, e.g., -50dB
            target_amplitude_ratio = 10**(-50 / 20) # Amplitude ratio for -50dB
            if late_part_length > 1:
                decay_factor = np.power(target_amplitude_ratio, 1.0 / late_part_length)
            else:
                decay_factor = 0.1 # Rapid decay if only one sample

            decay_factor = np.clip(decay_factor, 0.8, 0.99999) # Clamp for stability

            # Initial amplitude influenced by (1 - directionality) and overall duration
            initial_late_amp = 0.5 * (1.0 - np.clip(directionality, 0.0, 0.9)) # Less directional = more late reverb
            # Slightly adjust based on total IR duration (longer reverb = slightly quieter start?)
            initial_late_amp *= np.clip(1.0 / (1 + ir_duration), 0.3, 1.0)

            # Apply absorption effect to the overall late reverb level
            initial_late_amp *= (1.0 - absorption**0.5) # Less impact than direct absorption

            current_late_amp = initial_late_amp

            # Generate decaying noise
            late_noise = np.random.uniform(-1, 1, size=late_part_length)

            # Apply decay envelope
            decay_envelope = np.power(decay_factor, np.arange(late_part_length))
            late_ir[start_late_index:] = late_noise * current_late_amp * decay_envelope


        # --- Normalisierung ---
        # Normalize Early Reflections peak (excluding potential direct impulse at 0)
        early_max = np.max(np.abs(early_ir[1:])) if length > 1 else 0
        if early_max > 1e-6:
            early_ir[1:] /= early_max # Normalize reflections relative to each other
            early_ir[1:] *= 0.9 # Scale normalized reflections slightly below 1.0 peak

        # Normalize Late Reverb peak
        late_max = np.max(np.abs(late_ir))
        if late_max > 1e-6:
            late_ir /= late_max
            late_ir *= 0.7 # Scale late reverb down relative to potential early peaks

        return early_ir, late_ir

    except ValueError as ve:
         print(f"Error: Invalid numeric input for split IR generation: {ve}. Returning minimal IRs.")
         return np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32)
    except Exception as e:
        print(f"Unexpected error in generate_impulse_response_split: {e}")
        traceback.print_exc()
        return np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32)


def convolve_audio_split(
    data: np.ndarray,
    early_ir: np.ndarray,
    late_ir: np.ndarray,
    early_level: float, # Now controlled by adaptive function + slider
    late_level: float,  # Now controlled by adaptive function + slider
    dry_wet: float,
    bass_gain: float = 1.0,
    treble_gain: float = 1.0,
    rate: int = 44100,
    kill_start_dw: float = 0.5 # Add parameter for dynamic dry/wet
) -> np.ndarray:
    """
    Wendet Early und Late Impulsantworten getrennt an, mixt sie kontrolliert zusammen
    und verwendet dynamic_dry_wet_mix für die Dry/Wet-Mischung.

    Parameter:
    - data: Original-Audio (Stereo erwartet)
    - early_ir: Early Reflections Impulsantwort
    - late_ir: Late Reverb Impulsantwort
    - early_level: Effektiver Anteil der Early Reflections (nach Adaption)
    - late_level: Effektiver Anteil des Late Reverb (nach Adaption)
    - dry_wet: 0 = nur Originalsignal, 1 = nur Effekt
    - bass_gain: Bass-EQ (Faktor, 1.0 = neutral)
    - treble_gain: Höhen-EQ (Faktor, 1.0 = neutral)
    - rate: Samplingrate (z.B. 44100)
    - kill_start_dw: Ab welchem DryWet-Wert das Dry-Signal ausgeblendet wird (für dynamic_dry_wet_mix)

    Rückgabe:
    - gemischtes Signal (Stereo)
    """
    if data is None or data.size == 0:
        print("Warning: convolve_audio_split received empty data.")
        return np.zeros((0, 2), dtype=np.float32)

    # Ensure data is stereo float32
    if data.ndim == 1:
        data = np.stack((data, data), axis=1)
    elif data.shape[1] == 1:
         data = np.repeat(data, 2, axis=1)
    elif data.shape[1] != 2:
        print(f"Warning: Input audio has {data.shape[1]} channels. Using first two.")
        data = data[:, :2]
    data = data.astype(np.float32)

    # Ensure IRs are 1D float32
    early_ir = np.asarray(early_ir, dtype=np.float32).flatten() if early_ir is not None else np.zeros(1, dtype=np.float32)
    late_ir = np.asarray(late_ir, dtype=np.float32).flatten() if late_ir is not None else np.zeros(1, dtype=np.float32)

    # --- Convolution with Early IR ---
    early_wet = np.zeros_like(data)
    if early_ir is not None and early_ir.size > 0 and np.any(early_ir) and early_level > 1e-6: # Only compute if IR exists and level > 0
        try:
            # Truncate to data length directly after convolution
            early_left = fftconvolve(data[:, 0], early_ir, mode='full')[:data.shape[0]]
            early_right = fftconvolve(data[:, 1], early_ir, mode='full')[:data.shape[0]]
            early_wet = np.stack((early_left, early_right), axis=1)
        except Exception as e:
            print(f"Error during early convolution: {e}")
            traceback.print_exc() # Print full error
            # Fallback: zero early wet signal on error

    # --- Convolution with Late IR ---
    late_wet = np.zeros_like(data)
    if late_ir is not None and late_ir.size > 0 and np.any(late_ir) and late_level > 1e-6: # Only compute if IR exists and level > 0
        try:
            # Truncate to data length directly after convolution
            late_left = fftconvolve(data[:, 0], late_ir, mode='full')[:data.shape[0]]
            late_right = fftconvolve(data[:, 1], late_ir, mode='full')[:data.shape[0]]
            late_wet = np.stack((late_left, late_right), axis=1)
        except Exception as e:
            print(f"Error during late convolution: {e}")
            traceback.print_exc() # Print full error
            # Fallback: zero late wet signal on error

    # --- Combine Wet Signals with Levels ---
    # Ensure wet signals have the same length as data before mixing levels
    # Apply levels AFTER convolution
    wet_combined = (early_wet[:data.shape[0]] * early_level) + (late_wet[:data.shape[0]] * late_level)

    # --- Dry/Wet Mischung mit flexiblem Dynamic Dry Muting ---
    # Replace the old manual calculation with the new function call
    mixed = dynamic_dry_wet_mix(
        dry_signal=data[:wet_combined.shape[0]], # Use dry signal matched to wet length
        wet_signal=wet_combined,
        dry_wet=dry_wet,
        kill_start=kill_start_dw # Pass the kill_start parameter
    )
    # --- ENDE ERSETZUNG ---

    # --- Optional EQ auf das Mischsignal ---
    try:
        # Ensure 'mixed' array is not empty before proceeding
        if mixed is None or mixed.size == 0:
            print("Warning: Mixed signal is empty before EQ. Skipping EQ.")
            mixed_eq = np.zeros((0, data.shape[1] if data.ndim > 1 else 1), dtype=np.float32) # Return empty array of correct shape
        elif not np.isclose(bass_gain, 1.0) or not np.isclose(treble_gain, 1.0):
            fft_data = np.fft.rfft(mixed, axis=0)
            rate_for_freq = rate if rate > 0 else 44100 # Use default if rate invalid
            # Ensure sufficient length for FFT frequency calculation
            n_fft = mixed.shape[0]
            if n_fft < 2: # Need at least 2 samples for rfftfreq
                 print("Warning: Signal too short for EQ. Skipping.")
                 mixed_eq = mixed
            else:
                freqs = np.fft.rfftfreq(n_fft, d=1.0/rate_for_freq)

                bass_cutoff = 250 # Hz
                treble_cutoff = 4000 # Hz
                bass_mask = (freqs > 1e-6) & (freqs <= bass_cutoff)
                treble_mask = freqs >= treble_cutoff

                # Apply gain carefully to avoid excessive boost
                fft_data[bass_mask] *= np.clip(bass_gain, 0.1, 5.0) # Clip gain factor
                fft_data[treble_mask] *= np.clip(treble_gain, 0.1, 5.0) # Clip gain factor

                mixed_eq = np.fft.irfft(fft_data, n=n_fft, axis=0).astype(np.float32)
        else:
            mixed_eq = mixed # Skip EQ if gains are neutral

    except Exception as e:
        print(f"Error during EQ in split convolution: {e}")
        traceback.print_exc()
        mixed_eq = mixed # Fallback to non-EQ'd signal

    # --- Normalisierung ---
    # Ensure 'mixed_eq' exists and is not empty
    if mixed_eq is None or mixed_eq.size == 0:
         print("Warning: Signal is empty after EQ/Fallback. Returning empty.")
         # Ensure the returned empty array matches the expected stereo output shape
         return np.zeros((0, 2), dtype=np.float32)


    max_val = np.max(np.abs(mixed_eq))
    if max_val > 1e-6:
        mixed_norm = mixed_eq / max_val
    elif np.any(mixed_eq): # Check if it's not all zeros
        print("Warning: Signal close to zero after split processing, outputting silence.")
        mixed_norm = np.zeros_like(mixed_eq)
    else: # Already all zeros
        mixed_norm = mixed_eq

    # Ensure output is always stereo
    if mixed_norm.ndim == 1:
        mixed_norm = np.stack((mixed_norm, mixed_norm), axis=1)
    elif mixed_norm.shape[1] == 1:
        mixed_norm = np.repeat(mixed_norm, 2, axis=1)
    # If it already has more than 2 channels somehow (shouldn't happen here), take first two
    elif mixed_norm.shape[1] > 2:
        mixed_norm = mixed_norm[:, :2]

    return mixed_norm.astype(np.float32)


# --- Hall Type Helper Functions ---
def update_hall_info(selected_hall_type: str) -> str:
    """Updates the description text based on the selected hall type."""
    hall_info = {
        "Plate": "Klassischer Studioplate-Hall. Dicht, hell, relativ kurze Nachhallzeit, stark gerichtet (wenig diffus). Gut für Vocals, Snares.",
        "Room": "Natürlicher Raumklang. Ausgewogene frühe Reflexionen und Nachhall, mittlere Gerichtetheit. Universell einsetzbar für Realismus.",
        "Cathedral": "Große Kathedrale. Sehr langer, diffuser Nachhall, späte Reflexionen dominant, geringe Gerichtetheit. Für Ambient, orchestrale Sounds."
    }
    return f"ℹ️ **Beschreibung:** {hall_info.get(selected_hall_type, 'Keine Beschreibung verfügbar.')}"

def adjust_reverb_parameters_by_hall(hall_type: str) -> tuple[float, int, float, float]:
    """
    Gibt passende *Voreinstellungen* für ir_duration, reflection_count, max_delay (für Early Ref.)
    und early_late_split_time basierend auf dem Halltyp zurück.
    """
    if hall_type == "Plate":
        # Shorter duration, moderate reflections, short ER max delay, very short ER period
        return 0.8, 25, 0.025, 0.03 # Duration, RefCount, MaxERDelay, ERSplitTime
    elif hall_type == "Room":
        # Medium duration, more reflections, medium ER max delay, standard ER period
        return 1.5, 35, 0.06, 0.08
    elif hall_type == "Cathedral":
        # Long duration, fewer *distinct* early reflections, longer ER max delay, longer ER period
        return 4.0, 20, 0.10, 0.12
    else: # Fallback (Room-ähnlich)
        print(f"Warning: Unknown hall type '{hall_type}', using Room defaults.")
        return 1.5, 35, 0.06, 0.08

# --- NEUE HILFSFUNKTION: Adaptive Early/Late Balance ---
def adapt_early_late_levels(dry_wet: float, base_early: float = 0.8, base_late: float = 0.6) -> tuple[float, float]:
    """
    Passt Early- und Late-Reverb-Level dynamisch basierend auf dem DryWet-Wert an.

    dry_wet = 0.0 → Early dominant (klingt direkter, weniger Hallfahne)
    dry_wet = 1.0 → Late dominant (klingt diffuser, mehr Nachhall)

    Basiswerte (z.B. Early=0.8, Late=0.6) werden automatisch skaliert.

    Rückgabe:
    (angepasster Early-Level, angepasster Late-Level)
    """
    try:
        dry_wet = np.clip(float(dry_wet), 0.0, 1.0)
        base_early = float(base_early)
        base_late = float(base_late)

        # Early scale decreases as dry_wet increases (non-linearly, more effect towards wet)
        early_scale = 1.0 - (dry_wet**1.5 * 0.7) # Max reduction to 30% at wet=1.0

        # Late scale increases as dry_wet increases (non-linearly, more effect towards wet)
        late_scale = 1.0 + (dry_wet**1.5 * 0.6) # Max boost to 160% at wet=1.0

        adapted_early = base_early * early_scale
        adapted_late = base_late * late_scale

        # Clipping limits to prevent extreme values, ensure non-negative
        adapted_early = np.clip(adapted_early, 0.0, 2.0)
        adapted_late = np.clip(adapted_late, 0.0, 2.0)

        # print(f"  [Adaptive Balance] DryWet={dry_wet:.2f} -> BaseE={base_early:.2f}, BaseL={base_late:.2f} -> AdaptE={adapted_early:.2f}, AdaptL={adapted_late:.2f}")

        return adapted_early, adapted_late

    except Exception as e:
        print(f"Error in adapt_early_late_levels: {e}")
        traceback.print_exc()
        # Return base values in case of error to allow processing to continue
        return base_early, base_late

# --- Bestehende HELFER FUNKTION (MODIFIZIERT): compute_final_directionality ---
def compute_final_directionality(x_pos: float, y_pos: float, hall_type: str, dry_wet: float = 0.5) -> float:
    """
    Berechnet die endgültige Directionality (0=diffus, 1=gerichtet) basierend auf Position, Halltyp und DryWet-Mix.

    Positionseffekt:
    - Zentrum (0.5, 0.5) → eher gerichtet
    - Randbereiche → eher diffuser

    Halltyp-Effekt (Basis-Directionality):
    - Plate → ~0.9 (sehr gerichtet)
    - Room → ~0.6
    - Cathedral → ~0.2 (sehr diffus)

    DryWet-Effekt:
    - DryWet > 0.6 → Leichter Boost der Directionality für mehr Klarheit im Effektklang
    """

    try:
        x_clamped = np.clip(float(x_pos), 0.0, 1.0)
        y_clamped = np.clip(float(y_pos), 0.0, 1.0)

        # Distance from center (0.0 to ~0.707)
        distance_from_center = np.sqrt((x_clamped - 0.5)**2 + (y_clamped - 0.5)**2)
        # Normalize distance (0=center, 1=corner)
        normalized_distance = distance_from_center / 0.707
        # Position effect: Reduces directionality as distance increases
        position_factor = 1.0 - (normalized_distance * 0.4) # Max reduction 0.4 at corners
        position_factor = np.clip(position_factor, 0.6, 1.0)

        # Base directionality based on Hall Type
        hall_base_directionality = {
            "Plate": 0.9,
            "Room": 0.6,
            "Cathedral": 0.2
        }
        base_directionality = hall_base_directionality.get(hall_type, 0.6) # Default to Room

        # Apply position factor to the base hall directionality
        directionality_pos_hall = base_directionality * position_factor

        # Boost based on DryWet for clarity when effect is prominent
        boost = 0.0
        if dry_wet > 0.6:
            boost = (dry_wet - 0.6) * 0.4 # Max boost +0.16 at DryWet=1.0

        # Combine and clamp final directionality
        final_directionality = directionality_pos_hall + boost
        final_directionality = np.clip(final_directionality, 0.05, 0.95) # Final clamp (ensure not fully 0 or 1)

        # print(f"  [Directionality] Pos=({x_pos:.2f},{y_pos:.2f}), Hall={hall_type}, DryWet={dry_wet:.2f} -> "
        #       f"Dist={normalized_distance:.2f}, PosFactor={position_factor:.2f}, HallBase={base_directionality:.2f}, "
        #       f"Boost={boost:.2f}, Final={final_directionality:.2f}")

        return final_directionality

    except (ValueError, TypeError) as e:
        print(f"Error in compute_final_directionality (x={x_pos}, y={y_pos}, hall={hall_type}): {e}. Returning default 0.5")
        return 0.5
    except Exception as e:
        print(f"Unexpected error in compute_final_directionality: {e}")
        traceback.print_exc()
        return 0.5


# --- Surround Panning Function (Unchanged Logic, added validation) ---
def apply_surround_panning(audio_data, x_pos, y_pos):
    """Distributes a stereo signal to 5.1 channels based on X/Y position."""
    if audio_data is None or audio_data.size == 0:
        print("Warning: apply_surround_panning received empty data.")
        return np.zeros((0, 6), dtype=np.float32) # Return empty 6-channel array

    try:
        # Validate and clamp position inputs rigorously
        try:
            x_float = float(x_pos)
            y_float = float(y_pos)
            x_clamped = np.clip(x_float, 0.0, 1.0)
            y_clamped = np.clip(y_float, 0.0, 1.0)
        except (ValueError, TypeError):
             print(f"Warning: Invalid position values ('{x_pos}', '{y_pos}') in panning. Using center (0.5, 0.5).")
             x_clamped, y_clamped = 0.5, 0.5


        # Ensure input is stereo float32
        if audio_data.ndim == 1:
            audio_data = np.stack((audio_data, audio_data), axis=1)
        elif audio_data.shape[1] == 1:
            audio_data = np.repeat(audio_data, 2, axis=1)
        elif audio_data.shape[1] > 2:
            # print(f"Warning: Panning input has {audio_data.shape[1]} channels. Using first two.")
            audio_data = audio_data[:, :2]
        elif audio_data.shape[1] != 2:
             print(f"Error: Panning input has unexpected shape {audio_data.shape}. Cannot process.")
             # Return silent 6-channel audio matching input length
             return np.zeros((audio_data.shape[0], 6), dtype=np.float32)

        audio_data = audio_data.astype(np.float32)

        # Basic trigonometric panning might be more natural, but this is simpler
        gain_l = 1.0 - x_clamped
        gain_r = x_clamped
        gain_f = 1.0 - y_clamped
        gain_re = y_clamped

        fl = gain_l * gain_f
        fr = gain_r * gain_f
        rl = gain_l * gain_re
        rr = gain_r * gain_re
        # Center channel gain decreases as source moves away from center X
        center_x_factor = 1.0 - (abs(0.5 - x_clamped) * 2)
        center = np.clip(center_x_factor, 0, 1) * gain_f # Center only gets front signal portion
        # LFE gets a small, constant portion of the combined signal's energy (use RMS maybe?)
        lfe_gain = 0.15 # LFE gain factor - reduced a bit
        # Calculate LFE based on combined input signal power (RMS) to make it less dependent on pure level
        mono_mix_for_lfe = (audio_data[:, 0] + audio_data[:, 1]) * 0.5
        # rms_lfe = np.sqrt(np.mean(mono_mix_for_lfe**2)) # Calculate RMS over the whole signal
        lfe_signal = mono_mix_for_lfe * lfe_gain # Apply gain to the mono mix directly

        surround_data = np.zeros((audio_data.shape[0], 6), dtype=audio_data.dtype)
        surround_data[:, 0] = audio_data[:, 0] * fl       # Front Left (FL)
        surround_data[:, 1] = audio_data[:, 1] * fr       # Front Right (FR)
        surround_data[:, 2] = mono_mix_for_lfe * center   # Center (C) - based on mono mix * center factor
        surround_data[:, 3] = lfe_signal                  # LFE - gained mono mix
        surround_data[:, 4] = audio_data[:, 0] * rl       # Rear Left (RL / SL)
        surround_data[:, 5] = audio_data[:, 1] * rr       # Rear Right (RR / SR)

        # Normalize to prevent clipping (simple peak normalization across all 6 channels)
        max_val = np.max(np.abs(surround_data))
        if max_val > 1.0: # Only normalize if clipping occurs
            # print(f"  Panning peak detected: {max_val:.2f}. Normalizing.")
            surround_data /= max_val
        elif max_val < 1e-6:
             # print("  Panning output is silent or near-silent.")
             pass # Don't divide by zero


        return surround_data.astype(np.float32)

    except Exception as e:
        print(f"Unexpected error in apply_surround_panning: {e}")
        traceback.print_exc()
        # Fallback: return silent 6-channel audio of the same length
        if audio_data is not None and audio_data.size > 0:
             return np.zeros((audio_data.shape[0], 6), dtype=np.float32)
        else:
             return np.zeros((0, 6), dtype=np.float32) # Empty if input was empty


# --- Visualizer: Wellenform + Spektrogramm anzeigen (MODIFIED for Multi-Channel v2) ---
def plot_waveform_and_spectrogram(file_path, title="Audio"):
    """
    Loads audio using Pydub, plots waveform for ALL channels (up to 6 in grid)
    and spectrogram for the first channel. Handles errors gracefully. Corrected axis sharing.
    """
    actual_path = getattr(file_path, 'name', file_path)

    if actual_path is None or not isinstance(actual_path, str) or not os.path.exists(actual_path) or not os.path.isfile(actual_path):
         error_msg = f"Fehler: Datei nicht gefunden, ungültig oder kein Pfad:\n'{actual_path}'"
         print(f"Visualizer Error: {error_msg}")
         fig, ax = plt.subplots(1, 1, figsize=(10, 3))
         ax.text(0.5, 0.5, error_msg, horizontalalignment='center', verticalalignment='center', color='red', fontsize=10, wrap=True)
         ax.set_axis_off()
         tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
         plt.savefig(tmp_plot.name)
         plt.close(fig)
         return tmp_plot.name

    try:
        print(f"Visualizer: Loading audio from '{actual_path}' using Pydub...")
        try:
            # ... (Audio loading code remains the same as before) ...
            file_ext = os.path.splitext(actual_path)[1].lower()
            if file_ext == ".wav": audio = AudioSegment.from_wav(actual_path)
            elif file_ext == ".mp3": audio = AudioSegment.from_mp3(actual_path)
            else: audio = AudioSegment.from_file(actual_path)

            rate = audio.frame_rate
            channels = audio.channels
            samples = np.array(audio.get_array_of_samples())

            if channels > 0 and samples.size > 0:
                 samples = samples.reshape((-1, channels))
            elif samples.size == 0:
                 print("  Warning: Loaded audio appears empty.")
                 samples = np.zeros((0, channels if channels > 0 else 1))
            else:
                 print("  Warning: Could not reshape samples array correctly.")
                 samples = np.zeros((0, 1))

            if audio.sample_width == 2: norm_factor = 32768.0
            elif audio.sample_width == 4: norm_factor = 2147483648.0
            elif audio.sample_width == 1: samples = samples.astype(np.float32) - 128.0; norm_factor = 128.0
            else: norm_factor = np.max(np.abs(samples)) if np.any(samples) else 1.0
            if norm_factor < 1e-6: norm_factor = 1.0
            data_float = samples.astype(np.float32) / norm_factor
            print(f"  Pydub loaded: {len(audio)/1000:.2f}s, {rate} Hz, {channels} channels, Sample Width={audio.sample_width}")

        except FileNotFoundError:
             print(f"Visualizer Error: File not found during Pydub load: '{actual_path}'")
             raise
        except Exception as load_err:
            # ... (Error handling for loading remains the same) ...
             print(f"Visualizer Error: Pydub/FFmpeg failed to load file: {load_err}")
             error_detail = f"Ist FFmpeg verfügbar?\nDatei: {os.path.basename(actual_path)}\nDetails: {load_err}"
             gr.Error(f"Fehler beim Laden der Audiodatei. {error_detail}")
             fig, ax = plt.subplots(1, 1, figsize=(10, 3))
             ax.text(0.5, 0.5, f'Fehler beim Laden:\n{error_detail}',
                     horizontalalignment='center', verticalalignment='center', color='red', fontsize=9, wrap=True)
             ax.set_axis_off()
             tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
             plt.savefig(tmp_plot.name)
             plt.close(fig)
             return tmp_plot.name

        if data_float.size == 0:
            # ... (Handling for empty data remains the same) ...
            print("Visualizer: Audio data is empty, cannot plot.")
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.text(0.5, 0.5, f'Keine Audiodaten zum Plotten in:\n{os.path.basename(actual_path)}',
                    horizontalalignment='center', verticalalignment='center', color='orange', fontsize=10)
            ax.set_axis_off()
            tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(tmp_plot.name)
            plt.close(fig)
            return tmp_plot.name

        # --- Prepare Data for Plotting ---
        num_channels = data_float.shape[1]
        channel_names = ["FL", "FR", "C", "LFE", "RL", "RR"]
        if num_channels > 0:
            spec_data = data_float[:, 0].flatten()
            spec_title = f"Spektrogramm ({channel_names[0] if num_channels > 0 else 'Mono'})"
        else:
            spec_data = data_float.flatten()
            spec_title = "Spektrogramm (Datenfehler)"

        max_abs = np.max(np.abs(data_float))
        if max_abs > 1e-6: data_plot = data_float / max_abs
        else: data_plot = data_float

        # --- Create Figure (4 rows, 2 columns layout) ---
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 2], hspace=0.4, wspace=0.15)

        # --- Set Figure Title ---
        if num_channels == 1: plot_title = f"Audioanalyse: {title} - {os.path.basename(actual_path)} (Mono)"
        else: plot_title = f"Audioanalyse: {title} - {os.path.basename(actual_path)} ({num_channels}-Kanal)"
        fig.suptitle(plot_title, fontsize=14)

        # --- Waveform Plots (Grid Layout with CORRECT axis sharing) ---
        time_axis = np.linspace(0, data_plot.shape[0]/rate, num=data_plot.shape[0])
        waveform_axs = [] # Store all waveform axes
        base_ax = None    # Keep track of the first axis for sharing

        for i in range(min(num_channels, 6)):
            row = i // 2
            col = i % 2

            # Share X axis with base_ax if it exists, otherwise this is the base_ax
            ax = fig.add_subplot(gs[row, col], sharex=base_ax if base_ax else None)
            waveform_axs.append(ax)
            if base_ax is None: # First axis created becomes the base for sharing
                base_ax = ax

            chan_name = channel_names[i] if i < len(channel_names) else f"Kanal {i+1}"
            ax.plot(time_axis, data_plot[:, i], label=chan_name, lw=1)
            ax.set_title(f"{chan_name}", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(axis='y', labelsize='x-small')
            ax.set_ylim([-1.05, 1.05])
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
            if col == 0:
                ax.set_ylabel("Amplitude", fontsize='small')

            # Remove x-tick labels from all waveform plots except the bottom-most ones
            # and the spectrogram (which will be handled separately)
            if base_ax and ax is not base_ax: # Don't hide labels from the first axis initially
                plt.setp(ax.get_xticklabels(), visible=False)
            # Handle the base axis ticks later, depending on whether it's in the last row or not


        # --- Spectrogram Plot (Spanning bottom row) ---
        # Share X axis with the base_ax (first waveform plot)
        spec_ax = fig.add_subplot(gs[3, :], sharex=base_ax if base_ax else None)

        # Now, AFTER all axes are created and sharing is set up, hide tick labels correctly
        if base_ax: # Only if we have waveform plots
            # Hide x-tick labels from all waveform axes that are NOT in the last waveform row (row 2)
             for i, ax in enumerate(waveform_axs):
                  row = i // 2
                  if row < 2: # Hide for rows 0 and 1
                       plt.setp(ax.get_xticklabels(), visible=False)
                  else: # Ensure labels are visible for the last waveform row (row 2)
                       plt.setp(ax.get_xticklabels(), visible=True)
                       ax.tick_params(axis='x', labelsize='x-small') # Ensure correct size


        if spec_data.size > 0 and rate > 0:
            # ... (Spectrogram calculation code remains the same) ...
            nperseg = min(4096, max(256, spec_data.shape[0] // 100 if spec_data.shape[0] > 20000 else 256))
            noverlap = nperseg // 2

            try:
                f, t, Sxx = spectrogram(spec_data, fs=rate, nperseg=nperseg, noverlap=noverlap, window='hann')
                if Sxx.size == 0: raise ValueError("Spectrogram resulted in empty array (signal likely too short or parameters invalid).")

                Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
                vmin = np.max(Sxx_db) - 80
                vmax = np.max(Sxx_db)
                if vmin >= vmax: vmin = vmax - 10

                img = spec_ax.pcolormesh(t, f, Sxx_db, shading='auto', cmap='magma',
                                       vmin=vmin, vmax=vmax, rasterized=True)
                spec_ax.set_title(spec_title, fontsize=12)
                spec_ax.set_ylabel('Frequenz (Hz)')
                spec_ax.set_xlabel('Zeit (s)')
                spec_ax.set_yscale('symlog', linthresh=100, linscale=0.5)
                spec_ax.tick_params(axis='both', labelsize='small')
                cbar = fig.colorbar(img, ax=spec_ax, format='%+2.0f dB', pad=0.01, aspect=40)
                cbar.set_label('Intensität (dB)', size='small')
                cbar.ax.tick_params(labelsize='x-small')
            except ValueError as sve:
                # ... (Spectrogram error handling remains the same) ...
                 print(f"Error calculating spectrogram: {sve}")
                 spec_ax.text(0.5, 0.5, f'Spektrogramm nicht berechenbar.\n{sve}',
                            horizontalalignment='center', verticalalignment='center', color='orange', fontsize=9, transform=spec_ax.transAxes, wrap=True)
                 spec_ax.set_title("Spektrogramm", fontsize=12)
                 spec_ax.set_xlabel('Zeit (s)')
                 spec_ax.set_ylabel('Frequenz (Hz)')
            except Exception as spe:
                # ... (Spectrogram error handling remains the same) ...
                 print(f"Unexpected error during spectrogram calculation: {spe}")
                 traceback.print_exc()
                 spec_ax.text(0.5, 0.5, 'Fehler bei Spektrogramm-Berechnung.',
                            horizontalalignment='center', verticalalignment='center', color='red', fontsize=9, transform=spec_ax.transAxes)
                 spec_ax.set_title("Spektrogramm", fontsize=12)
                 spec_ax.set_xlabel('Zeit (s)')
                 spec_ax.set_ylabel('Frequenz (Hz)')
        else:
            # ... (Spectrogram empty data handling remains the same) ...
             spec_ax.text(0.5, 0.5, 'Keine gültigen Audiodaten für Spektrogramm.',
                        horizontalalignment='center', verticalalignment='center', transform=spec_ax.transAxes)
             spec_ax.set_title("Spektrogramm", fontsize=12)
             spec_ax.set_xlabel('Zeit (s)')
             spec_ax.set_ylabel('Frequenz (Hz)')


        # Adjust layout AFTER plotting
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save plot to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_plot:
            plt.savefig(tmp_plot.name, dpi=120)
            plot_path = tmp_plot.name
        plt.close(fig)
        return plot_path

    except FileNotFoundError:
        # ... (existing error handling) ...
        error_msg = f"Fehler: Eingabedatei\n'{actual_path}'\nwurde nicht gefunden."
        print(f"Visualizer Error: {error_msg}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.text(0.5, 0.5, error_msg, horizontalalignment='center', verticalalignment='center', color='red', fontsize=10, wrap=True)
        ax.set_axis_off()
        tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmp_plot.name)
        plt.close(fig)
        return tmp_plot.name
    except Exception as e:
        # ... (existing error handling) ...
        print(f"Visualizer Error: Unexpected error plotting '{actual_path}': {e}")
        traceback.print_exc()
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.text(0.5, 0.5, f'Allgemeiner Fehler beim Plotten:\n{e}',
                horizontalalignment='center', verticalalignment='center', color='red', fontsize=9, wrap=True)
        ax.set_axis_off()
        tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmp_plot.name)
        plt.close(fig)
        return tmp_plot.name


# --- Main Processing Function (MODIFIED TO USE SPLIT IR, NEW directionality, ADAPTIVE LEVELS, and DYNAMIC DRY/WET) ---
def apply_raytrace_convolution_split(
    audio_file_path,
    hall_type_val,      # Input: Selected hall type (e.g., "Room")
    base_early_level,   # Input: Base early level from slider
    base_late_level,    # Input: Base late level from slider
    dry_wet, bass_gain, treble_gain,
    x_pos, y_pos, material,
    dry_wet_kill_start  # Input: Kill start value from slider
    ):
    """Loads audio, applies SPLIT reverb simulation based on hall type, position-dependent directionality,
       adaptive levels, dynamic dry/wet mix, EQ, panning and saves the 6-channel result."""
    if audio_file_path is None:
        print("Error: No audio file provided for processing.")
        gr.Warning("Keine Audioquelle ausgewählt.")
        return None, None # Indicate failure gracefully

    try:
        # --- START: EXPLIZITE KONVERTIERUNG DER EINGABEN ---
        try:
            # Konvertiere Positions- und Sliderwerte explizit zu float
            x_pos = float(x_pos)
            y_pos = float(y_pos)
            base_early_level = float(base_early_level)
            base_late_level = float(base_late_level)
            dry_wet = float(dry_wet)
            bass_gain = float(bass_gain)
            treble_gain = float(treble_gain)
            dry_wet_kill_start = float(dry_wet_kill_start)
            # hall_type_val und material bleiben Strings
        except (ValueError, TypeError) as e:
            # Gib eine detailliertere Fehlermeldung aus, falls die Konvertierung fehlschlägt
            print(f"--- ERROR: Failed to convert input parameters to numbers ---")
            print(f"Received Values & Types:")
            print(f"  x_pos: {x_pos} ({type(x_pos)})")
            print(f"  y_pos: {y_pos} ({type(y_pos)})")
            print(f"  base_early_level: {base_early_level} ({type(base_early_level)})")
            print(f"  base_late_level: {base_late_level} ({type(base_late_level)})")
            print(f"  dry_wet: {dry_wet} ({type(dry_wet)})")
            print(f"  bass_gain: {bass_gain} ({type(bass_gain)})")
            print(f"  treble_gain: {treble_gain} ({type(treble_gain)})")
            print(f"  dry_wet_kill_start: {dry_wet_kill_start} ({type(dry_wet_kill_start)})")
            print(f"  Error Details: {e}")
            print(f"-----------------------------------------------------------")
            gr.Error(f"Fehler bei der Konvertierung von Eingabewerten für die Verarbeitung. Details siehe Konsole. Fehler: {e}")
            return None, None # Verhindere weitere Ausführung
        # --- ENDE: EXPLIZITE KONVERTIERUNG DER EINGABEN ---

        file_path = getattr(audio_file_path, 'name', audio_file_path)
        if not isinstance(file_path, str) or not os.path.exists(file_path):
             print(f"Error: Audio file not found or invalid path: {file_path}")
             gr.Error(f"Audiodatei nicht gefunden: {file_path}")
             return None, None

        # --- Load Audio using Pydub ---
        print(f"Processing: Loading audio '{os.path.basename(file_path)}'")
        try:
             # Determine format for explicit loading if possible
             file_ext = os.path.splitext(file_path)[1].lower()
             if file_ext == ".wav": audio = AudioSegment.from_wav(file_path)
             elif file_ext == ".mp3": audio = AudioSegment.from_mp3(file_path)
             # Add other formats as needed
             else: audio = AudioSegment.from_file(file_path) # Generic fallback

        except Exception as load_err:
             print(f"Pydub/FFmpeg error loading file: {load_err}")
             gr.Error(f"Fehler beim Laden der Audiodatei '{os.path.basename(file_path)}'. Ist FFmpeg korrekt installiert und im Pfad? Details: {load_err}")
             return None, None

        rate = audio.frame_rate
        channels = audio.channels
        print(f"  Audio Info: {len(audio)/1000:.2f}s, {rate} Hz, {channels} ch")

        samples = np.array(audio.get_array_of_samples())
        if channels > 0: samples = samples.reshape((-1, channels))
        else: samples = np.zeros((0,1)) # Handle empty/corrupt

        if audio.sample_width == 2: norm_factor = 32768.0
        elif audio.sample_width == 4: norm_factor = 2147483648.0
        elif audio.sample_width == 1: samples = samples.astype(np.float32) - 128.0; norm_factor = 128.0
        else: norm_factor = np.max(np.abs(samples)) if np.any(samples) else 1.0
        if norm_factor < 1e-6: norm_factor = 1.0
        samples_float = samples.astype(np.float32) / norm_factor

        # --- Process Audio using SPLIT IR ---
        # Diese Zeile sollte jetzt funktionieren, da x_pos und y_pos floats sind
        print(f"Processing Steps for Hall '{hall_type_val}', Pos ({x_pos:.2f},{y_pos:.2f}), Mat '{material}'")

        # 1. Get adjusted parameters based on Hall Type
        adj_ir_duration, adj_reflection_count, adj_max_delay, adj_early_split_time = adjust_reverb_parameters_by_hall(hall_type_val)
        print(f"  1. Hall Params: Dur={adj_ir_duration:.2f}s, RefCount={adj_reflection_count}, MaxERDelay={adj_max_delay:.3f}s, ERSplitT={adj_early_split_time:.3f}s")

        # 2. Calculate Final Directionality based on position, hall type and dry/wet
        directionality = compute_final_directionality(x_pos, y_pos, hall_type_val, dry_wet)
        print(f"  2. Directionality: {directionality:.3f}")

        # 3. Generate Split Impulse Responses using adjusted params and directionality
        print(f"  3. Generating Split IR...")
        early_ir, late_ir = generate_impulse_response_split(
            rate,
            adj_ir_duration,        # Use adjusted value from hall type
            adj_reflection_count,   # Use adjusted value
            adj_max_delay,          # Use adjusted value
            material,               # Use material from UI
            directionality=directionality, # Use the calculated final directionality
            early_late_split=adj_early_split_time # Use adjusted split time
        )
        print(f"     Generated Early IR (len={len(early_ir)}, max={np.max(np.abs(early_ir)):.3f}), Late IR (len={len(late_ir)}, max={np.max(np.abs(late_ir)):.3f})")

        # 4. Calculate Adaptive Early/Late Balance based on DryWet and BASE levels from UI
        adapted_early_level, adapted_late_level = adapt_early_late_levels(
            dry_wet,            # Current Dry/Wet value
            base_early_level,   # BASE Early level from slider
            base_late_level     # BASE Late level from slider
        )
        print(f"  4. Adaptive Levels: Early={adapted_early_level:.3f}, Late={adapted_late_level:.3f} (DryWet={dry_wet:.2f})")


        # 5. Apply Split Convolution, EQ, and Dynamic Mix using ADAPTED levels and DYNAMIC Dry/Wet
        print(f"  5. Applying Convolution, Mix (DryKillStart={dry_wet_kill_start:.2f}), EQ (B={bass_gain:.2f}, T={treble_gain:.2f})...")
        # Check if IRs are essentially silent before convolution
        run_convolution = np.any(early_ir) or np.any(late_ir)
        if not run_convolution:
            print("     Skipping convolution as both IRs are silent.")
            # If IRs are silent, the 'wet' signal is zero, dynamic_dry_wet_mix handles this
            output_stereo = dynamic_dry_wet_mix(samples_float, np.zeros_like(samples_float), dry_wet, dry_wet_kill_start)
            # Apply EQ even if reverb is off (acts on the potentially remaining dry signal)
            # Reuse EQ logic from convolve_audio_split for consistency
            try:
                if not np.isclose(bass_gain, 1.0) or not np.isclose(treble_gain, 1.0):
                     fft_data = np.fft.rfft(output_stereo, axis=0)
                     rate_for_freq = rate if rate > 0 else 44100
                     n_fft = output_stereo.shape[0]
                     if n_fft >= 2:
                         freqs = np.fft.rfftfreq(n_fft, d=1.0/rate_for_freq)
                         bass_cutoff = 250; treble_cutoff = 4000
                         bass_mask = (freqs > 1e-6) & (freqs <= bass_cutoff); treble_mask = freqs >= treble_cutoff
                         fft_data[bass_mask] *= np.clip(bass_gain, 0.1, 5.0)
                         fft_data[treble_mask] *= np.clip(treble_gain, 0.1, 5.0)
                         output_stereo = np.fft.irfft(fft_data, n=n_fft, axis=0).astype(np.float32)
            except Exception as eq_err:
                 print(f"     Error applying EQ to dry signal: {eq_err}")
            # Normalize final output
            max_val = np.max(np.abs(output_stereo))
            if max_val > 1e-6: output_stereo /= max_val

        else:
            # Proceed with full convolution if IRs are present
            output_stereo = convolve_audio_split(
                data=samples_float,
                early_ir=early_ir,
                late_ir=late_ir,
                early_level=adapted_early_level,  # Use adaptive level
                late_level=adapted_late_level,    # Use adaptive level
                dry_wet=dry_wet,                  # Pass dry_wet for mixing
                bass_gain=bass_gain,
                treble_gain=treble_gain,
                rate=rate,
                kill_start_dw=dry_wet_kill_start  # Pass the kill start value here
            )

        print(f"     Convolution/Mix/EQ completed. Output shape: {output_stereo.shape}")


        # 6. Apply Surround Panning
        print(f"  6. Applying Surround Panning (X={x_pos:.2f}, Y={y_pos:.2f})...")
        # Stelle sicher, dass apply_surround_panning auch mit floats umgehen kann (sollte es bereits tun)
        surround_output = apply_surround_panning(output_stereo, x_pos, y_pos)
        print(f"     Panning completed. Output shape: {surround_output.shape}")


        # --- Save Result ---
        print("Processing: Saving processed 6-channel WAV...")
        # Create a temporary file path first using a context manager for safety
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile_obj:
            tmpfile_path = tmpfile_obj.name

        # Ensure data is clipped JUST before converting to int16
        # Use a slightly smaller range to avoid potential issues with exact 1.0/-1.0 mapping
        surround_output_clamped = np.clip(surround_output, -0.9999, 0.9999)
        # Check for NaN or Inf values before conversion (can happen with extreme EQ/IRs)
        if not np.all(np.isfinite(surround_output_clamped)):
             print("Warning: Non-finite values detected in output signal! Clipping to zero.")
             surround_output_clamped = np.nan_to_num(surround_output_clamped, nan=0.0, posinf=0.0, neginf=0.0)

        surround_output_int16 = (surround_output_clamped * 32767).astype(np.int16)

        # Write the int16 data to the temporary file
        try:
            wavfile.write(tmpfile_path, rate, surround_output_int16)
            print(f"  Successfully saved to temporary file: {tmpfile_path}")
            # Return the path for both player and download link
            return tmpfile_path, tmpfile_path
        except Exception as write_err:
             print(f"Error writing WAV file: {write_err}")
             traceback.print_exc()
             gr.Error(f"Fehler beim Schreiben der WAV-Datei: {write_err}")
             # Clean up temp file if write failed
             if os.path.exists(tmpfile_path): os.remove(tmpfile_path)
             return None, None

    # Fange alle anderen unerwarteten Fehler ab
    except Exception as e:
        print(f"--- UNEXPECTED ERROR in apply_raytrace_convolution_split ---")
        print(f"File Path: {audio_file_path}")
        # Drucke die Werte, mit denen die Funktion aufgerufen wurde, um das Debugging zu erleichtern
        print(f"Parameters: hall='{hall_type_val}', early={base_early_level}, late={base_late_level}, dw={dry_wet}, "
              f"bass={bass_gain}, treble={treble_gain}, x={x_pos} ({type(x_pos)}), y={y_pos} ({type(y_pos)}), "
              f"mat='{material}', kill_start={dry_wet_kill_start}")
        print(f"Error Details: {e}")
        traceback.print_exc() # Detaillierter Stacktrace
        print(f"-----------------------------------------------------------")
        gr.Error(f"Ein unerwarteter Fehler ist bei der Audioverarbeitung aufgetreten. Details siehe Konsole. Fehler: {e}")
        return None, None # Sichere Rückgabe bei Fehlern


# === NEUE FUNKTIONEN für Audio Profiler (Hinzugefügt) ===
def calculate_stereo_width_metric(left_channel, right_channel):
    """Berechnet eine Stereo-Breiten-Metrik (RMS des Side-Signals)."""
    if left_channel.size != right_channel.size or left_channel.size == 0: return 0.0
    side_signal = (left_channel - right_channel) / 2.0; rms_side = np.sqrt(np.mean(side_signal**2))
    rms_left = np.sqrt(np.mean(left_channel**2)); rms_right = np.sqrt(np.mean(right_channel**2))
    if rms_left < 1e-9 and rms_right < 1e-9: return 0.0 # Silent
    if rms_side < 1e-7 * (rms_left + rms_right): return 0.0 # Effectively Mono
    return rms_side

def run_audio_profiler(original_file_obj, processed_file_obj):
    """Analysiert Original vs. Bearbeitet und erstellt Bericht."""
    report_lines = ["## 📊 Audio-Profiler Bericht"]; results = {}
    original_path = getattr(original_file_obj, 'name', None)
    processed_path = getattr(processed_file_obj, 'name', None)
    # Validierung
    if not original_path or not os.path.exists(original_path): report_lines.append("\n**Fehler:** Originaldatei fehlt."); return "\n".join(report_lines)
    if not processed_path or not os.path.exists(processed_path): report_lines.append("\n**Fehler:** Bearbeitete Datei fehlt."); return "\n".join(report_lines)
    # Laden (soundfile)
    try:
        print(f"Profiler: Loading Original '{os.path.basename(original_path)}'")
        data_orig, rate_orig = sf.read(original_path, dtype='float32')
        if data_orig.ndim == 1: data_orig = data_orig[:, np.newaxis]
        channels_orig = data_orig.shape[1]
        print(f"  Original Loaded: {data_orig.shape[0]/rate_orig:.2f}s, {rate_orig}Hz, {channels_orig}ch")

        print(f"Profiler: Loading Processed '{os.path.basename(processed_path)}'")
        data_proc, rate_proc = sf.read(processed_path, dtype='float32')
        if data_proc.ndim == 1: data_proc = data_proc[:, np.newaxis]
        channels_proc = data_proc.shape[1]
        print(f"  Processed Loaded: {data_proc.shape[0]/rate_proc:.2f}s, {rate_proc}Hz, {channels_proc}ch")

    except Exception as e: report_lines.append(f"\n**Ladefehler:**\n```\n{traceback.format_exc()}\n```"); return "\n".join(report_lines)
    # Checks
    if rate_orig != rate_proc: report_lines.append(f"\n**Fehler:** Sample-Raten unterschiedlich ({rate_orig} Hz vs {rate_proc} Hz)."); return "\n".join(report_lines)
    rate = rate_orig
    # Lautheit (LUFS)
    report_lines.append("\n### 🔊 Lautheit (Integrierte LUFS)")
    try:
        meter = pyln.Meter(rate)
        data_lufs_orig = data_orig[:, 0] if channels_orig == 1 else np.mean(data_orig[:, :min(2, channels_orig)], axis=1)
        loudness_orig = meter.integrated_loudness(data_lufs_orig); results['lufs_orig'] = loudness_orig
        report_lines.append(f"- Original: {loudness_orig:.2f} LUFS")
        if channels_proc >= 2:
            data_lufs_proc = np.mean(data_proc[:, :2], axis=1) # Use FL/FR
            loudness_proc = meter.integrated_loudness(data_lufs_proc); results['lufs_proc'] = loudness_proc
            lufs_diff = loudness_proc - loudness_orig; results['lufs_diff'] = lufs_diff
            report_lines.append(f"- Bearbeitet: {loudness_proc:.2f} LUFS")
            report_lines.append(f"- **Änderung:** {lufs_diff:+.2f} LU")
        else: report_lines.append("- Bearbeitet: < 2 Kanäle, LUFS nicht verglichen")
    except Exception as e: report_lines.append(f"- Fehler Lautheit: {e}")
    # Stereo-Breite
    report_lines.append("\n### ↔️ Stereo-Breite (FL/FR)")
    try:
        width_metric_orig = 0.0; width_metric_proc = 0.0
        orig_is_stereo = channels_orig >= 2; proc_is_stereo = channels_proc >= 2
        if orig_is_stereo:
            width_metric_orig = calculate_stereo_width_metric(data_orig[:, 0], data_orig[:, 1]); results['width_orig'] = width_metric_orig
            report_lines.append(f"- Original (Side RMS): {width_metric_orig:.4f}")
        else: report_lines.append("- Original: Mono")
        if proc_is_stereo:
            width_metric_proc = calculate_stereo_width_metric(data_proc[:, 0], data_proc[:, 1]); results['width_proc'] = width_metric_proc
            report_lines.append(f"- Bearbeitet (Side RMS): {width_metric_proc:.4f}")
        else: report_lines.append("- Bearbeitet: Mono")
        # Prozentuale Änderung
        if orig_is_stereo and proc_is_stereo and width_metric_orig > 1e-9:
            width_change_percent = ((width_metric_proc / width_metric_orig) - 1) * 100; results['width_change_percent'] = width_change_percent
            report_lines.append(f"- **Änderung:** {width_change_percent:+.1f}%")
        elif proc_is_stereo and not orig_is_stereo: report_lines.append("- **Änderung:** Von Mono zu Stereo")
        elif not proc_is_stereo and orig_is_stereo: report_lines.append("- **Änderung:** Von Stereo zu Mono")
        elif not orig_is_stereo and not orig_is_stereo: report_lines.append("- **Änderung:** Beide Mono")
    except Exception as e: report_lines.append(f"- Fehler Breite: {e}")
    # LFE-Analyse
    report_lines.append("\n###  subwoofer LFE-Kanal Energie (Bearbeitet)")
    try:
        lfe_idx = 3 # Standard 5.1 index for LFE
        if channels_proc > lfe_idx:
            lfe_channel = data_proc[:, lfe_idx]; rms_lfe = np.sqrt(np.mean(lfe_channel**2)); results['lfe_rms'] = rms_lfe
            if rms_lfe > 1e-15: # Avoid log10(0) or near zero
                dbfs_lfe = 20 * math.log10(rms_lfe); results['lfe_dbfs'] = dbfs_lfe
                report_lines.append(f"- Pegel (ca.): {dbfs_lfe:.1f} dBFS")
            else: report_lines.append("- Pegel: Nahezu Stille"); results['lfe_dbfs'] = -np.inf
            # Energie > 50 Hz (grob)
            try:
                freqs_fft = np.fft.rfftfreq(lfe_channel.shape[0], d=1.0/rate); lfe_fft = np.fft.rfft(lfe_channel)
                energy_total = np.sum(np.abs(lfe_fft)**2); energy_above_50hz = np.sum(np.abs(lfe_fft[freqs_fft > 50])**2)
                if energy_total > 1e-12:
                    percent_above_50hz = (energy_above_50hz / energy_total) * 100; results['lfe_perc_above_50hz'] = percent_above_50hz
                    report_lines.append(f"- Energie > 50 Hz (approx.): {percent_above_50hz:.1f}%")
                    if percent_above_50hz > 30: report_lines.append("  ⚠️ *Hinweis: Relativ hoher Anteil > 50 Hz im LFE.*")
                else: report_lines.append("- Energie > 50 Hz: Nicht berechenbar (LFE zu leise).")
            except Exception as e_fft: report_lines.append(f"- Fehler LFE Frequenzanalyse: {e_fft}")
        else: report_lines.append("- LFE-Kanal nicht vorhanden (Datei hat < 4 Kanäle).")
    except Exception as e: report_lines.append(f"- Fehler LFE Analyse: {e}")
    # Zusammenfassung
    report_lines.append("\n### 📜 Zusammenfassung")
    summary = "Die bearbeitete Datei zeigt "; changes = []
    if 'lufs_diff' in results:
        lufs_val = results['lufs_diff']
        if abs(lufs_val) < 0.1: changes.append("eine kaum veränderte Lautheit")
        elif lufs_val > 0: changes.append(f"eine Lautheitssteigerung von {lufs_val:.1f} LU")
        else: changes.append(f"einen Lautheitsverlust von {abs(lufs_val):.1f} LU")
    if 'width_change_percent' in results:
        width_val = results['width_change_percent']
        if abs(width_val) < 5: changes.append("eine kaum veränderte Stereo-Breite")
        elif width_val > 0: changes.append(f"eine um {width_val:.0f}% erhöhte Stereo-Breite")
        else: changes.append(f"eine um {abs(width_val):.0f}% reduzierte Stereo-Breite")
    elif results.get('width_proc', 0) > 0 and results.get('width_orig', 0) == 0: changes.append("eine Verbreiterung von Mono zu Stereo")
    if 'lfe_dbfs' in results:
        lfe_db = results['lfe_dbfs']
        if lfe_db > -15: changes.append(f"einen prominenten LFE-Kanal ({lfe_db:.0f} dBFS)")
        elif lfe_db > -35: changes.append(f"einen moderaten LFE-Kanal ({lfe_db:.0f} dBFS)")
        elif not np.isinf(lfe_db): changes.append(f"einen leisen LFE-Kanal ({lfe_db:.0f} dBFS)")
    if not changes: summary += "kaum messbare Veränderungen gegenüber dem Original."
    else: summary += ", ".join(changes) + "."
    report_lines.append(summary)
    return "\n".join(report_lines)
# === ENDE NEUE FUNKTIONEN Audio Profiler ===


# --- Gradio UI Definition ---
# Define theme outside the block for potential reuse
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate
).set(
    # Customize specific component styles if needed
    button_primary_background_fill="#007bff", # Example: Bootstrap primary blue
    button_primary_text_color="white",
)

# --- UI START ---
with gr.Blocks(theme=theme, title="Audio Raytracing Studio v3.5 (Profiler)") as demo:

    # --- Define Shared Components ---
    # (No global shared needed for this specific integration)

    # --- Define UI Tabs ---
    with gr.Tab("🎶 Audio-Verarbeitung"):
        gr.Markdown("# 🎶 Audio Raytracing Studio v3.5 (Profiler)\n### Lade Audio hoch oder nutze das Mikrofon, wähle Einstellungen und Position.")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="🔊 Audio hochladen (WAV/MP3)", type="filepath", show_download_button=False)
                mic_input = gr.Audio(label="🎤 Mikrofonaufnahme", sources=["microphone"], type="filepath", show_download_button=False)
            with gr.Column(scale=1):
                # Use format="wav" to ensure browser compatibility for playback where possible
                output_audio = gr.Audio(label="🎧 Ergebnis anhören (6-Kanal WAV)", type="filepath", interactive=False, format="wav")
                download = gr.File(label="💾 Verarbeitetes Audio herunterladen (6-Kanal WAV)", interactive=False)

        with gr.Row():
             with gr.Column(scale=1):
                gr.Markdown("#### Raum & Reflexionen")
                # --- NEW: Hall Type Selection ---
                hall_type = gr.Dropdown(
                    choices=["Plate", "Room", "Cathedral"],
                    label="🏩️ Hall-Typ auswählen",
                    value=DEFAULT_HALL_TYPE,
                    info="Bestimmt Grundcharakter, interne Reverb-Parameter und Basis-Diffusität.",
                    interactive=True
                )
                hall_info_text = gr.Markdown(update_hall_info(DEFAULT_HALL_TYPE), elem_id="hall-info-md") # Initial description

                # --- OLD Sliders (Hidden, values derived from Hall Type) ---
                ir_duration = gr.Slider(0.01, 5.0, value=1.5, step=0.05, label="Nachhallzeit (Basis)", info="Basis-Dauer der IR (wird vom Hall-Typ überschrieben!).", visible=False)
                reflection_count = gr.Slider(1, 60, value=35, step=1, label="Reflexionen (Basis)", info="Basis-Anzahl Reflexionen (wird vom Hall-Typ überschrieben!).", visible=False)
                max_delay = gr.Slider(0.005, 0.25, value=0.06, step=0.001, label="Max. Verzögerung (Basis)", info="Basis max. Delay (wird vom Hall-Typ überschrieben!).", visible=False)
                # --- Material Choice ---
                material_choice = gr.Dropdown(choices=list(material_absorption.keys()), value=DEFAULT_MATERIAL, label="🧱 Material", info="Beeinflusst Dämpfung der Reflexionen.")

             with gr.Column(scale=1):
                gr.Markdown("#### Mix & EQ")
                 # --- NEW: Early/Late Levels (Base levels, adapted dynamically) ---
                early_level = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Basis Early Level", info="Grundlautstärke früher Reflexionen (dynamisch angepasst durch Dry/Wet).")
                late_level = gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="Basis Late Level", info="Grundlautstärke des Nachhalls (dynamisch angepasst durch Dry/Wet).")
                # --- Existing Mix/EQ ---
                dry_wet = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Dry/Wet Mix", info="0=Original, 1=Nur Effekt. Passt Early/Late Balance & Dry-Signal dynamisch an.")
                # --- ADDED: Control for Dynamic Dry/Wet Kill Start ---
                dry_wet_kill_start_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Dry Kill Start", info="Dry/Wet-Wert, ab dem Dry-Signal ausgeblendet wird (0=sofort, 1=nie).")

                bass_gain = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Bass Gain", info="EQ: Verstärkung/Absenkung tiefer Frequenzen (<250Hz). Vorsicht bei hohen Werten!")
                treble_gain = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Treble Gain", info="EQ: Verstärkung/Absenkung hoher Frequenzen (>4kHz). Vorsicht bei hohen Werten!")
                process_button = gr.Button("➡️ Verarbeiten & Anhören!", variant="primary", scale=2)


    with gr.Tab("📡 5.1 Surround Map"):
        gr.Markdown("## 📡 Position der Audioquelle im Raum auswählen")
        with gr.Row():
            with gr.Column(scale=2):
                # --- Setup Surround Map Image ---
                surround_layout_path = "surround_layout.png"
                # Check and potentially create placeholder ONCE at startup is better
                # For now, keep the check here for robustness if file disappears
                placeholder_created = False
                if not os.path.exists(surround_layout_path) or not os.path.isfile(surround_layout_path):
                    try:
                        print(f"Warning: '{surround_layout_path}' not found or invalid. Creating placeholder...")
                        img = Image.new('RGB', (600, 450), color = (210, 210, 225)) # Light grey-blue BG
                        d = ImageDraw.Draw(img)
                        # Speaker placeholders (simple rectangles)
                        sw, sh = 25, 15 # Speaker width, height
                        cx, cy = 300, 225 # Center approx
                        # Front
                        d.rectangle((cx-sw/2, 50, cx+sw/2, 50+sh), fill='darkgrey', outline='black') # C
                        d.text((cx, 50+sh+2), "C", fill="black", anchor="mt")
                        d.rectangle((100-sw/2, 50, 100+sw/2, 50+sh), fill='darkgrey', outline='black') # FL
                        d.text((100, 50+sh+2), "FL", fill="black", anchor="mt")
                        d.rectangle((500-sw/2, 50, 500+sw/2, 50+sh), fill='darkgrey', outline='black') # FR
                        d.text((500, 50+sh+2), "FR", fill="black", anchor="mt")
                        # Rear
                        d.rectangle((100-sw/2, 380, 100+sw/2, 380+sh), fill='darkgrey', outline='black') # RL
                        d.text((100, 380-2), "RL", fill="black", anchor="mb")
                        d.rectangle((500-sw/2, 380, 500+sw/2, 380+sh), fill='darkgrey', outline='black') # RR
                        d.text((500, 380-2), "RR", fill="black", anchor="mb")
                        # LFE (symbolic)
                        d.rectangle((cx-sw, cy-sh/2, cx+sw, cy+sh/2), fill='grey', outline='black') # LFE area near center
                        d.text((cx, cy), "LFE\n(Zone)", fill="black", anchor="mm", align="center")

                        d.text((10,10), "Layout 'surround_layout.png' fehlt - Placeholder", fill=(50,50,50))
                        # Save to a known temporary location if needed, but better to just save as surround_layout.png
                        img.save(surround_layout_path) # Overwrite/create the file
                        print(f"  Placeholder image saved as: {surround_layout_path}")
                        placeholder_created = True
                    except Exception as img_err:
                        print(f"Error creating placeholder image: {img_err}")
                        gr.Warning(f"Konnte Placeholder für '{surround_layout_path}' nicht erstellen: {img_err}")
                        surround_layout_path = None # Indicate failure

                # Use the potentially created placeholder path, or None if creation failed
                surround_image = gr.Image(
                    value=surround_layout_path, # Path to the image file
                    label="🎛️ Klicke auf die Karte oder bewege die Slider (X/Y Position)",
                    interactive=True,
                    type="filepath" # Keep as filepath
                )
                # This image will SHOW the marker, it's updated by events
                surround_output_image = gr.Image(
                    label="🎯 Gewählte Position (roter Marker)",
                    interactive=False,
                    type="filepath" # Keep as filepath
                )
            with gr.Column(scale=1):
                gr.Markdown("#### Numerische Position")
                surround_x = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="↔️ Links (0) / Rechts (1) Position (X)", interactive=True)
                surround_y = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="↕️ Vorne (0) / Hinten (1) Position (Y)", interactive=True)
                gr.Markdown("*(0,0 ≈ Vorne Links, 0.5,0 = Vorne Mitte, 1,1 ≈ Hinten Rechts)*\nBeeinflusst Panning und Hall-Gerichtetheit.")

        # --- Marker Update Functions (Improved Error Handling) ---
        # Store the original base image path globally or pass it reliably
        BASE_SURROUND_MAP_PATH = surround_layout_path

        def update_marker_image(x_pos, y_pos, base_image_path_param):
            """Draws marker on the base image and returns path to the new temporary image."""
            # Prioritize the globally stored path if the passed one is invalid somehow
            base_path = base_image_path_param
            if not base_path or not isinstance(base_path, str) or not os.path.exists(base_path):
                print(f"  Warning: Invalid base_image_path_param ('{base_path}'). Trying global path '{BASE_SURROUND_MAP_PATH}'.")
                base_path = BASE_SURROUND_MAP_PATH

            # Final check if we have a valid path to work with
            if not base_path or not isinstance(base_path, str) or not os.path.exists(base_path):
                 print(f"  Error: Cannot draw marker, base image path is invalid ('{base_path}').")
                 return None # Indicate failure to generate marker image

            try:
                # Validate position inputs
                try:
                    x_float = float(x_pos)
                    y_float = float(y_pos)
                except (ValueError, TypeError):
                     print(f"  Warning: Invalid position values for marker ('{x_pos}', '{y_pos}'). Using center (0.5, 0.5).")
                     x_float, y_float = 0.5, 0.5

                # print(f"Updating marker image for x={x_float:.2f}, y={y_float:.2f} on base: {base_path}")
                with Image.open(base_path) as bg:
                    bg = bg.convert("RGBA") # Ensure RGBA for transparency
                    img_width, img_height = bg.size
                    # Ensure width/height are positive before calculating pixels
                    if img_width <= 0 or img_height <= 0:
                        print(f"  Error: Invalid image dimensions ({img_width}x{img_height}) for marker drawing.")
                        return None

                    x_pixel = int(np.clip(x_float, 0, 1) * (img_width - 1)) # Use width-1 for index safety
                    y_pixel = int(np.clip(y_float, 0, 1) * (img_height - 1)) # Use height-1

                    output_img = bg.copy() # Work on a copy
                    draw = ImageDraw.Draw(output_img)
                    radius = max(6, min(img_width, img_height) // 50) # Slightly larger, adaptive radius
                    outline_width = max(1, radius // 4)

                    # Draw a clear red circle with a white outline
                    draw.ellipse(
                        (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius),
                        fill=(255, 0, 0, 190), # Semi-transparent red
                        outline=(255, 255, 255, 210), # Semi-transparent white outline
                        width=outline_width
                    )

                # Save the modified image to a new temporary file
                # Ensure the temp file is properly closed before returning its name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_marked_img_file:
                    output_img.save(tmp_marked_img_file.name, "PNG")
                    tmp_marked_img_path = tmp_marked_img_file.name
                # print(f"  Marker image saved to temporary file: {tmp_marked_img_path}")
                return tmp_marked_img_path

            except FileNotFoundError:
                print(f"Error critical: Background image '{base_path}' disappeared during marker drawing.")
                return None # Indicate failure
            except Exception as e_inner:
                print(f"Error during image manipulation in update_marker_image: {e_inner}")
                traceback.print_exc()
                return None # Indicate failure

        def update_controls_from_click(evt: gr.SelectData):
            """Handles clicks on the surround map image."""
            # Use the global path directly here, assuming it was set correctly at startup
            base_path = BASE_SURROUND_MAP_PATH
            if not base_path or not os.path.exists(base_path):
                 print("  Error: Cannot process click, base surround map path is invalid.")
                 # Return updates that don't change the current state
                 # Need 3 outputs: x_slider, y_slider, output_image
                 return gr.update(), gr.update(), gr.update()

            try:
                # Get image dimensions
                with Image.open(base_path) as img:
                    img_width, img_height = img.size

                if img_width <=0 or img_height <= 0:
                     print(" Error: Invalid image dimensions from base map.")
                     return gr.update(), gr.update(), gr.update()

                x_click, y_click = evt.index[0], evt.index[1]
                # Normalize click coordinates safely
                x_norm = np.clip(x_click / img_width, 0.0, 1.0)
                y_norm = np.clip(y_click / img_height, 0.0, 1.0)
                # print(f"  Click detected at ({x_click}, {y_click}), normalized to ({x_norm:.2f}, {y_norm:.2f})")

                # Update the marker image based on the new normalized coordinates
                new_marker_path = update_marker_image(x_norm, y_norm, base_path)

                # Return the new normalized coordinates for the sliders and the path to the new marker image
                # Handle case where marker generation failed
                marker_update = gr.update(value=new_marker_path) if new_marker_path else gr.update()
                return x_norm, y_norm, marker_update

            except Exception as e:
                 print(f"Error processing image click event: {e}")
                 traceback.print_exc()
                 # Return updates that don't change the current state on error
                 return gr.update(), gr.update(), gr.update()

        # --- Link UI Events for Surround Map ---
        # When the interactive image (surround_image) is clicked...
        surround_image.select(
            fn=update_controls_from_click,
            inputs=None, # No direct inputs needed if using global base path
            outputs=[surround_x, surround_y, surround_output_image] # Update sliders and the *output* image
        )

        # Function to handle slider changes and update the marker image
        def handle_slider_change(x_pos, y_pos):
             # Generate new marker image based on slider values using the base path
             new_marker_path = update_marker_image(x_pos, y_pos, BASE_SURROUND_MAP_PATH)
             # Update only the output image component
             # Handle case where marker generation failed
             return gr.update(value=new_marker_path) if new_marker_path else gr.update()


        # Link sliders' input/change events to update the marker image
        # Use 'input' for smoother updates as the slider is dragged
        surround_x.input(
             fn=handle_slider_change,
             inputs=[surround_x, surround_y], # Pass current slider values
             outputs=[surround_output_image] # Update the output image
        )
        surround_y.input(
             fn=handle_slider_change,
             inputs=[surround_x, surround_y], # Pass current slider values
             outputs=[surround_output_image] # Update the output image
        )


    with gr.Tab("📊 Visualizer"):
        gr.Markdown("## 📈 Vergleiche Wellenform und Frequenzverteilung")
        with gr.Row():
            # Use gr.File which allows selecting various types and handles paths better
            input_file_vis = gr.File(label="🔍 Original-Audio für Visualisierung auswählen (WAV/MP3 etc.)", file_types=['audio'])
            # Reference the download component directly for the processed file if desired, or allow separate upload
            # Let's allow separate selection for flexibility
            output_file_vis = gr.File(label="🔍 Verarbeitetes Audio für Visualisierung auswählen (z.B. heruntergeladene 6-Kanal WAV)", file_types=['audio'])
        show_visuals_button = gr.Button("📊 Visualisierungen generieren", variant="secondary")
        with gr.Row():
            input_image = gr.Image(label="🔵 Original Visualisierung", interactive=False)
            output_image = gr.Image(label="🟠 Verarbeitet Visualisierung", interactive=False)

        # Link button click to plotting function
        # Use internal Textbox components to pass titles to the plotting function
        show_visuals_button.click(
             fn=plot_waveform_and_spectrogram,
             # Pass the file component directly, the function handles extracting the path
             inputs=[input_file_vis, gr.Textbox("Original", visible=False)],
             outputs=[input_image]
        )
        show_visuals_button.click(
             fn=plot_waveform_and_spectrogram,
             # Pass the file component directly
             inputs=[output_file_vis, gr.Textbox("Verarbeitet", visible=False)],
             outputs=[output_image]
        )

    # === NEUER TAB: Audio-Profiler (Hinzugefügt) ===
    with gr.Tab("⚖️ Audio-Profiler"):
        gr.Markdown("## ⚖️ Audio-Profiler: Original vs. Bearbeitet")
        gr.Markdown("Vergleicht zwei Audiodateien (z.B. Original-Upload und das heruntergeladene Ergebnis) und analysiert Unterschiede in Lautheit, Stereo-Breite und LFE-Energie.")
        with gr.Row():
            profiler_input_original = gr.File(
                label=" Lade Original-Datei (Mono/Stereo)",
                file_types=['audio']
            )
            profiler_input_processed = gr.File(
                label=" Lade bearbeitete Datei (z.B. 6-Kanal WAV)",
                file_types=['audio', '.wav'] # Erlaube explizit WAV
            )
        profiler_analyze_button = gr.Button("🚀 Analyse starten!", variant="primary")
        profiler_report_output = gr.Markdown(label="📋 Analysebericht", value="*Bericht wird hier angezeigt...*")

        # Link profiler button to backend function
        profiler_analyze_button.click(
            fn=run_audio_profiler,
            inputs=[profiler_input_original, profiler_input_processed],
            outputs=[profiler_report_output]
        )
    # === ENDE Neuer Tab ===

    with gr.Tab("🛠 Preset-Editor"):
        gr.Markdown("## 🛠 Presets speichern, laden, aktualisieren und löschen")
        with gr.Row():
            with gr.Column(scale=1):
                preset_name_input = gr.Textbox(label="📝 Preset-Name", placeholder="z.B. Mein Raumsetup", scale=2)
                save_preset_button = gr.Button("💾 Aktuelle Einstellungen als Preset speichern", variant="primary")
                save_status = gr.Label(label="Status") # Use Label for dynamic status messages
            with gr.Column(scale=1):
                preset_list = gr.Dropdown(
                    label="📂 Gespeicherte Presets (.json)",
                    choices=[], # Populated by on_start/refresh
                    interactive=True,
                    allow_custom_value=False # Prevent user from typing non-existent filenames
                )
                with gr.Row():
                    load_preset_button = gr.Button("📥 Laden", scale=1)
                    delete_preset_button = gr.Button("🗑️ Löschen", variant="stop", scale=1)
                    refresh_presets_button = gr.Button("🔄 Liste neu laden", scale=1)


        with gr.Row():
            export_presets_button = gr.Button("📦 Alle Presets als ZIP exportieren")
            zip_download = gr.File(label="📦 ZIP-Datei herunterladen", interactive=False)

        # --- Preset Editor Functions (MODIFIED to include ALL relevant controls) ---
        ALL_PRESET_CONTROLS = [ # List of components whose values need to be saved/loaded
            ir_duration, reflection_count, max_delay, # Hidden originals
            hall_type, early_level, late_level,      # Hall/Level controls
            dry_wet, dry_wet_kill_start_slider,       # Mix controls
            bass_gain, treble_gain,                 # EQ
            material_choice, surround_x, surround_y  # Material & Position
        ]
        # The number of controls in this list MUST match the load/save functions
        NUM_PRESET_CONTROLS = len(ALL_PRESET_CONTROLS)

        def list_presets_for_dropdown():
            """ Fetches preset files and formats them for the Dropdown."""
            ensure_preset_dir()
            try:
                files = [f for f in os.listdir(PRESET_DIR) if f.endswith(".json")]
                files.sort(key=str.lower) # Sort alphabetically, case-insensitive
                # Return list of filenames (Gradio handles value=filename internally)
                return files
            except Exception as e:
                print(f"Error listing presets: {e}")
                return []

        # MODIFIED: Takes values from ALL_PRESET_CONTROLS as input arguments
        def save_current_preset(preset_name, *control_values):
            """Saves the current UI settings as a JSON preset file."""
            ensure_preset_dir()
            preset_name = preset_name.strip() if isinstance(preset_name, str) else ""
            if not preset_name:
                return "⚠️ Bitte gültigen Preset-Namen angeben!", gr.update()

            # Sanitize filename
            safe_filename_base = "".join(c for c in preset_name if c.isalnum() or c in ('_', '-', ' ')).strip()
            safe_filename = safe_filename_base.replace(' ', '_') + ".json"
            if not safe_filename_base or safe_filename == ".json":
                 return "⚠️ Ungültiger oder leerer Preset-Name nach Bereinigung.", gr.update()

            preset_path = os.path.join(PRESET_DIR, safe_filename)

            # Check if the number of received values matches expected
            if len(control_values) != NUM_PRESET_CONTROLS:
                 error_msg = f"❌ Interner Fehler: Falsche Anzahl von Werten ({len(control_values)} statt {NUM_PRESET_CONTROLS}) beim Speichern empfangen."
                 print(error_msg)
                 return error_msg, gr.update()

            preset_data = {
                # Map values to keys based on the order in ALL_PRESET_CONTROLS
                "ir_duration": control_values[0],
                "reflection_count": control_values[1],
                "max_delay": control_values[2],
                "hall_type": control_values[3],
                "early_level": control_values[4],
                "late_level": control_values[5],
                "dry_wet": control_values[6],
                "dry_wet_kill_start": control_values[7],
                "bass_gain": control_values[8],
                "treble_gain": control_values[9],
                "material": control_values[10],
                "x_pos": control_values[11],
                "y_pos": control_values[12],
                # Optionally store original user-given name if sanitization changed it
                "_source_name": preset_name if safe_filename_base != preset_name else None
            }
            try:
                with open(preset_path, "w", encoding='utf-8') as f:
                    json.dump(preset_data, f, indent=4, ensure_ascii=False)
                print(f"Preset '{safe_filename}' saved successfully.")
                save_last_preset(safe_filename) # Save this as the last used preset
                new_choices = list_presets_for_dropdown() # Refresh list after saving
                # Update the dropdown with the new list and select the saved preset
                return f"✅ Preset '{safe_filename}' gespeichert!", gr.update(choices=new_choices, value=safe_filename)
            except Exception as e:
                print(f"Error saving preset '{safe_filename}': {e}")
                traceback.print_exc()
                return f"❌ Fehler beim Speichern: {e}", gr.update() # Return update without changing choices

        # MODIFIED: Returns a tuple of NUM_PRESET_CONTROLS values using gr.update() for each control
        def load_selected_preset(preset_file):
            """Loads settings from the selected JSON file into the UI controls."""
            if not preset_file or not isinstance(preset_file, str):
                 print("Load Preset: No preset file selected or invalid type.")
                 # Return updates for all controls to do nothing
                 return [gr.update()] * NUM_PRESET_CONTROLS

            preset_path = os.path.join(PRESET_DIR, preset_file)
            if not os.path.exists(preset_path):
                print(f"Load Preset Error: File '{preset_file}' not found at '{preset_path}'.")
                gr.Warning(f"Preset-Datei '{preset_file}' nicht gefunden.")
                return [gr.update()] * NUM_PRESET_CONTROLS

            try:
                print(f"Loading preset from: {preset_path}")
                with open(preset_path, "r", encoding='utf-8') as f:
                    preset_data = json.load(f)
                save_last_preset(preset_file) # Mark this as the last used preset
                print(f"  Preset data loaded: {preset_data}")

                # Define defaults for all values in case they are missing in the JSON
                # Order must match ALL_PRESET_CONTROLS
                defaults_load = (
                    1.5, 35, 0.06,               # Hidden sliders defaults
                    DEFAULT_HALL_TYPE, 0.8, 0.6, # Hall/Level defaults
                    0.5, 0.5,                    # Mix/Kill Start defaults
                    1.0, 1.0,                    # EQ defaults
                    DEFAULT_MATERIAL, 0.5, 0.5   # Material, Position defaults
                )

                # Create a list of gr.update(value=...) for each control
                updates = []
                keys_in_order = [ # Match the order in preset_data mapping and ALL_PRESET_CONTROLS
                    "ir_duration", "reflection_count", "max_delay", "hall_type", "early_level",
                    "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain",
                    "material", "x_pos", "y_pos"
                ]

                for i, key in enumerate(keys_in_order):
                     # Get value from loaded data, fall back to default if key missing or value is None
                     value = preset_data.get(key, defaults_load[i])
                     if value is None: # Handle case where JSON has null value
                         value = defaults_load[i]
                     updates.append(gr.update(value=value))

                print(f"  Prepared {len(updates)} updates for UI controls.")
                return updates # Return list of Gradio update objects

            except json.JSONDecodeError as jde:
                 print(f"Load Preset Error: Failed to decode JSON from '{preset_file}': {jde}")
                 gr.Error(f"Fehler beim Lesen der Preset-Datei '{preset_file}'. Ist sie korrekt formatiert?")
                 return [gr.update()] * NUM_PRESET_CONTROLS
            except Exception as e:
                print(f"Load Preset Error: Unexpected error loading '{preset_file}': {e}")
                traceback.print_exc()
                gr.Error(f"Unbekannter Fehler beim Laden von '{preset_file}'.")
                return [gr.update()] * NUM_PRESET_CONTROLS

        # --- Delete and Export Functions (Unchanged logic, improved messages) ---
        def delete_selected_preset(preset_file):
            """Deletes the selected preset file from disk."""
            if not preset_file or not isinstance(preset_file, str):
                return "⚠️ Kein Preset zum Löschen ausgewählt!", gr.update()

            preset_path = os.path.join(PRESET_DIR, preset_file)
            status_message = ""
            new_selection = None # Deselect after deleting

            if os.path.exists(preset_path):
                try:
                    print(f"Deleting preset file: {preset_path}")
                    os.remove(preset_path)
                    status_message = f"🗑️ Preset '{preset_file}' gelöscht!"
                    print(f"  Preset '{preset_file}' deleted successfully.")
                    # If the deleted preset was the last used one, clear the last preset file
                    if load_last_preset() == preset_file:
                         print("  Clearing last preset reference as it was deleted.")
                         save_last_preset("")
                except OSError as e: # Catch specific OS errors like permission denied
                    status_message = f"❌ Fehler beim Löschen von '{preset_file}': {e}"
                    print(status_message)
                    traceback.print_exc()
                except Exception as e:
                    status_message = f"❌ Unerwarteter Fehler beim Löschen von '{preset_file}': {e}"
                    print(status_message)
                    traceback.print_exc()
            else:
                status_message = f"⚠️ Preset '{preset_file}' nicht gefunden (kann nicht gelöscht werden)."
                print(status_message)

            print("Refreshing preset list after deletion attempt.")
            new_choices = list_presets_for_dropdown()
            # Update status message and refresh dropdown list, clearing the selection
            return status_message, gr.update(choices=new_choices, value=new_selection)

        def export_presets_as_zip():
            """Creates a ZIP archive of all .json preset files."""
            ensure_preset_dir()
            # Use a more descriptive temporary filename if needed, though path is returned
            zip_path = tempfile.NamedTemporaryFile(delete=False, suffix="_presets.zip", prefix="audio_studio_").name
            preset_files_found = 0
            try:
                print("Exporting presets to ZIP...")
                json_files = [f for f in os.listdir(PRESET_DIR) if f.endswith(".json")]

                if not json_files:
                    print("  No preset files (.json) found to export.")
                    gr.Info("Keine Preset-Dateien (.json) im 'presets'-Ordner gefunden.")
                    if os.path.exists(zip_path): os.remove(zip_path) # Clean up empty zip stub
                    return None # Return None if no files were zipped

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for filename in json_files:
                        file_path = os.path.join(PRESET_DIR, filename)
                        # Add file to zip using its base name (relative path within zip)
                        zipf.write(file_path, arcname=filename)
                        preset_files_found += 1

                if preset_files_found > 0:
                    print(f"  Successfully added {preset_files_found} presets to {zip_path}")
                    # Return the path to the generated zip file for the download component
                    return zip_path
                else:
                    # This case should be covered by the initial check, but as safety
                    print("  No preset files found after iterating (unexpected).")
                    if os.path.exists(zip_path): os.remove(zip_path)
                    return None


            except Exception as e:
                 print(f"Error creating preset ZIP: {e}")
                 traceback.print_exc()
                 gr.Error(f"Fehler beim Erstellen der ZIP-Datei: {e}")
                 if os.path.exists(zip_path): os.remove(zip_path) # Clean up failed zip attempt
                 return None

        # --- Link Preset Editor Buttons (MODIFIED inputs/outputs list sizes) ---
        save_preset_button.click(
            fn=save_current_preset,
            # Pass preset name + all control values in the correct order
            inputs=[preset_name_input] + ALL_PRESET_CONTROLS,
            outputs=[save_status, preset_list] # Update status label and preset dropdown
        )

        # MODIFIED: Outputs list now uses ALL_PRESET_CONTROLS
        load_preset_button.click(
            fn=load_selected_preset,
            inputs=[preset_list],
            outputs=ALL_PRESET_CONTROLS # The function returns a list of updates for these controls
        ).then( # Chain the update of the marker image AFTER loading preset values
            # This 'then' block executes after the preset values are loaded into the sliders
            fn=handle_slider_change,
            # Read the *updated* slider values from the UI state
            inputs=[surround_x, surround_y],
            outputs=[surround_output_image] # Update the marker image display
        ).then(
            # Also update the hall info text after loading a preset
            fn=update_hall_info,
            inputs=[hall_type], # Read the newly loaded hall_type
            outputs=[hall_info_text]
        )


        refresh_presets_button.click(
            fn=list_presets_for_dropdown,
            inputs=[],
            outputs=[preset_list] # Only update the dropdown choices
        ).then(lambda: "Presetliste aktualisiert.", outputs=save_status) # Update status too


        delete_preset_button.click(
            fn=delete_selected_preset,
            inputs=[preset_list],
            outputs=[save_status, preset_list] # Update status and refresh dropdown
        )

        export_presets_button.click(
            fn=export_presets_as_zip,
            inputs=[],
            outputs=[zip_download] # Output path to the zip file component
        ).then(lambda x: "ZIP Export abgeschlossen." if x else "Keine Presets zum Exportieren gefunden.", inputs=[zip_download], outputs=save_status)


    with gr.Tab("ℹ️ Hilfe & Dokumentation"):
        # --- HILFETEXT AKTUALISIERT ---
        gr.Markdown("""
        ## 🎶 Audio Raytracing Studio v3.5 (Profiler) - Hilfe

        **Neu:** Der **Audio-Profiler**-Tab (⚖️) erlaubt den direkten Vergleich eines Original-Audios mit einer bearbeiteten Version.

        **Profiler Bedienung:**
        1. Lade die **Originaldatei** (Mono/Stereo) in das linke Feld.
        2. Lade die **bearbeitete Datei** (z.B. die heruntergeladene 6-Kanal WAV) in das rechte Feld.
        3. Klicke **"🚀 Analyse starten!"**.
        4. Der Bericht zeigt Änderungen bei:
            *   **Lautheit (LUFS):** Gemessen über die Front-Kanäle (FL/FR).
            *   **Stereo-Breite:** Vergleich der "Side"-Signal-Energie von FL/FR. Gibt prozentuale Änderung an, wenn beide Dateien Stereo sind.
            *   **LFE-Energie:** Pegel des LFE-Kanals (Kanal 4 / Index 3) in der bearbeiteten Datei (in RMS und dBFS). Enthält eine *grobe* Schätzung des Energieanteils über 50 Hz.

        ---

        **Ziel:** Simuliert den Klang eines Audiosignals in verschiedenen Räumen mit anpassbaren Materialien und ermöglicht die Positionierung im 5.1 Surround-Feld. Nutzt eine getrennte Verarbeitung für frühe Reflexionen (Early Reflections, ER) und späten Nachhall (Late Reverb, LR).

        **Kernfunktionen & Zusammenspiel:**

        *   **Hall-Typ:** Wählt eine Grundcharakteristik (`Plate`, `Room`, `Cathedral`). Dies bestimmt interne Parameter (Nachhallzeit, ER/LR-Verteilung, Basis-Diffusität).
        *   **Position (X/Y):** Beeinflusst das 5.1 Panning UND die **Gerichtetheit (Directionality)** des Halls.
            *   *Mitte (0.5, 0.5):* Eher gerichteter Hall (klarere frühe Reflexionen).
            *   *Rand:* Eher diffuser Hall (weichere, unschärfere Reflexionen).
        *   **Directionality:** Wird automatisch berechnet aus `Hall-Typ` und `Position`. Ein Wert nahe 1 bedeutet sehr gerichtet, nahe 0 sehr diffus. Beeinflusst die Balance und Stärke der Early Reflections in der IR-Generierung.
        *   **Adaptive Early/Late Balance:** Die Regler `Basis Early Level` und `Basis Late Level` stellen die *Grundlautstärke* ein. Das tatsächliche Verhältnis wird aber **dynamisch** durch den `Dry/Wet`-Regler angepasst:
            *   *Niedriger Dry/Wet:* Early Reflections werden betont (direkter Klang).
            *   *Hoher Dry/Wet:* Late Reverb wird betont (diffuserer, "weiter" Klang).
        *   **Dynamisches Dry Signal Muting:** Das **Originalsignal (Dry)** wird automatisch leiser, je höher der `Dry/Wet`-Regler steht, beginnend ab dem Wert des `Dry Kill Start`-Reglers. Dies verhindert oft eine Überlagerung von direktem Schall und sehr lautem Hall.

        **Bedienung:**

        1.  **Audioquelle wählen (Tab: Audio-Verarbeitung):**
            *   **Hochladen:** Nutze "Audio hochladen" für WAV/MP3 etc.
            *   **Mikrofon:** Nutze "Mikrofonaufnahme". Priorität: Upload > Mikrofon.

        2.  **Raum einstellen (Tab: Audio-Verarbeitung):**
            *   **Hall-Typ:** Wähle `Plate`, `Room` oder `Cathedral`. Die Beschreibung darunter erklärt den Charakter. Interne Parameter werden automatisch angepasst.
            *   **Material:** Wähle das Oberflächenmaterial (beeinflusst Dämpfung/Klangfarbe der Reflexionen).

        3.  **Mix & EQ einstellen (Tab: Audio-Verarbeitung):**
            *   **Basis Early/Late Level:** Stelle die *Grund*lautstärke für ER und LR ein. Die effektive Balance wird durch `Dry/Wet` beeinflusst!
            *   **Dry/Wet:** Regelt das Verhältnis Original zu Effekt. Beeinflusst auch ER/LR-Balance und Dry-Signal-Ausblendung.
            *   **Dry Kill Start:** Ab welchem `Dry/Wet`-Wert das Originalsignal leiser wird (Standard 0.5 = ab 50% Wet). Bei 1.0 ist das Ausblenden deaktiviert.
            *   **Bass/Treble Gain:** EQ für das *Gesamtsignal* (nach Dry/Wet-Mix).

        4.  **Position wählen (Tab: 5.1 Surround Map):**
            *   **Klicken/Slider:** Wähle die X/Y-Position. Beeinflusst Panning *und* die berechnete Hall-Gerichtetheit. Die Marker-Anzeige visualisiert die Wahl.

        5.  **Verarbeiten (Tab: Audio-Verarbeitung):**
            *   Klicke **"➡️ Verarbeiten & Anhören!"**.
            *   Das Ergebnis (6-Kanal WAV) erscheint im Player und kann heruntergeladen werden. *Dein System/Player muss 6-Kanal-Audio unterstützen.*

        6.  **Visualisieren (Tab: Visualizer):**
            *   Wähle Original- und verarbeitete Audiodatei aus.
            *   Klicke "Visualisierungen generieren" für Wellenform/Spektrogramm-Vergleich.

        7.  **Presets verwalten (Tab: Preset-Editor):**
            *   **Speichern:** Namen eingeben, speichern. Sichert alle aktuellen Einstellungen (Halltyp, Basis-Level, Mix, Dry Kill Start, EQ, Material, Position).
            *   **Laden:** Preset wählen, laden. Alle Regler und die Position werden gesetzt.
            *   **Löschen/Aktualisieren/Exportieren:** Weitere Verwaltungsoptionen.

        **Technische Hinweise:**
        *   Ausgabe: Standard WAV 5.1 (FL, FR, C, LFE, RL, RR).
        *   Reverb: Approximierte Synthese (Split ER/LR, Directionality, Adaptive Balance, Dynamic Dry Muting).
        *   **FFmpeg:** Für das Laden von Nicht-WAV-Dateien (MP3 etc.) wird Pydub verwendet, welches oft **FFmpeg** benötigt. Stelle sicher, dass `ffmpeg` und `ffprobe` installiert und im System-PATH sind, sonst schlägt das Laden fehl!
        *   **Profiler Libraries:** Benötigt `soundfile` und `pyloudnorm`. Installiere sie mit: `pip install soundfile pyloudnorm`
        """)
        # --- ENDE HILFETEXT AKTUALISIERT ---

    # --- Main Processing Trigger (MODIFIED inputs list) ---
    # Wrapper function to decide which audio source to use
    def process_audio_main(
        audio_upload_path, mic_record_path,
        # Pass all controls needed by apply_raytrace_convolution_split
        hall_type_val, base_early_level_val, base_late_level_val, # Hall/Level controls
        drywet, dw_kill_start,                                   # Mix controls
        bass, treble,                                            # EQ controls
        x, y,                                                    # Position controls
        material                                                 # Material control
        ):
        """Wrapper function to decide audio source and call the main SPLIT processing function."""
        source_file_to_process = None
        source_type = "None"

        # Check microphone input validity (exists and has some size)
        mic_valid = False
        if mic_record_path and isinstance(mic_record_path, str) and os.path.exists(mic_record_path):
            try:
                if os.path.getsize(mic_record_path) > 1024: # Basic check for non-empty file (e.g., > 1KB)
                    mic_valid = True
                else:
                     print("Microphone input file is too small or empty, ignoring.")
            except Exception as e:
                 print(f"Error checking microphone file size: {e}")
                 mic_valid = False

        # Check uploaded file validity
        upload_valid = False
        upload_path = getattr(audio_upload_path, 'name', audio_upload_path) # Handle Gradio file object
        if upload_path and isinstance(upload_path, str) and os.path.exists(upload_path):
             try:
                 if os.path.getsize(upload_path) > 100: # Basic size check
                     upload_valid = True
                 else:
                      print("Uploaded file is too small or empty, ignoring.")
             except Exception as e:
                  print(f"Error checking upload file size: {e}")
                  upload_valid = False

        # Prioritize uploaded file over microphone recording
        if upload_valid:
             source_file_to_process = upload_path
             source_type = "Upload"
             print(f"Processing -> Source: Uploaded File ('{os.path.basename(source_file_to_process)}')")
        elif mic_valid:
             source_file_to_process = mic_record_path
             source_type = "Microphone"
             print(f"Processing -> Source: Microphone Recording ('{os.path.basename(source_file_to_process)}')")
        else:
            print("Processing -> Error: No valid audio source selected (Upload or Microphone).")
            gr.Warning("Keine gültige Audioquelle ausgewählt (weder Upload noch Mikrofonaufnahme).")
            # Return None for both outputs (player and download link)
            return None, None

        # --- Call the main processing function with all required arguments ---
        processed_path_player, processed_path_download = apply_raytrace_convolution_split(
            audio_file_path=source_file_to_process,
            hall_type_val=hall_type_val,
            base_early_level=base_early_level_val,
            base_late_level=base_late_level_val,
            dry_wet=drywet,
            bass_gain=bass,
            treble_gain=treble,
            x_pos=x,
            y_pos=y,
            material=material,
            dry_wet_kill_start=dw_kill_start
        )

        # Return the paths for the audio player and download file components
        # If processing failed, these will be None, which Gradio handles
        return processed_path_player, processed_path_download

    # Link the main process button (Inputs match the arguments of process_audio_main wrapper)
    process_button.click(
        fn=process_audio_main,
        inputs=[
            audio_input, mic_input,             # Audio sources
            # Pass controls in the order expected by process_audio_main
            hall_type, early_level, late_level, # Hall and Level controls
            dry_wet, dry_wet_kill_start_slider, # Mix controls
            bass_gain, treble_gain,             # EQ controls
            surround_x, surround_y,             # Position controls
            material_choice                     # Material control
        ],
        outputs=[output_audio, download]        # Output components: player and download link
    )

    # --- Link Hall Type Change Event ---
    # Update description text when hall type changes
    hall_type.change(
        fn=update_hall_info,
        inputs=[hall_type],
        outputs=[hall_info_text]
    )


    # --- App Initialization Function (on_start) (MODIFIED to handle all preset controls) ---
    def on_start():
        """Initializes presets, loads last used settings, sets initial marker & hall info."""
        print("App starting, running on_start initialization...")
        ensure_preset_dir()
        available_presets_list = list_presets_for_dropdown()
        last_preset_filename = load_last_preset()

        # --- Default values for ALL preset controls (NUM_PRESET_CONTROLS total) ---
        # Order must match ALL_PRESET_CONTROLS
        defaults = (
            1.5, 35, 0.06,               # ir_duration, reflection_count, max_delay (Hidden)
            DEFAULT_HALL_TYPE, 0.8, 0.6, # hall_type, early_level, late_level
            0.5, 0.5,                    # dry_wet, dry_wet_kill_start_slider
            1.0, 1.0,                    # bass_gain, treble_gain
            DEFAULT_MATERIAL, 0.5, 0.5   # material_choice, surround_x, surround_y
        )
        loaded_values = list(defaults) # Use a list to potentially modify values
        preset_value_to_select = None # Which preset to show as selected in dropdown

        # --- Try to load the last used preset ---
        if last_preset_filename:
            print(f"Found last preset file reference: {last_preset_filename}")
            try:
                # Use load_selected_preset which now returns a list of update objects
                # We need the raw values here, so we'll load the file directly again (or adapt load_selected_preset)
                # Let's load directly here for simplicity in on_start:
                preset_path = os.path.join(PRESET_DIR, last_preset_filename)
                if os.path.exists(preset_path):
                    with open(preset_path, "r", encoding='utf-8') as f:
                        preset_data = json.load(f)

                    # Map loaded data back to the 'loaded_values' list, using defaults as fallback
                    keys_in_order = [
                        "ir_duration", "reflection_count", "max_delay", "hall_type", "early_level",
                        "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain",
                        "material", "x_pos", "y_pos"
                    ]
                    valid_load = True
                    for i, key in enumerate(keys_in_order):
                        value = preset_data.get(key, defaults[i])
                        if value is None: # Handle null values in JSON gracefully
                             value = defaults[i]
                        # Basic type check/conversion might be good here if needed
                        try:
                             # Example: Ensure numeric types are float/int
                             if key in ["ir_duration", "max_delay", "early_level", "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain", "x_pos", "y_pos"]:
                                 loaded_values[i] = float(value)
                             elif key == "reflection_count":
                                 loaded_values[i] = int(value)
                             else: # Strings like hall_type, material
                                 loaded_values[i] = str(value)
                        except (ValueError, TypeError):
                             print(f"  Warning: Invalid type for '{key}' in preset '{last_preset_filename}'. Using default.")
                             loaded_values[i] = defaults[i]
                             valid_load = False # Mark as potentially problematic load

                    if valid_load:
                         preset_value_to_select = last_preset_filename
                         print(f"Successfully loaded settings from {last_preset_filename}")
                    else:
                         print(f"Loaded settings from {last_preset_filename} with some warnings/defaults.")
                         preset_value_to_select = last_preset_filename # Still select it, but values might be default

                else:
                    print(f"Last preset file '{last_preset_filename}' not found. Using defaults.")
                    save_last_preset("") # Clear invalid last preset reference

            except Exception as e:
                print(f"Error processing last preset '{last_preset_filename}' during startup: {e}. Using defaults.")
                traceback.print_exc()
                loaded_values = list(defaults) # Reset to defaults on error
                save_last_preset("") # Clear invalid last preset reference
        else:
             print("No last preset file found or specified. Using default settings.")

        # --- Generate the initial marker image based on loaded/default position ---
        initial_marker_path = None
        # Use the global path BASE_SURROUND_MAP_PATH which should be set when the UI is defined
        try:
            # loaded_values[11] is x_pos, loaded_values[12] is y_pos
            initial_marker_path = update_marker_image(loaded_values[11], loaded_values[12], BASE_SURROUND_MAP_PATH)
            print(f"on_start: Initial marker path generated: {initial_marker_path}")
            if initial_marker_path is None:
                 print("Warning: update_marker_image returned None during on_start. Marker image might not display.")
        except Exception as e:
            print(f"Error calling update_marker_image during on_start: {e}")
            traceback.print_exc()
            initial_marker_path = None # Ensure it's None on error

        # --- Update hall info text based on loaded/default hall type ---
        # loaded_values[3] is hall_type
        initial_hall_info = update_hall_info(loaded_values[3])

        # --- Return updates for ALL components controlled by presets/load ---
        # Order MUST match the 'outputs' list in demo.load() below
        # We return a list containing:
        # 1. Update for the preset dropdown (choices, selected value)
        # 2. Values for all the preset controls
        # 3. Update for the marker image
        # 4. Update for the hall info text

        output_updates = [
            # 1. Preset list update
            gr.update(choices=available_presets_list, value=preset_value_to_select), # preset_list

            # 2. Values for the NUM_PRESET_CONTROLS (unpack loaded_values)
            *loaded_values, # Unpack the list of loaded/default values

            # 3. Update the surround map marker image display
            gr.update(value=initial_marker_path), # surround_output_image

            # 4. Update the hall info text display
            initial_hall_info # hall_info_text
        ]

        print(f"on_start: Prepared {len(output_updates)} updates for demo.load.")
        return output_updates

    # --- Link the on_start function to the demo.load event ---
    # The outputs list must match the components updated by on_start's return list
    demo.load(
        fn=on_start,
        inputs=[],
        outputs=[
            # Order must match the return list in on_start
            preset_list,            # 1. Preset Dropdown update object
            # 2. The NUM_PRESET_CONTROLS components (in order of ALL_PRESET_CONTROLS)
            *ALL_PRESET_CONTROLS,
            # 3. The marker image component
            surround_output_image,
            # 4. The hall info text component
            hall_info_text
        ]
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("-------------------------------------------------------------")
    print("Starting Audio Raytracing Studio Gradio App (v3.5 Profiler)...")
    print(f"Preset Directory: {os.path.abspath(PRESET_DIR)}")
    print("Ensure FFmpeg is installed and accessible in PATH for MP3/other format support.")
    print("Ensure Profiler dependencies are installed: pip install soundfile pyloudnorm")
    print("Surround output is 6-channel WAV (FL, FR, C, LFE, RL, RR).")
    print("-------------------------------------------------------------")
    # Ensure preset directory exists before launching
    ensure_preset_dir()
    # Ensure the base surround map exists or is created before launch
    # The check inside the UI build does this, but double-checking won't hurt
    if not os.path.exists(BASE_SURROUND_MAP_PATH) or not os.path.isfile(BASE_SURROUND_MAP_PATH):
         print(f"Warning: Base surround map '{BASE_SURROUND_MAP_PATH}' not found at launch. Placeholder attempt will happen during UI build.")

    # Launch the app
    # share=True allows access via public URL (requires internet connection)
    # debug=True provides more detailed logs in console
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=False)

