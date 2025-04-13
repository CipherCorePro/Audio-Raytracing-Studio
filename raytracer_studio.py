# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import gradio as gr
# from scipy.io import wavfile # Keep commented unless truly needed
from scipy.signal import fftconvolve, spectrogram, resample, lfilter # removed firwin if unused
# from pydub import AudioSegment # Keep commented
import tempfile
import matplotlib
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import shutil # Wichtig für das Kopieren von Dateien benötigt
import zipfile
import traceback
import soundfile as sf # Bibliothek zum Lesen/Schreiben von Audiodateien
import pyloudnorm as pyln # Bibliothek zur Lautheitsmessung (LUFS)
import math

# --- Globale Konstanten ---
APP_VERSION = "v4.1 ('Astonishing Attempt Edition')"
PRESET_DIR = "presets_v4" # Verzeichnis zum Speichern von Presets
LAST_PRESET_FILE = os.path.join(PRESET_DIR, "last_preset_v4.txt") # Datei zum Speichern des zuletzt verwendeten Presets
BASE_SURROUND_MAP_PATH = "surround_layout_3d.png" # Dateiname für das Hintergrundbild der Surround-Karte

# --- Materialeigenschaften (Absorptionskoeffizienten) ---
material_absorption = {
    "Stein": 0.15, "Holz": 0.35, "Teppich": 0.7, "Glas": 0.2,
    "Beton": 0.1, "Vorhang (schwer)": 0.8
}
DEFAULT_MATERIAL = "Holz" # Standardmaterial
DEFAULT_HALL_TYPE = "Room" # Standard-Halltyp

# --- Definitionen der Kanal-Layouts ---
CHANNEL_LAYOUTS = {
    "Stereo": {"channels": 2, "names": ["FL", "FR"]},
    "5.1 (Standard)": {"channels": 6, "names": ["FL", "FR", "C", "LFE", "RL", "RR"]},
    "7.1 (Surround)": {"channels": 8, "names": ["FL", "FR", "C", "LFE", "RL", "RR", "SL", "SR"]},
    "5.1.2 (Atmos Light)": {"channels": 8, "names": ["FL", "FR", "C", "LFE", "RL", "RR", "TFL", "TFR"]},
}
DEFAULT_CHANNEL_LAYOUT = "5.1 (Standard)" # Standard-Ausgabeformat

# --- Hilfsfunktionen für Presets ---

def ensure_preset_dir():
    """Stellt sicher, dass das Verzeichnis zum Speichern von Presets existiert."""
    os.makedirs(PRESET_DIR, exist_ok=True)

def save_last_preset(preset_name):
    """Speichert den Dateinamen des zuletzt verwendeten Presets."""
    ensure_preset_dir()
    try:
        # Speichert den Namen oder einen leeren String, wenn None übergeben wird
        name_to_save = preset_name if isinstance(preset_name, str) else ""
        with open(LAST_PRESET_FILE, "w", encoding='utf-8') as f:
            f.write(name_to_save)
    except Exception as e:
        print(f"Fehler beim Speichern des letzten Presets '{preset_name}': {e}")

def load_last_preset():
    """Lädt den Dateinamen des zuletzt verwendeten Presets."""
    ensure_preset_dir()
    try:
        if not os.path.exists(LAST_PRESET_FILE): return None # Keine Datei vorhanden
        with open(LAST_PRESET_FILE, "r", encoding='utf-8') as f: last_preset = f.read().strip()
        if not last_preset: return None # Datei ist leer
        preset_path = os.path.join(PRESET_DIR, last_preset)
        # Prüft, ob die Datei existiert und eine JSON-Datei ist
        if os.path.exists(preset_path) and preset_path.endswith(".json"):
            return last_preset
        else:
            # Wenn die Datei ungültig ist, Referenz löschen
            print(f"Warnung: Letzte Preset-Datei '{last_preset}' nicht gefunden oder ungültig. Referenz wird gelöscht.")
            save_last_preset("")
            return None
    except Exception as e:
        print(f"Fehler beim Laden der letzten Preset-Datei: {e}")
        return None

# --- Kernfunktionen für Audioverarbeitung und Hall ---

def dynamic_dry_wet_mix(dry_signal, wet_signal, dry_wet, kill_start=0.5):
    """
    Mischt das trockene (dry) und das bearbeitete (wet) Signal dynamisch.
    Das trockene Signal wird ausgeblendet, wenn der Wet-Anteil den 'kill_start'-Wert übersteigt.
    """
    try:
        # Sicherstellen, dass Inputs Numpy-Arrays vom Typ float32 sind
        dry_signal = np.asarray(dry_signal, dtype=np.float32)
        wet_signal = np.asarray(wet_signal, dtype=np.float32)
        dry_wet = np.clip(float(dry_wet), 0.0, 1.0)
        kill_start = np.clip(float(kill_start), 0.0, 1.0)

        # Berechnet den Faktor, mit dem das Dry-Signal multipliziert wird
        dry_mix_factor = 1.0
        if kill_start < 1.0 and dry_wet >= kill_start:
            fade_range = 1.0 - kill_start
            if fade_range < 1e-6: # Vermeide Division durch Null
                dry_mix_factor = 0.0
            else:
                # Linearer Fade-Out des Dry-Signals im Bereich [kill_start, 1.0]
                progress_in_fade = (dry_wet - kill_start) / fade_range
                dry_mix_factor = np.clip(1.0 - progress_in_fade, 0.0, 1.0)

        # Passt die Längen der Signale an (kürzeste Länge bestimmt Überlappung)
        min_len = min(dry_signal.shape[0], wet_signal.shape[0])
        dry_matched = dry_signal[:min_len]
        wet_matched = wet_signal[:min_len]

        # Führt die Mischung durch
        mixed = (dry_mix_factor * (1.0 - dry_wet) * dry_matched) + (dry_wet * wet_matched)

        # Hängt den Rest des längeren Signals an (entsprechend skaliert)
        if dry_signal.shape[0] > min_len:
            mixed = np.concatenate((mixed, dry_signal[min_len:] * dry_mix_factor * (1.0 - dry_wet)), axis=0)
        elif wet_signal.shape[0] > min_len:
            mixed = np.concatenate((mixed, wet_signal[min_len:] * dry_wet), axis=0)

        return mixed.astype(np.float32)

    except Exception as e:
        # Fallback-Mechanismus bei Fehlern
        print(f"Fehler in dynamic_dry_wet_mix: {e}"); traceback.print_exc()
        try:
            dry_valid = dry_signal is not None and dry_signal.size > 0
            wet_valid = wet_signal is not None and wet_signal.size > 0

            if dry_valid and wet_valid:
                min_len_fallback = min(dry_signal.shape[0], wet_signal.shape[0])
                mixed_fallback = (((1.0 - dry_wet) * dry_signal[:min_len_fallback]) + (dry_wet * wet_signal[:min_len_fallback])).astype(np.float32)
                if dry_signal.shape[0] > min_len_fallback: mixed_fallback = np.concatenate((mixed_fallback, dry_signal[min_len_fallback:] * (1.0 - dry_wet)), axis=0)
                elif wet_signal.shape[0] > min_len_fallback: mixed_fallback = np.concatenate((mixed_fallback, wet_signal[min_len_fallback:] * dry_wet), axis=0)
                return mixed_fallback
            elif wet_valid: return (wet_signal * dry_wet).astype(np.float32)
            elif dry_valid: return (dry_signal * (1.0 - dry_wet)).astype(np.float32)
            else: return np.array([], dtype=np.float32)

        except Exception as fallback_e:
            print(f"Fehler während Fallback-Mixing: {fallback_e}")
            if wet_signal is not None and wet_signal.size > 0: return (wet_signal * dry_wet).astype(np.float32)
            elif dry_signal is not None and dry_signal.size > 0: return (dry_signal * (1.0-dry_wet)).astype(np.float32)
            else: return np.array([], dtype=np.float32)


def update_hall_info(selected_hall_type: str) -> str:
    """Generiert einen Beschreibungstext für den ausgewählten Hall-Typ."""
    hall_info = {
        "Plate": "Klassischer Studioplate-Hall. Dicht, hell, relativ kurze Nachhallzeit, stark gerichtet (wenig diffus). Gut für Vocals, Snares.",
        "Room": "Natürlicher Raumklang. Ausgewogene frühe Reflexionen und Nachhall, mittlere Gerichtetheit. Universell einsetzbar für Realismus.",
        "Cathedral": "Große Kathedrale. Sehr langer, diffuser Nachhall, späte Reflexionen dominant, geringe Gerichtetheit. Für Ambient, orchestrale Sounds."
    }
    # Gibt die Beschreibung oder einen Standardtext zurück
    return f"ℹ️ **Beschreibung:** {hall_info.get(selected_hall_type, hall_info.get(DEFAULT_HALL_TYPE, 'Keine Beschreibung verfügbar.'))}"

def adjust_reverb_parameters_by_hall(hall_type: str) -> tuple[float, int, float, float]:
    """Liefert grundlegende Hallparameter-Presets basierend auf dem Hall-Typ."""
    # Gibt Tupel zurück: (Dauer, Anzahl Reflexionen, max. Delay frühe Refl., Split-Zeitpunkt)
    if hall_type == "Plate": return 0.8, 25, 0.025, 0.03
    elif hall_type == "Room": return 1.5, 35, 0.06, 0.08
    elif hall_type == "Cathedral": return 4.0, 20, 0.10, 0.12
    else:
        # Fallback auf Standardwerte, wenn Typ unbekannt ist
        print(f"Warnung: Unbekannter Hall-Typ '{hall_type}', verwende Standardwerte für 'Room'.")
        return 1.5, 35, 0.06, 0.08

def adapt_early_late_levels(dry_wet: float, base_early: float = 0.8, base_late: float = 0.6) -> tuple[float, float]:
    """Passt die Pegel für frühe Reflexionen und späten Hall dynamisch an den Dry/Wet-Mix an."""
    try:
        dry_wet = np.clip(float(dry_wet), 0.0, 1.0)
        base_early = float(base_early)
        base_late = float(base_late)
        # Skaliert Early leiser und Late lauter bei höherem Wet-Anteil
        early_scale = 1.0 - (dry_wet**1.5 * 0.7)
        late_scale = 1.0 + (dry_wet**1.5 * 0.6)
        adapted_early = np.clip(base_early * early_scale, 0.0, 2.0)
        adapted_late = np.clip(base_late * late_scale, 0.0, 2.0)
        return adapted_early, adapted_late
    except Exception as e:
        print(f"Fehler in adapt_early_late_levels: {e}")
        return base_early, base_late # Fallback auf Basiswerte

def compute_final_directionality_3d(x_pos: float, y_pos: float, z_pos: float, hall_type: str, diffusion_grade: float, dry_wet: float = 0.5) -> float:
    """Berechnet die Direktionalität des Halls basierend auf 3D-Position, Halltyp, Diffusion und Dry/Wet."""
    try:
        # Normalisiert und beschränkt Input-Werte
        x = np.clip(float(x_pos), 0.0, 1.0); y = np.clip(float(y_pos), 0.0, 1.0); z = np.clip(float(z_pos), 0.0, 1.0)
        diffusion = np.clip(float(diffusion_grade), 0.0, 1.0); dw = np.clip(float(dry_wet), 0.0, 1.0)

        # Berechnet Distanzfaktoren basierend auf der Position
        distance_from_center_xz = np.sqrt(((x - 0.5)*2)**2 + ((z - 0.5)*1.0)**2) / np.sqrt(1**2 + 0.5**2)
        distance_from_front_back = abs(y - 0.5) * 2
        position_factor = np.clip((1.0 - distance_from_center_xz * 0.3) * (1.0 - distance_from_front_back * 0.2), 0.5, 1.0)

        # Basis-Direktionalität basierend auf Halltyp
        hall_base = {"Plate": 0.95, "Room": 0.65, "Cathedral": 0.25}.get(hall_type, 0.65)
        # Diffusion reduziert die Direktheit
        diffusion_factor = 1.0 - (diffusion * 0.8)
        # Kombinierte Basis-Direktionalität
        directionality_base = hall_base * position_factor * diffusion_factor
        # Leichter Boost der Direktheit bei hohem Wet-Anteil
        boost = max(0.0, (dw - 0.6) * 0.4)
        # Endgültige Direktheit (beschränkt auf sinnvollen Bereich)
        final_directionality = np.clip(directionality_base + boost, 0.05, 0.95)
        return final_directionality
    except Exception as e:
        print(f"Fehler in compute_final_directionality_3d: {e}"); traceback.print_exc()
        return 0.5 # Sicherer Standardwert

def adjust_parameters_for_3d(hall_type: str, room_size: float, z_pos: float) -> tuple[float, int, float, float]:
    """Passt Basis-Hallparameter an Raumgröße und Z-Position an."""
    try:
        room_size = float(room_size); z_pos = float(z_pos)
        base_dur, base_ref, base_max_delay, base_split = adjust_reverb_parameters_by_hall(hall_type)

        # Skalierungsfaktoren basierend auf Raumgröße (nicht-linear)
        size_factor_dur = np.clip( (room_size / 100.0)**0.33, 0.5, 2.5)
        size_factor_delay = np.clip( (room_size / 100.0)**0.25, 0.7, 1.8)
        size_factor_ref = np.clip( 1 + (room_size - 100)/500.0, 0.8, 1.5)

        # Angepasste Dauer, Reflexionsanzahl
        adj_duration = np.clip(base_dur * size_factor_dur, 0.1, 10.0)
        adj_ref_count = np.clip(int(base_ref * size_factor_ref), 5, 80)

        # Z-Position beeinflusst max. Delay leicht
        z_delay_factor = 1.0 + ((z_pos - 0.5) * 0.1) # +/- 5% Anpassung

        # Angepasstes max. Delay und Split-Zeitpunkt
        adj_max_delay = np.clip(base_max_delay * size_factor_delay * z_delay_factor, 0.01, 0.3)
        adj_split_time = np.clip(base_split * size_factor_delay, 0.02, 0.2)

        return adj_duration, adj_ref_count, adj_max_delay, adj_split_time
    except Exception as e:
        print(f"Fehler in adjust_parameters_for_3d: {e}"); traceback.print_exc()
        return adjust_reverb_parameters_by_hall(DEFAULT_HALL_TYPE) # Fallback

def generate_impulse_response_split_3d(rate: int, ir_duration: float, reflection_count: int, max_delay: float, material: str, directionality: float, early_late_split: float, diffusion_grade: float) -> tuple[np.ndarray, np.ndarray]:
    """Generiert getrennte Impulsantworten für frühe Reflexionen und späten Hall unter Berücksichtigung von 3D-Parametern."""
    try:
        # Parameter validieren und konvertieren
        rate = int(rate); ir_duration = float(ir_duration); reflection_count = int(reflection_count)
        max_delay = float(max_delay); directionality = float(directionality)
        early_late_split_time = float(early_late_split); diffusion = float(diffusion_grade)

        if rate <= 0 or ir_duration <= 0: return np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32)

        # Berechnet Länge der Impulsantwort und initialisiert Arrays
        length = max(1, int(ir_duration * rate));
        early_ir = np.zeros(length, dtype=np.float32)
        late_ir = np.zeros(length, dtype=np.float32)

        absorption = material_absorption.get(material, material_absorption.get(DEFAULT_MATERIAL, 0.35))
        split_point_samples = max(1, min(int(early_late_split_time * rate), length - 1))
        max_delay_samples = max(2, int(max_delay * rate))

        # --- Generierung der frühen Reflexionen ---
        if reflection_count > 0 and split_point_samples > 1:
            actual_max_early_delay = min(max_delay_samples, split_point_samples)
            if actual_max_early_delay > 1:
                for _ in range(reflection_count):
                    delay_samples = np.random.randint(1, max(2, actual_max_early_delay))
                    if 0 < delay_samples < split_point_samples:
                        base_strength = np.random.uniform(0.3, 0.8)
                        strength = base_strength * (1.0 - absorption)
                        strength *= np.clip(directionality, 0.1, 1.0)
                        strength *= (1.0 - (delay_samples / actual_max_early_delay)**0.7)
                        early_ir[delay_samples] += strength

        # --- Generierung des späten Halls (Reverb Tail) ---
        start_late_index = split_point_samples
        late_part_length = length - start_late_index
        if late_part_length > 0:
            target_amplitude_ratio = 10**(-50 / 20) # Zielamplitude am Ende (-50dB)
            if late_part_length > 1: decay_factor = np.power(target_amplitude_ratio, 1.0 / late_part_length)
            else: decay_factor = 0.1
            decay_factor = np.clip(decay_factor * (1.0 - absorption * 0.1), 0.8, 0.99999)

            initial_late_amp = 0.6 * (1.0 - np.clip(directionality, 0.0, 0.9))
            initial_late_amp *= np.clip(1.0 / (1 + ir_duration*0.5), 0.3, 1.0)
            initial_late_amp *= (1.0 - absorption**0.5)

            # Erzeugt geglättetes Rauschen basierend auf Diffusion
            noise_smooth_factor = int(np.clip(rate * 0.001 * (1.0 + diffusion * 2.0), 1, 10))
            late_noise_raw = np.random.uniform(-1, 1, size=late_part_length)
            if noise_smooth_factor > 1 and late_part_length >= noise_smooth_factor :
                 smooth_kernel = np.ones(noise_smooth_factor) / noise_smooth_factor
                 late_noise_smoothed = np.convolve(late_noise_raw, smooth_kernel, mode='same')
                 std_raw = np.std(late_noise_raw); std_smooth = np.std(late_noise_smoothed)
                 if std_smooth > 1e-6: late_noise_smoothed = late_noise_smoothed / std_smooth * std_raw # Skaliert Varianz zurück
                 else: late_noise_smoothed = late_noise_raw # Fallback bei Stille
            else: late_noise_smoothed = late_noise_raw

            initial_late_amp *= (1.0 + diffusion * 0.2) # Diffusion erhöht Startamplitude leicht
            decay_envelope = np.power(decay_factor, np.arange(late_part_length))
            late_ir[start_late_index:] = late_noise_smoothed * initial_late_amp * decay_envelope

        # --- Normalisierung der Teile (optional, aber sinnvoll) ---
        if length > 1:
            early_max = np.max(np.abs(early_ir[1:])) # Ignoriert Impuls bei 0
            if early_max > 1e-6: early_ir[1:] = (early_ir[1:] / early_max) * 0.9
        late_max = np.max(np.abs(late_ir))
        if late_max > 1e-6: late_ir = (late_ir / late_max) * 0.7

        return early_ir, late_ir
    except Exception as e:
        print(f"Fehler in generate_impulse_response_split_3d: {e}"); traceback.print_exc()
        return np.array([1.0], dtype=np.float32), np.zeros(1, dtype=np.float32)

def apply_simple_lp_filter(signal: np.ndarray, rate: int, air_absorption_factor: float) -> np.ndarray:
    """Wendet einen einfachen Tiefpassfilter an, um Luftabsorption zu simulieren."""
    if air_absorption_factor < 0.01: return signal # Filter nicht nötig
    if not isinstance(signal, np.ndarray) or signal.ndim != 2 or signal.size == 0: return signal

    try:
        n_fft = signal.shape[0]
        if n_fft < 2: return signal
        fft_data = np.fft.rfft(signal, axis=0)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / rate)
        absorption_start_freq = 2000
        damping_range_mask = (freqs >= absorption_start_freq)
        gain_factor = np.ones_like(freqs)
        max_freq = freqs[-1] if len(freqs) > 0 else absorption_start_freq + 1

        if np.any(damping_range_mask) and max_freq > absorption_start_freq:
             max_damping = np.clip(air_absorption_factor, 0.0, 1.0) * 0.8
             freq_ramp = (freqs[damping_range_mask] - absorption_start_freq) / (max_freq - absorption_start_freq)
             freq_ramp = np.clip(freq_ramp, 0, 1)
             gain_factor[damping_range_mask] = 1.0 - (freq_ramp * max_damping)

        fft_data *= gain_factor[:, np.newaxis] # Wendet Gain auf alle Kanäle an
        filtered_signal = np.fft.irfft(fft_data, n=n_fft, axis=0).astype(np.float32)
        return filtered_signal
    except Exception as e:
        print(f"Fehler beim Anwenden des Luftabsorptionsfilters: {e}"); traceback.print_exc()
        return signal

def convolve_audio_split_3d(data: np.ndarray, early_ir: np.ndarray, late_ir: np.ndarray, early_level: float, late_level: float, dry_wet: float, bass_gain: float = 1.0, treble_gain: float = 1.0, rate: int = 44100, kill_start_dw: float = 0.5, air_absorption_factor: float = 0.0) -> np.ndarray:
    """Faltet Audiodaten mit getrennten Early/Late IRs, wendet EQ, Mix und Luftabsorption an."""
    if data is None or data.size == 0: return np.zeros((0, 2), dtype=np.float32)

    # Stellt Stereo-Eingang sicher
    if data.ndim == 1: data = np.stack((data, data), axis=1)
    elif data.shape[1] == 1: data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2: data = data[:, :2]
    data = data.astype(np.float32)

    early_ir = np.asarray(early_ir, dtype=np.float32).flatten() if early_ir is not None else np.zeros(1)
    late_ir = np.asarray(late_ir, dtype=np.float32).flatten() if late_ir is not None else np.zeros(1)

    # Berechnet Ausgabelänge
    len_data = data.shape[0]; len_early = len(early_ir); len_late = len(late_ir)
    len_out_early = len_data + len_early - 1 if len_early > 0 else len_data
    len_out_late = len_data + len_late - 1 if len_late > 0 else len_data
    len_out_max = max(len_data, len_out_early, len_out_late)
    data_padded = np.pad(data, ((0, len_out_max - len_data), (0, 0)), mode='constant') if len_out_max > len_data else data

    # Faltung für Early Reflections
    early_wet = np.zeros((len_out_max, data_padded.shape[1]), dtype=data_padded.dtype)
    if early_ir.size > 1 and np.any(early_ir) and early_level > 1e-6:
        try:
            early_left = fftconvolve(data[:, 0], early_ir, mode='full')
            early_right = fftconvolve(data[:, 1], early_ir, mode='full')
            early_wet = np.stack((early_left[:len_out_max], early_right[:len_out_max]), axis=1)
        except Exception as e: print(f"Fehler (conv_split3d_early): {e}"); traceback.print_exc(); early_wet = np.zeros_like(data_padded)

    # Faltung für Late Reverb
    late_wet_conv = np.zeros((len_out_max, data_padded.shape[1]), dtype=data_padded.dtype)
    if late_ir.size > 1 and np.any(late_ir) and late_level > 1e-6:
        try:
            late_left = fftconvolve(data[:, 0], late_ir, mode='full')
            late_right = fftconvolve(data[:, 1], late_ir, mode='full')
            late_wet_conv = np.stack((late_left[:len_out_max], late_right[:len_out_max]), axis=1)
        except Exception as e: print(f"Fehler (conv_split3d_late): {e}"); traceback.print_exc(); late_wet_conv = np.zeros_like(data_padded)

    # Luftabsorption auf Late Reverb anwenden
    late_wet = late_wet_conv
    if air_absorption_factor > 0.01 and late_wet_conv.size > 0:
        try: late_wet = apply_simple_lp_filter(late_wet_conv, rate, air_absorption_factor)
        except Exception as e: print(f"Fehler bei Luftabsorption: {e}"); traceback.print_exc()

    # Kombiniert Wet-Signale und mischt mit Dry
    wet_combined = (early_wet * early_level) + (late_wet * late_level)
    mixed = dynamic_dry_wet_mix(data_padded, wet_combined, dry_wet, kill_start_dw)

    # EQ anwenden
    mixed_eq = mixed
    try:
        if mixed is not None and mixed.size > 0 and (not np.isclose(bass_gain, 1.0) or not np.isclose(treble_gain, 1.0)):
             n_fft = mixed.shape[0]
             if n_fft >= 2:
                 fft_data = np.fft.rfft(mixed, axis=0)
                 freqs = np.fft.rfftfreq(n_fft, d=1.0 / rate); bass_cutoff = 250; treble_cutoff = 4000
                 bass_mask = (freqs > 1e-6) & (freqs <= bass_cutoff); treble_mask = freqs >= treble_cutoff
                 fft_data[bass_mask] *= np.clip(bass_gain, 0.1, 5.0)
                 fft_data[treble_mask] *= np.clip(treble_gain, 0.1, 5.0)
                 mixed_eq = np.fft.irfft(fft_data, n=n_fft, axis=0).astype(np.float32)
    except Exception as e: print(f"Fehler (EQ_split3d): {e}"); traceback.print_exc()

    # Endgültige Normalisierung (nur bei Bedarf)
    if mixed_eq is None or mixed_eq.size == 0: return np.zeros((0, 2), dtype=np.float32)
    max_val = np.max(np.abs(mixed_eq)); mixed_norm = mixed_eq
    if max_val > 1.0: print(f"  Normalisiere Faltungsergebnis (max val war {max_val:.3f})"); mixed_norm = mixed_eq / max_val
    elif np.any(mixed_eq) and max_val < 1e-9: mixed_norm = np.zeros_like(mixed_eq)

    if mixed_norm.ndim == 1: mixed_norm = np.stack((mixed_norm, mixed_norm), axis=1)
    elif mixed_norm.shape[1] != 2: mixed_norm = mixed_norm[:, :2]
    return mixed_norm.astype(np.float32)

def convolve_audio_external_ir(data: np.ndarray, external_ir_data: np.ndarray, dry_wet: float, bass_gain: float = 1.0, treble_gain: float = 1.0, rate: int = 44100, kill_start_dw: float = 0.5) -> np.ndarray:
    """Faltet Stereo-Audiodaten mit einer externen Stereo-IR, wendet EQ und Mix an."""
    if data is None or data.size == 0: return np.zeros((0, 2), dtype=np.float32)
    if external_ir_data is None or not isinstance(external_ir_data, np.ndarray) or external_ir_data.ndim != 2 or external_ir_data.shape[1] != 2:
        print("Fehler: Externe IR ungültig oder nicht Stereo. Faltung übersprungen."); return data.astype(np.float32)

    # Stellt Stereo-Eingang sicher
    if data.ndim == 1: data = np.stack((data, data), axis=1)
    elif data.shape[1] == 1: data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2: data = data[:, :2]
    data = data.astype(np.float32); external_ir_data = external_ir_data.astype(np.float32)

    # Berechnet Ausgabelänge
    len_data = data.shape[0]; len_ir = external_ir_data.shape[0]
    len_out_max = len_data + len_ir - 1 if len_ir > 0 else len_data
    data_padded = np.pad(data, ((0, len_out_max - len_data), (0, 0)), mode='constant') if len_out_max > len_data else data

    # Echte Stereo-Faltung (L mit L-IR, R mit R-IR)
    wet_signal = np.zeros((len_out_max, data_padded.shape[1]), dtype=data_padded.dtype)
    try:
        wet_left = fftconvolve(data[:, 0], external_ir_data[:, 0], mode='full')
        wet_right = fftconvolve(data[:, 1], external_ir_data[:, 1], mode='full')
        wet_signal = np.stack((wet_left[:len_out_max], wet_right[:len_out_max]), axis=1)
    except Exception as e:
        print(f"Fehler während externer IR-Faltung: {e}"); traceback.print_exc()
        return (data_padded * (1.0 - dry_wet)).astype(np.float32) # Fallback: skaliertes Dry-Signal

    # Mischt Dry/Wet
    mixed = dynamic_dry_wet_mix(data_padded, wet_signal, dry_wet, kill_start_dw)

    # Wendet EQ an
    mixed_eq = mixed
    try:
        if mixed is not None and mixed.size > 0 and (not np.isclose(bass_gain, 1.0) or not np.isclose(treble_gain, 1.0)):
             n_fft = mixed.shape[0]
             if n_fft >= 2:
                 fft_data = np.fft.rfft(mixed, axis=0)
                 freqs = np.fft.rfftfreq(n_fft, d=1.0 / rate); bass_cutoff = 250; treble_cutoff = 4000
                 bass_mask = (freqs > 1e-6) & (freqs <= bass_cutoff); treble_mask = freqs >= treble_cutoff
                 fft_data[bass_mask] *= np.clip(bass_gain, 0.1, 5.0)
                 fft_data[treble_mask] *= np.clip(treble_gain, 0.1, 5.0)
                 mixed_eq = np.fft.irfft(fft_data, n=n_fft, axis=0).astype(np.float32)
    except Exception as e: print(f"Fehler (EQ_external): {e}"); traceback.print_exc()

    # Endgültige Normalisierung (nur bei Bedarf)
    if mixed_eq is None or mixed_eq.size == 0: return np.zeros((0, 2), dtype=np.float32)
    max_val = np.max(np.abs(mixed_eq)); mixed_norm = mixed_eq
    if max_val > 1.0: print(f"  Normalisiere externen IR-Output (max val war {max_val:.3f})"); mixed_norm = mixed_eq / max_val
    elif np.any(mixed_eq) and max_val < 1e-9: mixed_norm = np.zeros_like(mixed_eq)

    if mixed_norm.ndim == 1: mixed_norm = np.stack((mixed_norm, mixed_norm), axis=1)
    elif mixed_norm.shape[1] != 2: mixed_norm = mixed_norm[:, :2]
    return mixed_norm.astype(np.float32)

def apply_surround_panning_3d(audio_data, x_pos, y_pos, z_pos):
    """Verteilt ein Stereo-Signal auf 6 Kanäle (5.1) basierend auf 3D-Position."""
    if audio_data is None or audio_data.size == 0: return np.zeros((0, 6), dtype=np.float32)
    try:
        x = np.clip(float(x_pos), 0.0, 1.0); y = np.clip(float(y_pos), 0.0, 1.0); z = np.clip(float(z_pos), 0.0, 1.0)
        if audio_data.ndim == 1: audio_data = np.stack((audio_data, audio_data), axis=1)
        elif audio_data.shape[1] == 1: audio_data = np.repeat(audio_data, 2, axis=1)
        elif audio_data.shape[1] != 2: audio_data = audio_data[:, :2]
        audio_data = audio_data.astype(np.float32); num_samples = audio_data.shape[0]

        # Berechnet Gains für Links/Rechts und Vorne/Hinten (Square Root Panning)
        gain_l = math.sqrt(1.0 - x); gain_r = math.sqrt(x)
        gain_f_base = math.sqrt(1.0 - y); gain_re_base = math.sqrt(y)
        # Z-Position beeinflusst Vorne/Hinten-Balance leicht
        z_effect_scale = abs(y - 0.5) * 0.3; z_pull = (0.5 - z) * z_effect_scale
        gain_f = max(0, gain_f_base + z_pull); gain_re = max(0, gain_re_base - z_pull)

        # Berechnet Kanal-Gains
        fl = gain_l * gain_f; fr = gain_r * gain_f; rl = gain_l * gain_re; rr = gain_r * gain_re
        center_x_factor = math.cos((x - 0.5) * math.pi); center = center_x_factor * gain_f
        mono_mix_for_c_lfe = (audio_data[:, 0] + audio_data[:, 1]) * 0.707
        lfe_gain = 0.15; lfe_signal = mono_mix_for_c_lfe * lfe_gain

        # Weist Signale den Kanälen zu
        surround_data = np.zeros((num_samples, 6), dtype=audio_data.dtype)
        surround_data[:, 0] = audio_data[:, 0] * fl # FL
        surround_data[:, 1] = audio_data[:, 1] * fr # FR
        surround_data[:, 2] = mono_mix_for_c_lfe * center # C
        surround_data[:, 3] = lfe_signal # LFE
        surround_data[:, 4] = audio_data[:, 0] * rl # RL
        surround_data[:, 5] = audio_data[:, 1] * rr # RR

        # Normalisiert das gesamte 6-Kanal-Signal, falls Clipping auftritt
        max_val = np.max(np.abs(surround_data))
        if max_val > 1.0: print(f"  Normalisiere 3D Panning Output (max val war {max_val:.3f})"); surround_data /= max_val
        elif np.any(surround_data) and max_val < 1e-9: surround_data = np.zeros_like(surround_data)

        return surround_data.astype(np.float32)
    except Exception as e:
        print(f"Fehler in apply_surround_panning_3d: {e}"); traceback.print_exc()
        num_samples_fallback = audio_data.shape[0] if audio_data is not None and audio_data.ndim == 2 else 0
        return np.zeros((num_samples_fallback, 6), dtype=np.float32)

def apply_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Wendet eine einfache Verzögerung an (Zero-Padding am Anfang, Trimmen am Ende)."""
    if not isinstance(signal, np.ndarray) or signal.ndim != 2: return signal
    delay_samples = int(delay_samples);
    if delay_samples <= 0: return signal
    num_samples, num_channels = signal.shape
    padding = np.zeros((delay_samples, num_channels), dtype=signal.dtype)
    delayed_signal = np.concatenate((padding, signal), axis=0)
    return delayed_signal[:num_samples, :] # Trimmt auf Original-Länge

def map_channels(data_5_1: np.ndarray, target_layout_name: str, rate: int, z_pos: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """Mappt internes 6-Kanal-Audio auf das Ziel-Layout, versucht Inhalte für 7.1/5.1.2 zu generieren."""
    if target_layout_name not in CHANNEL_LAYOUTS:
        print(f"Warnung: Unbekanntes Ziel-Layout '{target_layout_name}'. Nutze 5.1 Standard.")
        target_layout_name = DEFAULT_CHANNEL_LAYOUT
    layout_info = CHANNEL_LAYOUTS[target_layout_name]
    target_channels = layout_info["channels"]; target_names = layout_info["names"]

    if data_5_1 is None or not isinstance(data_5_1, np.ndarray) or data_5_1.ndim != 2 or data_5_1.shape[1] != 6:
        print("Fehler (map_channels): Eingangsdaten sind kein gültiges 6-Kanal-Audio."); return np.zeros((0, target_channels), dtype=np.float32), target_names

    num_samples = data_5_1.shape[0]
    output_data = np.zeros((num_samples, target_channels), dtype=data_5_1.dtype)

    try:
        if target_layout_name == "Stereo":
            c_gain = 0.707; r_gain = 0.5
            output_data[:, 0] = data_5_1[:, 0] + data_5_1[:, 2] * c_gain + data_5_1[:, 4] * r_gain # L
            output_data[:, 1] = data_5_1[:, 1] + data_5_1[:, 2] * c_gain + data_5_1[:, 5] * r_gain # R
            print(f"  6ch -> {target_channels}ch (Stereo Downmix)")
        elif target_layout_name == "5.1 (Standard)":
            output_data = data_5_1 # Direkte Kopie
            print(f"  6ch -> {target_channels}ch (5.1) - Direkte Kopie")
        elif target_layout_name == "7.1 (Surround)":
            output_data[:, 0:6] = data_5_1[:, 0:6] # Kopiert FL, FR, C, LFE, RL, RR
            delay_ms_side = 12; delay_samples_side = int(rate * delay_ms_side / 1000)
            rl_ch = data_5_1[:, 4:5]; rr_ch = data_5_1[:, 5:6]
            output_data[:, 6:7] = apply_delay(rl_ch, delay_samples_side) * 0.7 # SL
            output_data[:, 7:8] = apply_delay(rr_ch, delay_samples_side) * 0.7 # SR
            print(f"  6ch -> {target_channels}ch (7.1) - Generiere SL/SR mit {delay_ms_side}ms Delay")
        elif target_layout_name == "5.1.2 (Atmos Light)":
            output_data[:, :6] = data_5_1[:, :6] # Kopiert FL, FR, C, LFE, RL, RR
            delay_ms_height = 18; delay_samples_height = int(rate * delay_ms_height / 1000)
            height_gain_factor = np.clip(float(z_pos), 0.0, 1.0) * 0.6
            rl_ch = data_5_1[:, 4:5]; rr_ch = data_5_1[:, 5:6]
            tfl = apply_delay(rl_ch, delay_samples_height) * height_gain_factor # TFL
            tfr = apply_delay(rr_ch, delay_samples_height) * height_gain_factor # TFR
            output_data[:, 6:7] = tfl; output_data[:, 7:8] = tfr
            print(f"  6ch -> {target_channels}ch (5.1.2) - Generiere TFL/TFR mit {delay_ms_height}ms Delay, Z-Skala: {height_gain_factor:.2f}")

        # Finale Normalisierung bei Bedarf
        max_final_val = np.max(np.abs(output_data))
        if max_final_val > 1.0: print(f"  Normalisiere gemappten Output (max val war {max_final_val:.3f})."); output_data /= max_final_val
        elif np.any(output_data) and max_final_val < 1e-9: output_data = np.zeros_like(output_data)

        print(f"  Kanalmapping zu {target_channels}ch ({target_layout_name}) erfolgreich.")
        return output_data, target_names
    except Exception as e:
        print(f"Unerwarteter Fehler beim Kanalmapping: {e}"); traceback.print_exc()
        if data_5_1 is not None and data_5_1.shape[1] == 6:
             print("  Mapping fehlgeschlagen, gebe 5.1 als Fallback zurück."); layout_51 = CHANNEL_LAYOUTS["5.1 (Standard)"]
             max_51_val = np.max(np.abs(data_5_1));
             if max_51_val > 1.0: data_5_1 /= max_51_val
             return data_5_1, layout_51["names"]
        else: print("  Mapping fehlgeschlagen, gebe leeres Array zurück."); return np.zeros((0, target_channels), dtype=np.float32), target_names

def plot_waveform_and_spectrogram_v4(file_path, title="Audio"):
    """Lädt eine Audiodatei und plottet Wellenformen aller Kanäle sowie ein Spektrogramm des ersten Kanals."""
    actual_path = getattr(file_path, 'name', file_path)
    fig = None; plot_path = None

    try:
        if not actual_path or not isinstance(actual_path, str) or not os.path.exists(actual_path):
            raise FileNotFoundError(f"Visualizer Fehler (v4): Ungültiger Pfad '{actual_path}'")
        print(f"Visualizer v4: Lade Audio von '{actual_path}'...")
        try:
            info = sf.info(actual_path); rate = info.samplerate; channels = info.channels; duration = info.duration
            if channels == 0 or duration <= 0: raise ValueError(f'Keine gültigen Audiodaten in: {os.path.basename(actual_path)}')
            data_float, rate_check = sf.read(actual_path, dtype='float32', always_2d=True)
            if rate != rate_check: rate = rate_check
            if data_float.size == 0: raise ValueError(f'Leere Audiodaten nach Laden von: {os.path.basename(actual_path)}')
            print(f"  Geladen: {duration:.2f}s, {rate} Hz, {channels} Kanäle")
        except Exception as load_err: print(f"Visualizer Fehler (v4): Laden fehlgeschlagen: {load_err}"); raise load_err

        # Kanalnamen bestimmen
        plot_ch_names = [f"Ch {i+1}" for i in range(channels)]
        for layout_name, layout_info in CHANNEL_LAYOUTS.items():
            if layout_info["channels"] == channels: plot_ch_names = layout_info["names"]; print(f"  Layout erkannt: {layout_name}"); break

        # Plot-Layout bestimmen
        max_wf_rows = 4; wf_rows = min(max_wf_rows, (channels + 1) // 2); total_rows = wf_rows + 1
        height_ratios = [1] * wf_rows + [max(2, wf_rows)]; fig_height = 2.0 * total_rows + 1.0
        print(f"  Plotte: {wf_rows} Wellenform-Reihen, 1 Spektrogramm-Reihe. Höhe: {fig_height:.1f}")
        fig = plt.figure(figsize=(12, fig_height))
        gs = fig.add_gridspec(total_rows, 2, height_ratios=height_ratios, hspace=0.5, wspace=0.15)
        plot_title = f"Audioanalyse (v4): {title} - {os.path.basename(actual_path)} ({channels}-Kanal)"
        fig.suptitle(plot_title, fontsize=14)
        time_axis = np.linspace(0, duration, num=data_float.shape[0]) if rate > 0 else np.arange(data_float.shape[0])
        base_ax = None

        # Wellenformen plotten
        for i in range(channels):
            row = i // 2; col = i % 2
            if row >= wf_rows: print(f"  Stoppe Wellenform-Plot bei Kanal {i}."); break
            ax = fig.add_subplot(gs[row, col], sharex=base_ax if base_ax else None)
            if base_ax is None: base_ax = ax
            ax.plot(time_axis, data_float[:, i], lw=1); ax.set_title(f"{plot_ch_names[i]}", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.6); ax.tick_params(axis='y', labelsize='x-small')
            ax.set_ylim([-1.05, 1.05]); ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
            if col == 0: ax.set_ylabel("Amplitude", fontsize='small')
            if row < wf_rows - 1: plt.setp(ax.get_xticklabels(), visible=False)
            else: ax.tick_params(axis='x', labelsize='x-small', rotation=0)

        # Spektrogramm plotten (erster Kanal)
        spec_ax = fig.add_subplot(gs[wf_rows, :], sharex=base_ax if base_ax else None)
        spec_data = data_float[:, 0].flatten(); spec_title = f"Spektrogramm ({plot_ch_names[0]})"
        if spec_data.size > 0 and rate > 0:
            try:
                print("  Berechne Spektrogramm...")
                if duration > 30: nperseg = 4096
                elif duration > 5: nperseg = 2048
                else: nperseg = 1024
                nperseg = min(nperseg, spec_data.shape[0]); noverlap = nperseg // 2
                if nperseg < 2: raise ValueError("Signal zu kurz für Spektrogramm.")
                f, t, Sxx = spectrogram(spec_data, fs=rate, nperseg=nperseg, noverlap=noverlap, window='hann')
                if Sxx.size == 0: raise ValueError("Spektrogramm-Berechnung ergab leeres Array.")
                print("  Plotte Spektrogramm...")
                Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
                median_db = np.median(Sxx_db); max_db = np.max(Sxx_db)
                vmin = max(median_db - 40, max_db - 80); vmax = max_db
                if vmin >= vmax: vmin = vmax - 10
                img = spec_ax.pcolormesh(t, f, Sxx_db, shading='auto', cmap='magma', vmin=vmin, vmax=vmax, rasterized=True)
                spec_ax.set_yscale('symlog', linthresh=100, linscale=0.5); spec_ax.set_ylim(bottom=20, top=rate/2)
                cbar = fig.colorbar(img, ax=spec_ax, format='%+2.0f dB', pad=0.01, aspect=40)
                cbar.set_label('Intensität (dB)', size='small'); cbar.ax.tick_params(labelsize='x-small')
                print("  Spektrogramm geplottet.")
            except MemoryError as me: print(f"Fehler (Spektrogramm MemoryError): {me}"); spec_ax.text(0.5, 0.5, f'Spektrogramm:\nMemoryError', ha='center', va='center', color='red', transform=spec_ax.transAxes)
            except ValueError as ve: print(f"Fehler (Spektrogramm ValueError): {ve}"); spec_ax.text(0.5, 0.5, f'Spektrogramm nicht berechenbar:\n{ve}', ha='center', va='center', color='orange', transform=spec_ax.transAxes)
            except Exception as spe: print(f"Fehler (Spektrogramm): {spe}"); traceback.print_exc(); spec_ax.text(0.5, 0.5, f'Spektrogramm Fehler:\n{type(spe).__name__}', ha='center', va='center', color='orange', transform=spec_ax.transAxes)
            spec_ax.set_title(spec_title, fontsize=12); spec_ax.set_ylabel('Frequenz (Hz)'); spec_ax.set_xlabel('Zeit (s)')
            spec_ax.tick_params(axis='both', labelsize='small'); plt.setp(spec_ax.get_xticklabels(), visible=True); spec_ax.tick_params(axis='x', labelsize='x-small', rotation=0)
        else:
            spec_ax.text(0.5, 0.5, 'Keine Daten für Spektrogramm.', ha='center', va='center', transform=spec_ax.transAxes)
            spec_ax.set_title("Spektrogramm", fontsize=12); spec_ax.set_xlabel('Zeit (s)'); spec_ax.set_ylabel('Frequenz (Hz)')
            spec_ax.tick_params(axis='both', labelsize='small'); plt.setp(spec_ax.get_xticklabels(), visible=True)

        print("  Layout anpassen und Plot speichern...")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="vis_v4_") as tmp_plot:
            plot_path = tmp_plot.name; plt.savefig(plot_path, dpi=120)
        print(f"  Plot gespeichert: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Visualizer Fehler (v4): Unerwarteter Fehler beim Plotten von '{actual_path}': {e}"); traceback.print_exc()
        try:
            err_fig, err_ax = plt.subplots(1, 1, figsize=(10, 3))
            error_message = f'Fehler beim Plotten (v4):\n{type(e).__name__}: {str(e)[:100]}'
            err_ax.text(0.5, 0.5, error_message, ha='center', va='center', color='red', fontsize=9, wrap=True)
            err_ax.set_axis_off()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="vis_err_") as tmp_plot_err: error_plot_path = tmp_plot_err.name
            err_fig.savefig(error_plot_path); print(f"  Fehler-Plot gespeichert: {error_plot_path}")
            return error_plot_path
        except Exception as fallback_err: print(f"!!! KRITISCHER FEHLER: Konnte Fehler-Plot nicht erstellen: {fallback_err}"); return None
    finally:
        if fig is not None: plt.close(fig)
        if 'err_fig' in locals() and locals()['err_fig'] is not None: plt.close(locals()['err_fig'])

def calculate_audio_metrics(data, rate):
    """Berechnet LUFS, True Peak (dBFS) und RMS (dBFS) für Audiodaten."""
    metrics = {'lufs': None, 'true_peak_dbfs': None, 'rms_dbfs': None}
    if data is None or not isinstance(data, np.ndarray) or data.size == 0 or rate <= 0: return metrics
    if data.ndim != 2:
        if data.ndim == 1: data = data[:, np.newaxis] # Versucht 1D zu 2D zu machen
        else: print(f"Warnung (Metriken): Ungültige Datenform {data.shape}."); return metrics
    num_samples, num_channels = data.shape
    if num_channels == 0: return metrics

    try:
        meter = pyln.Meter(rate)
        # LUFS: Mittelwert der ersten beiden Kanäle (oder erster bei Mono)
        num_lufs_ch = min(num_channels, 2)
        data_lufs = data[:, 0] if num_lufs_ch == 1 else np.mean(data[:, :num_lufs_ch], axis=1)
        if np.max(np.abs(data_lufs)) < 1e-6: metrics['lufs'] = -np.inf
        else:
            try: metrics['lufs'] = meter.integrated_loudness(data_lufs)
            except Exception as lufs_err: print(f"Fehler bei LUFS-Berechnung: {lufs_err}"); metrics['lufs'] = None

        # Peak und RMS über alle Kanäle
        linear_peak = np.max(np.abs(data)) if data.size > 0 else 0.0
        rms_linear = np.sqrt(np.mean(data**2)) if data.size > 0 else 0.0
        metrics['true_peak_dbfs'] = 20 * math.log10(linear_peak) if linear_peak > 1e-15 else -np.inf
        metrics['rms_dbfs'] = 20 * math.log10(rms_linear) if rms_linear > 1e-15 else -np.inf
    except ValueError as ve: # Fehler bei Meter-Initialisierung
        print(f"Fehler beim Initialisieren des Lautheitsmessers (Rate={rate}?): {ve}")
        # Versucht Peak/RMS trotzdem zu berechnen
        if 'true_peak_dbfs' not in metrics or metrics['true_peak_dbfs'] is None:
             linear_peak = np.max(np.abs(data)) if data.size > 0 else 0.0
             metrics['true_peak_dbfs'] = 20 * math.log10(linear_peak) if linear_peak > 1e-15 else -np.inf
        if 'rms_dbfs' not in metrics or metrics['rms_dbfs'] is None:
             rms_linear = np.sqrt(np.mean(data**2)) if data.size > 0 else 0.0
             metrics['rms_dbfs'] = 20 * math.log10(rms_linear) if rms_linear > 1e-15 else -np.inf
    except Exception as e:
        print(f"Fehler bei Metrikberechnung: {e}"); traceback.print_exc()
        metrics = {k: metrics.get(k) for k in ['lufs', 'true_peak_dbfs', 'rms_dbfs']} # Stellt sicher, dass Keys existieren
    return metrics

def run_audio_profiler_v4(original_file_obj, processed_file_obj):
    """Analysiert und vergleicht Original- und bearbeitete Audiodatei."""
    report_lines = [f"## 📊 Audio-Profiler Bericht ({APP_VERSION})"]
    results_orig = {}; results_proc = {}

    original_path = getattr(original_file_obj, 'name', None)
    processed_path = getattr(processed_file_obj, 'name', None)
    if not original_path or not os.path.exists(original_path): return "\n".join(report_lines + ["\n**Fehler:** Originaldatei fehlt."])
    if not processed_path or not os.path.exists(processed_path): return "\n".join(report_lines + ["\n**Fehler:** Bearbeitete Datei fehlt."])

    try:
        print(f"Profiler v4: Lade Original '{original_path}'")
        data_orig, rate_orig = sf.read(original_path, dtype='float32', always_2d=True)
        channels_orig = data_orig.shape[1]; duration_orig = data_orig.shape[0] / rate_orig if rate_orig > 0 else 0
        print(f"  Original Info: {duration_orig:.2f}s, {rate_orig} Hz, {channels_orig} ch")
        print(f"Profiler v4: Lade Bearbeitet '{processed_path}'")
        data_proc, rate_proc = sf.read(processed_path, dtype='float32', always_2d=True)
        channels_proc = data_proc.shape[1]; duration_proc = data_proc.shape[0] / rate_proc if rate_proc > 0 else 0
        print(f"  Bearbeitet Info: {duration_proc:.2f}s, {rate_proc} Hz, {channels_proc} ch")

        proc_ch_names = [f"Ch{i+1}" for i in range(channels_proc)]
        for layout_name, info in CHANNEL_LAYOUTS.items():
            if info["channels"] == channels_proc: proc_ch_names = info["names"]; break
    except Exception as e:
        report_lines.append(f"\n**Ladefehler:**\n```\n{traceback.format_exc()}\n```")
        return "\n".join(report_lines)

    if rate_orig != rate_proc: return "\n".join(report_lines + [f"\n**Fehler:** Sample-Raten unterschiedlich ({rate_orig} vs {rate_proc})."])
    rate = rate_orig

    print("  Berechne Metriken für Original...")
    results_orig = calculate_audio_metrics(data_orig, rate)
    print("  Berechne Metriken für Bearbeitet...")
    results_proc = calculate_audio_metrics(data_proc, rate)

    # --- Bericht erstellen ---
    report_lines.append(f"\n### 📋 Basis-Infos")
    report_lines.append(f"- **Original:** {channels_orig} Kanal{'e' if channels_orig!=1 else ''}, {duration_orig:.2f}s @ {rate} Hz")
    report_lines.append(f"- **Bearbeitet:** {channels_proc} Kanal{'e' if channels_proc!=1 else ''} ({', '.join(proc_ch_names)}), {duration_proc:.2f}s @ {rate} Hz")
    report_lines.append("\n### 🔊 Lautheit & Pegel")
    report_lines.append("| Metrik          | Original              | Bearbeitet            | Änderung      |")
    report_lines.append("|-----------------|-----------------------|-----------------------|---------------|")

    def fmt_met(v, u, d=1): return f"{v:.{d}f} {u}" if v is not None and not np.isinf(v) else ("-inf " + u if np.isinf(v) and v < 0 else "N/A")
    def fmt_diff(vp, vo, u, d=1): return f"{vp - vo:+.{d}f} {u}" if vp is not None and vo is not None and not np.isinf(vp) and not np.isinf(vo) else "N/A"

    lufs_o = fmt_met(results_orig.get('lufs'), "LUFS", 2); lufs_p = fmt_met(results_proc.get('lufs'), "LUFS", 2); lufs_diff = fmt_diff(results_proc.get('lufs'), results_orig.get('lufs'), "LU", 2)
    peak_o = fmt_met(results_orig.get('true_peak_dbfs'), "dBFS", 1); peak_p = fmt_met(results_proc.get('true_peak_dbfs'), "dBFS", 1); peak_diff = fmt_diff(results_proc.get('true_peak_dbfs'), results_orig.get('true_peak_dbfs'), "dB", 1)
    rms_o = fmt_met(results_orig.get('rms_dbfs'), "dBFS", 1); rms_p = fmt_met(results_proc.get('rms_dbfs'), "dBFS", 1); rms_diff = fmt_diff(results_proc.get('rms_dbfs'), results_orig.get('rms_dbfs'), "dB", 1)

    report_lines.append(f"| Integrated LUFS | {lufs_o:<21} | {lufs_p:<21} | {lufs_diff:<13} |")
    report_lines.append(f"| True Peak       | {peak_o:<21} | {peak_p:<21} | {peak_diff:<13} |")
    report_lines.append(f"| RMS             | {rms_o:<21} | {rms_p:<21} | {rms_diff:<13} |")

    report_lines.append("\n### ↔️ Stereo-Breite (FL/FR, Side RMS)")
    # calculate_stereo_width_metric definition wird hier angenommen
    def calculate_stereo_width_metric(left_channel, right_channel):
        """Calculates a stereo width metric based on the RMS of the side signal."""
        if left_channel.size != right_channel.size or left_channel.size == 0: return 0.0
        side_signal = (left_channel - right_channel) * 0.5; rms_side = np.sqrt(np.mean(side_signal**2))
        return rms_side # Einfacher RMS-Wert des Seitensignals

    width_metric_orig = 0.0; width_metric_proc = 0.0; width_change_str = "N/A"
    try:
        if channels_orig >= 2: width_metric_orig = calculate_stereo_width_metric(data_orig[:, 0], data_orig[:, 1])
        if channels_proc >= 2: width_metric_proc = calculate_stereo_width_metric(data_proc[:, 0], data_proc[:, 1])
        orig_str = f"{width_metric_orig:.4f}" if channels_orig >= 2 else "Mono/N/A"; proc_str = f"{width_metric_proc:.4f}" if channels_proc >= 2 else "Mono/N/A"
        report_lines.append(f"- Original: {orig_str}"); report_lines.append(f"- Bearbeitet: {proc_str}")
        if channels_orig >= 2 and channels_proc >= 2:
            if width_metric_orig > 1e-9: width_change_percent = ((width_metric_proc / width_metric_orig) - 1) * 100; width_change_str = f"{width_change_percent:+.1f}%"
            else: width_change_str = "Änderung von Stille" if width_metric_proc > 1e-9 else "Bleibt Stille"
        elif channels_proc >= 2 and channels_orig < 2: width_change_str = "Mono -> Stereo"
        elif channels_orig >= 2 and channels_proc < 2: width_change_str = "Stereo -> Mono"
        else: width_change_str = "Beide Mono oder <2 Kanäle"
        report_lines.append(f"- **Änderung:** {width_change_str}")
    except Exception as e: report_lines.append(f"- Fehler bei Breitenberechnung: {e}")

    report_lines.append("\n### 🔊 Kanalpegel (Bearbeitet, RMS dBFS)")
    if channels_proc > 0 and data_proc.size > 0:
         report_lines.append("| Kanal     | RMS Pegel |"); report_lines.append("|-----------|-----------|")
         lfe_level = -np.inf
         for i in range(channels_proc):
             ch_name = proc_ch_names[i]; ch_data = data_proc[:, i]
             rms_ch = np.sqrt(np.mean(ch_data**2)) if ch_data.size > 0 else 0.0
             dbfs_ch = 20 * math.log10(rms_ch) if rms_ch > 1e-15 else -np.inf
             dbfs_ch_str = fmt_met(dbfs_ch, "dBFS", 1); report_lines.append(f"| {ch_name:<9} | {dbfs_ch_str:<9} |")
             if i == 3 and ch_name == "LFE": lfe_level = dbfs_ch
         if not np.isinf(lfe_level): report_lines.append(f"\n*Hinweis: LFE-Pegel ({fmt_met(lfe_level, 'dBFS', 1)}) ist typischerweise niedriger.*")
    else: report_lines.append("- Keine Kanäle oder leere Daten in bearbeiteter Datei.")

    report_lines.append("\n### 📜 Zusammenfassung")
    summary = "Vergleich zeigt: "; changes = []
    if lufs_diff != "N/A": changes.append(f"Lautheitsänderung ({lufs_diff})")
    if width_change_str not in ["N/A", "Beide Mono oder <2 Kanäle", "Bleibt Stille"]: changes.append(f"Stereobreite ({width_change_str})")
    if not np.isinf(lfe_level) and lfe_level > -40: changes.append(f"LFE ({fmt_met(lfe_level, 'dBFS', 0)})")
    if not changes: summary += "minimale Unterschiede oder nicht zutreffend."
    else: summary += ", ".join(changes) + "."
    report_lines.append(summary)

    print("Profiler v4: Bericht generiert.")
    return "\n".join(report_lines)

# --- Funktionen für UI-Interaktion (Marker, Presets) ---

def update_marker_image(x_pos, y_pos, base_image_path_param=None):
    """Zeichnet einen Marker auf das Hintergrundbild und gibt den Pfad zum temporären Bild zurück."""
    base_path = base_image_path_param
    if not base_path or not isinstance(base_path, str) or not os.path.exists(base_path):
        if BASE_SURROUND_MAP_PATH and os.path.exists(BASE_SURROUND_MAP_PATH): base_path = BASE_SURROUND_MAP_PATH
        else: print("Marker Fehler: Basispfad ungültig."); return None
    if not base_path or not os.path.exists(base_path): print(f"Marker Fehler: Finaler Basispfad ungültig ('{base_path}')."); return None

    try:
        x_float = float(x_pos); y_float = float(y_pos)
        with Image.open(base_path).convert("RGBA") as bg:
            img_width, img_height = bg.size;
            if img_width <= 0 or img_height <= 0: print("Marker Fehler: Ungültige Bilddimensionen."); return None
            x_pixel = int(np.clip(x_float, 0.0, 1.0) * (img_width - 1))
            y_pixel = int(np.clip(y_float, 0.0, 1.0) * (img_height - 1))
            output_img = bg.copy(); draw = ImageDraw.Draw(output_img)
            radius = max(5, min(img_width, img_height) // 60); outline_width = max(1, radius // 4)
            bbox = (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius)
            draw.ellipse(bbox, fill=(255, 0, 0, 200), outline=(255, 255, 255, 220), width=outline_width)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="marker_") as tmp_file:
            tmp_file_path = tmp_file.name; output_img.save(tmp_file_path, "PNG")
        return tmp_file_path
    except Exception as e: print(f"Fehler beim Aktualisieren des Markerbildes: {e}"); traceback.print_exc(); return None

def update_controls_from_click(evt: gr.SelectData):
    """Verarbeitet Klicks auf das Surround-Bild, aktualisiert Slider und Markerbild."""
    base_map_path = BASE_SURROUND_MAP_PATH
    if not base_map_path or not os.path.exists(base_map_path): print("Kartenklick Fehler: Basispfad fehlt."); return gr.update(), gr.update(), gr.update(value=None)
    try:
        with Image.open(base_map_path) as img: img_width, img_height = img.size
        if img_width <=0 or img_height <= 0: print("Kartenklick Fehler: Ungültige Bilddimensionen."); return gr.update(), gr.update(), gr.update(value=None)
        if not evt or not hasattr(evt, 'index') or not isinstance(evt.index, (list, tuple)) or len(evt.index) < 2: print(f"Kartenklick Fehler: Ungültige Eventdaten: {evt}"); return gr.update(), gr.update(), gr.update()
        x_click, y_click = evt.index[0], evt.index[1]
        x_norm = np.clip(x_click / img_width, 0.0, 1.0); y_norm = np.clip(y_click / img_height, 0.0, 1.0)
        new_marker_path = update_marker_image(x_norm, y_norm, base_map_path)
        # Gibt Updates für X-Slider, Y-Slider und das *untere* Markerbild zurück
        return gr.update(value=x_norm), gr.update(value=y_norm), gr.update(value=new_marker_path) if new_marker_path else gr.update()
    except Exception as e: print(f"Fehler bei Verarbeitung des Kartenklicks: {e}"); traceback.print_exc(); return gr.update(), gr.update(), gr.update()

def handle_slider_change(x_pos, y_pos):
    """Aktualisiert das Markerbild bei Änderung der X/Y-Slider."""
    base_map_path = BASE_SURROUND_MAP_PATH
    if not base_map_path: print("Slider Update Fehler: Basispfad fehlt."); return gr.update()
    new_marker_path = update_marker_image(x_pos, y_pos, base_map_path)
    # Gibt Update für das *untere* Markerbild zurück
    return gr.update(value=new_marker_path) if new_marker_path else gr.update()

def list_presets_for_dropdown_v4():
    """Holt Preset-Dateien für das Dropdown-Menü."""
    ensure_preset_dir()
    try: return sorted([f for f in os.listdir(PRESET_DIR) if f.endswith(".json")], key=str.lower)
    except Exception as e: print(f"Fehler beim Auflisten der v4 Presets: {e}"); traceback.print_exc(); return []

def save_current_preset_v4(preset_name, *control_values):
    """Speichert die aktuellen UI-Einstellungen als JSON-Preset."""
    ensure_preset_dir(); preset_name = preset_name.strip() if isinstance(preset_name, str) else ""
    if not preset_name: return "⚠️ Bitte gültigen Preset-Namen angeben!", gr.update()
    safe_filename_base = "".join(c for c in preset_name if c.isalnum() or c in (' ', '_', '-')).strip()
    safe_filename = safe_filename_base.replace(' ', '_') + "_v4.json"
    if not safe_filename_base or safe_filename == "_v4.json": return "⚠️ Ungültiger Preset-Name.", gr.update()
    preset_path = os.path.join(PRESET_DIR, safe_filename)

    if 'NUM_PRESET_CONTROLS_V4' not in globals() or len(control_values) != NUM_PRESET_CONTROLS_V4:
        err_msg = f"❌ Interner Fehler: Falsche Anzahl Steuerungswerte ({len(control_values)})."
        print(err_msg); return err_msg, gr.update()

    keys_in_order = [
        "use_external_ir", "hall_type", "material", "room_size", "diffusion", "air_absorption",
        "early_level", "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain",
        "x_pos", "y_pos", "z_pos", "target_layout"
    ]
    preset_data = {}
    for i, key in enumerate(keys_in_order):
        preset_data[key] = bool(control_values[i]) if key == "use_external_ir" else control_values[i]
    preset_data["_source_name"] = preset_name if safe_filename_base != preset_name else None
    preset_data["_version"] = APP_VERSION

    try:
        with open(preset_path, "w", encoding='utf-8') as f: json.dump(preset_data, f, indent=4, ensure_ascii=False)
        print(f"Preset gespeichert: {preset_path}"); save_last_preset(safe_filename)
        new_choices = list_presets_for_dropdown_v4()
        return f"✅ Preset '{safe_filename}' gespeichert!", gr.update(choices=new_choices, value=safe_filename)
    except Exception as e: print(f"Fehler beim Speichern: {e}"); traceback.print_exc(); return f"❌ Fehler beim Speichern: {e}", gr.update()

def load_selected_preset_v4(preset_file):
    """Lädt Einstellungen aus einer ausgewählten JSON-Preset-Datei."""
    num_controls = NUM_PRESET_CONTROLS_V4 if 'NUM_PRESET_CONTROLS_V4' in globals() else 16
    if not preset_file or not isinstance(preset_file, str): print("Preset Laden: Keine Datei gewählt."); return [gr.update()] * num_controls
    preset_path = os.path.join(PRESET_DIR, preset_file)
    if not os.path.exists(preset_path): print(f"Preset Laden Fehler: {preset_path} nicht gefunden."); gr.Warning(f"Preset '{preset_file}' nicht gefunden."); return [gr.update()] * num_controls

    try:
        print(f"Lade Preset: {preset_path}")
        with open(preset_path, "r", encoding='utf-8') as f: preset_data = json.load(f)
        save_last_preset(preset_file)

        defaults_load = {
            "use_external_ir": False, "hall_type": DEFAULT_HALL_TYPE, "material": DEFAULT_MATERIAL, "room_size": 100.0, "diffusion": 0.5, "air_absorption": 0.1,
            "early_level": 0.8, "late_level": 0.6, "dry_wet": 0.5, "dry_wet_kill_start": 0.5, "bass_gain": 1.0, "treble_gain": 1.0, "x_pos": 0.5, "y_pos": 0.5, "z_pos": 0.5, "target_layout": DEFAULT_CHANNEL_LAYOUT
        }
        keys_in_order = [
            "use_external_ir", "hall_type", "material", "room_size", "diffusion", "air_absorption", "early_level", "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain", "x_pos", "y_pos", "z_pos", "target_layout"
        ]
        updates = []
        for i, key in enumerate(keys_in_order):
             value = preset_data.get(key, defaults_load.get(key))
             if value is None: value = defaults_load.get(key)
             if key == "use_external_ir": value = bool(value)
             elif key in ["room_size", "diffusion", "air_absorption", "early_level", "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain", "x_pos", "y_pos", "z_pos"]:
                 try: value = float(value)
                 except (ValueError, TypeError): print(f"Warnung: Konnte Preset-Wert für '{key}' nicht in Float konvertieren."); value = defaults_load.get(key)
             updates.append(gr.update(value=value))
        if len(updates) != num_controls: print(f"!!! WARNUNG: Preset Laden gibt {len(updates)} Updates zurück, erwartet {num_controls}"); updates.extend([gr.update()] * (num_controls - len(updates)))
        print(f"Preset '{preset_file}' erfolgreich geladen (v{preset_data.get('_version', 'Unbekannt')}).")
        return updates
    except Exception as e: print(f"Fehler beim Laden von Preset '{preset_file}': {e}"); traceback.print_exc(); gr.Error(f"Fehler Laden Preset: {e}"); return [gr.update()] * num_controls

def delete_selected_preset_v4(preset_file):
    """Löscht die ausgewählte Preset-Datei."""
    if not preset_file or not isinstance(preset_file, str): return "⚠️ Kein Preset zum Löschen gewählt!", gr.update()
    preset_path = os.path.join(PRESET_DIR, preset_file); status_message = ""
    if os.path.exists(preset_path):
        try:
            os.remove(preset_path); status_message = f"🗑️ Preset '{preset_file}' gelöscht!"
            print(status_message);
            if load_last_preset() == preset_file: save_last_preset(""); print("  Letzte Preset-Referenz gelöscht.")
        except Exception as e: status_message = f"❌ Fehler beim Löschen: {e}"; print(f"ERROR: {status_message}"); traceback.print_exc()
    else: status_message = f"⚠️ Preset '{preset_file}' nicht gefunden."; print(status_message)
    new_choices = list_presets_for_dropdown_v4()
    return status_message, gr.update(choices=new_choices, value=None)

def export_presets_as_zip_v4():
    """Erstellt ein ZIP-Archiv aller Preset-Dateien."""
    ensure_preset_dir()
    zip_path = None # Initialize zip_path to None
    try:
        json_files = [f for f in os.listdir(PRESET_DIR) if f.endswith(".json")]
        if not json_files:
            gr.Info("Keine v4 Presets zum Exportieren gefunden.")
            print("ZIP Export: No presets found.")
            return None # Return None if no files to zip

        # Create zip file in temp directory safely
        with tempfile.NamedTemporaryFile(delete=False, suffix="_presets_v4.zip", prefix="audio_studio_") as tmp_zip:
            zip_path = tmp_zip.name # Assign the path *after* successful creation

        print(f"Exporting {len(json_files)} presets to {zip_path}...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in json_files:
                full_path = os.path.join(PRESET_DIR, filename)
                # arcname ensures no directory structure is stored in the zip
                zipf.write(full_path, arcname=filename)

        print(f"Presets successfully exported to {zip_path}")
        return zip_path # Return the path to the created zip file

    except Exception as e:
        error_msg = f"ZIP Export Fehler: {e}"
        print(f"ERROR: {error_msg}"); traceback.print_exc()
        gr.Error(error_msg)

        # Clean up temp zip file if it was created before the error occurred
        if zip_path and os.path.exists(zip_path):
            print(f"  Attempting to clean up temporary zip file: {zip_path}")
            try:
                os.remove(zip_path)
                print(f"  Successfully removed temporary zip file.")
            except OSError as remove_err:
                # Log if cleanup fails, but don't raise another error
                print(f"  Warning: Failed to remove temporary zip file {zip_path}: {remove_err}")

        return None # Return None on error

# --- Definition der Funktion apply_raytrace_convolution_3d (Muss VOR process_audio_main_v41 stehen) ---
def apply_raytrace_convolution_3d(audio_file_path, external_ir_path, use_external_ir_cb, hall_type_val, room_size_val, diffusion_val, air_absorption_val, base_early_level, base_late_level, dry_wet, dry_wet_kill_start, bass_gain, treble_gain, x_pos, y_pos, z_pos, material, target_channel_layout):
    """Haupt-Audioverarbeitungs-Pipeline: Lädt Audio, wendet Hall/IR, EQ, Panning und Kanalmapping an."""
    output_metrics_text = "Verarbeitung fehlgeschlagen."
    temp_output_file_path = None # Initialize path to None

    try:
        # --- Parameter Validation and Conversion ---
        print("--- apply_raytrace_convolution_3d ---")
        print(f"  Input Audio: {audio_file_path}, Use Ext IR: {use_external_ir_cb}, Hall: {hall_type_val}, Mat: {material}, Target: {target_channel_layout}")
        try:
            use_external_ir = bool(use_external_ir_cb)
            room_size = float(room_size_val); diffusion = float(diffusion_val); air_absorption = float(air_absorption_val)
            early_lvl = float(base_early_level); late_lvl = float(base_late_level); dw = float(dry_wet); dw_kill = float(dry_wet_kill_start)
            bass = float(bass_gain); treble = float(treble_gain); x = float(x_pos); y = float(y_pos); z = float(z_pos)
            print(f"  Params: Size={room_size:.1f}, Diff={diffusion:.2f}, AirAbs={air_absorption:.2f}, Early={early_lvl:.2f}, Late={late_lvl:.2f}, DW={dw:.2f}, DWKill={dw_kill:.2f}, Bass={bass:.2f}, Treble={treble:.2f}, X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
            if not isinstance(hall_type_val, str) or not isinstance(material, str) or not isinstance(target_channel_layout, str): raise ValueError("Ungültiger String-Inputtyp.")
        except (ValueError, TypeError, AttributeError) as e: error_msg = f"Fehlerhafte Eingabeparameter: {e}"; print(f"ERROR: {error_msg}"); return None, None, error_msg

        # --- Audio Input Handling ---
        file_path = audio_file_path
        print(f"Lade Audio '{os.path.basename(file_path)}'")
        try:
            samples_float, rate = sf.read(file_path, dtype='float32', always_2d=True)
            channels_in = samples_float.shape[1]; duration_in = samples_float.shape[0] / rate if rate > 0 else 0
            if samples_float.size == 0: raise ValueError("Audiodatei ist leer.")
            print(f"  Audio Info: {duration_in:.2f}s, {rate} Hz, {channels_in} ch")
        except Exception as load_err: error_msg = f"Fehler beim Laden: {load_err}"; print(f"ERROR: {error_msg}"); traceback.print_exc(); return None, None, error_msg

        # --- Stereo-Signal für Verarbeitung sicherstellen ---
        if channels_in == 1: samples_stereo = np.repeat(samples_float, 2, axis=1); print("  Input Mono -> Stereo dupliziert.")
        elif channels_in > 2: samples_stereo = samples_float[:, :2]; print(f"  Input {channels_in}ch -> Erste 2 Kanäle verwendet.")
        else: samples_stereo = samples_float

        output_stereo = None # Initialisiert Faltungsergebnis

        # --- Verarbeitungspfad: Externe IR oder Interner Hall ---
        if use_external_ir:
            # (Code für Externe IR - unverändert)
            print("Verarbeitungsmodus: Externe IR")
            ir_file_path = getattr(external_ir_path, 'name', external_ir_path)
            if not ir_file_path or not os.path.exists(ir_file_path): error_msg = "Externe IR gewählt, aber keine Datei gefunden."; print(f"WARNUNG: {error_msg}"); return None, None, error_msg
            try:
                print(f"  Lade externe IR: '{os.path.basename(ir_file_path)}'")
                loaded_external_ir_data, ir_rate = sf.read(ir_file_path, dtype='float32', always_2d=True)
                ir_ch = loaded_external_ir_data.shape[1]; ir_dur = loaded_external_ir_data.shape[0] / ir_rate if ir_rate > 0 else 0
                if loaded_external_ir_data.size == 0: raise ValueError("Externe IR-Datei ist leer.")
                if ir_rate != rate:
                    print(f"  WARNUNG: IR Rate ({ir_rate}Hz) != Audio Rate ({rate}Hz). Resample IR..."); num_samples_resampled = int(loaded_external_ir_data.shape[0] * rate / ir_rate)
                    if num_samples_resampled > 0: loaded_external_ir_data = resample(loaded_external_ir_data, num_samples_resampled, axis=0); print(f"  IR resampled auf {loaded_external_ir_data.shape[0]} Samples @ {rate} Hz."); ir_dur = loaded_external_ir_data.shape[0] / rate
                    else: raise ValueError("Resampling würde IR-Länge Null ergeben.")
                if loaded_external_ir_data.ndim != 2 or loaded_external_ir_data.shape[1] != 2: error_msg = "Externe IR muss Stereo sein."; print(f"ERROR: {error_msg}"); return None, None, error_msg
                print(f"  Externe IR geladen: {ir_dur:.3f}s, {rate}Hz, 2ch")
            except Exception as ir_err: error_msg = f"Fehler Laden/Resample IR: {ir_err}"; print(f"ERROR: {error_msg}"); traceback.print_exc(); return None, None, error_msg
            print("  Wende externe IR-Faltung, Mix, EQ an...")
            output_stereo = convolve_audio_external_ir(samples_stereo, loaded_external_ir_data, dw, bass, treble, rate, dw_kill)
        else:
            # (Code für Internen Hall - unverändert)
            print("Verarbeitungsmodus: Intern generierter Hall")
            adj_duration, adj_ref_count, adj_max_delay, adj_split_time = adjust_parameters_for_3d(hall_type_val, room_size, z)
            directionality = compute_final_directionality_3d(x, y, z, hall_type_val, diffusion, dw)
            print(f"  Angepasste Parameter: Dur={adj_duration:.2f}s, Refs={adj_ref_count}, MaxDel={adj_max_delay:.3f}s, Split={adj_split_time:.3f}s, Dir={directionality:.3f}")
            early_ir, late_ir = generate_impulse_response_split_3d(rate, adj_duration, adj_ref_count, adj_max_delay, material, directionality, adj_split_time, diffusion)
            print(f"  Generierte IRs: Early len={len(early_ir)}, Late len={len(late_ir)}")
            adapted_early_level, adapted_late_level = adapt_early_late_levels(dw, early_lvl, late_lvl)
            print(f"  Wende interne Faltung an: EarlyLvl={adapted_early_level:.2f}, LateLvl={adapted_late_level:.2f}, DW={dw:.2f}, AirAbs={air_absorption:.2f}")
            output_stereo = convolve_audio_split_3d(samples_stereo, early_ir, late_ir, adapted_early_level, adapted_late_level, dw, bass, treble, rate, dw_kill, air_absorption)

        if output_stereo is None or output_stereo.size == 0: error_msg = "Fehler während Faltung (Ergebnis leer)."; print(f"ERROR: {error_msg}"); return None, None, error_msg
        print(f"  Faltungsergebnis-Shape: {output_stereo.shape}")

        # --- Panning und Kanalmapping ---
        print(f"  Wende 3D Surround Panning an..."); internal_surround_output = apply_surround_panning_3d(output_stereo, x, y, z)
        if internal_surround_output is None or internal_surround_output.shape[1] != 6: error_msg = "Fehler beim 3D Panning."; print(f"ERROR: {error_msg}"); return None, None, error_msg
        print(f"  Internes 6ch Output-Shape: {internal_surround_output.shape}")
        print(f"  Mappe auf Ziel-Layout: {target_channel_layout}"); final_output_data, final_channel_names = map_channels(internal_surround_output, target_channel_layout, rate, z)
        if final_output_data is None or final_output_data.size == 0: error_msg = "Fehler beim Kanal-Mapping."; print(f"ERROR: {error_msg}"); return None, None, error_msg
        final_channel_count = final_output_data.shape[1]; print(f"  Finales Output-Shape: {final_output_data.shape}, Namen: {final_channel_names}")

        # --- Metriken berechnen ---
        print(f"  Berechne finale Audio-Metriken..."); calculated_metrics = calculate_audio_metrics(final_output_data, rate)
        lufs_val = calculated_metrics.get('lufs'); peak_val = calculated_metrics.get('true_peak_dbfs'); rms_val = calculated_metrics.get('rms_dbfs')
        lufs_str = f"{lufs_val:.2f}" if lufs_val is not None and not np.isinf(lufs_val) else "N/A"
        peak_str = f"{peak_val:.1f}" if peak_val is not None and not np.isinf(peak_val) else "-inf"
        rms_str = f"{rms_val:.1f}" if rms_val is not None and not np.isinf(rms_val) else "-inf"
        output_metrics_text = f"LUFS: {lufs_str} | Peak: {peak_str} dBFS | RMS: {rms_str} dBFS"; print(f"    Metriken: {output_metrics_text}")

        # --- Ergebnis speichern ---
        print(f"Speichere finales {final_channel_count}-Kanal WAV...");
        try:
            # Wichtig: temp_output_file_path wird hier *definiert*
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="processed_") as tmpfile_obj: temp_output_file_path = tmpfile_obj.name
            final_output_clipped = np.clip(final_output_data, -0.9999, 0.9999)
            if not np.all(np.isfinite(final_output_clipped)): print("  Warnung: Ersetze nicht-finite Werte durch 0."); final_output_clipped = np.nan_to_num(final_output_clipped, nan=0.0, posinf=0.0, neginf=0.0)
            sf.write(temp_output_file_path, final_output_clipped, rate, subtype='PCM_16', format='WAV')
            print(f"  Erfolgreich gespeichert: {temp_output_file_path}")
            # Gibt den Pfad zur temporären Datei und den Metrik-String zurück
            return temp_output_file_path, temp_output_file_path, output_metrics_text
        except Exception as write_err: error_msg = f"Fehler beim Schreiben der WAV-Datei: {write_err}"; print(f"ERROR: {error_msg}"); traceback.print_exc();
        # Aufräumen, wenn Schreiben fehlschlägt, aber Pfad schon existiert
        if temp_output_file_path and os.path.exists(temp_output_file_path):
             try: os.remove(temp_output_file_path);
             except OSError: pass
        return None, None, error_msg

    # --- Allgemeiner Exception Handler ---
    except Exception as e:
        error_msg = f"Unerwarteter Fehler in apply_raytrace_convolution_3d: {e}"
        print(f"--- UNERWARTETER FEHLER ---"); traceback.print_exc()
        # --- KORREKTUR HIER ---
        # Räumt die temporäre Datei auf, falls sie vor dem Fehler erstellt wurde
        if temp_output_file_path and os.path.exists(temp_output_file_path):
            print(f"  Versuche, temporäre Ausgabedatei aufzuräumen: {temp_output_file_path}")
            try:
                os.remove(temp_output_file_path)
                print("  Temporäre Ausgabedatei erfolgreich entfernt.")
            except OSError as remove_err:
                print(f"  Warnung: Konnte temporäre Ausgabedatei nicht entfernen: {remove_err}")
        # --- ENDE KORREKTUR ---
        return None, None, error_msg

    # --- Allgemeiner Exception Handler ---
    except Exception as e:
        error_msg = f"Unerwarteter Fehler in apply_raytrace_convolution_3d: {e}"
        print(f"--- UNERWARTETER FEHLER ---"); traceback.print_exc()
        # --- KORREKTUR HIER ---
        # Räumt die temporäre Datei auf, falls sie vor dem Fehler erstellt wurde
        if temp_output_file_path and os.path.exists(temp_output_file_path):
            print(f"  Versuche, temporäre Ausgabedatei aufzuräumen: {temp_output_file_path}")
            try:
                os.remove(temp_output_file_path)
                print("  Temporäre Ausgabedatei erfolgreich entfernt.")
            except OSError as remove_err:
                print(f"  Warnung: Konnte temporäre Ausgabedatei nicht entfernen: {remove_err}")
        # --- ENDE KORREKTUR ---
        return None, None, error_msg


# --- Haupt-Wrapper-Funktion für die Verarbeitung (Ruft apply_raytrace_convolution_3d auf) ---
def process_audio_main_v41(audio_upload_path, mic_record_path, external_ir_file, *args):
    """Wählt Audioquelle, ruft Verarbeitung auf, kopiert Ergebnis für Gradio."""
    source_file_to_process = None
    upload_path = getattr(audio_upload_path, 'name', audio_upload_path)
    mic_path = getattr(mic_record_path, 'name', mic_record_path)
    print("--- process_audio_main_v41 ---"); print(f"  Upload: {upload_path}, Mic: {mic_path}, ExtIR: {getattr(external_ir_file, 'name', external_ir_file)}, Args: {len(args)}")
    valid_upload = upload_path and os.path.exists(upload_path) and os.path.getsize(upload_path) > 100
    valid_mic = mic_path and os.path.exists(mic_path) and os.path.getsize(mic_path) > 1024
    if valid_upload: source_file_to_process = upload_path; print(f"Processing -> Quelle: Upload ({os.path.basename(source_file_to_process)})")
    elif valid_mic: source_file_to_process = mic_path; print(f"Processing -> Quelle: Mikrofon ({os.path.basename(source_file_to_process)})")
    else: warning_msg = "Keine gültige Audioquelle."; gr.Warning(warning_msg); print(f"Warnung: {warning_msg}"); return None, None, "Keine gültige Quelle"

    expected_args = NUM_PRESET_CONTROLS_V4
    if len(args) != expected_args: error_msg = f"Interner Fehler: Argumentanzahl ({len(args)} statt {expected_args})."; print(f"ERROR: {error_msg}"); gr.Error(error_msg); return None, None, error_msg

    print(f"Rufe apply_raytrace_convolution_3d auf mit Quelle: {source_file_to_process}")
    try:
        # HIER wird apply_raytrace_convolution_3d aufgerufen
        player_path_temp, download_path_temp, metrics_str = apply_raytrace_convolution_3d(
            audio_file_path=source_file_to_process, external_ir_path=external_ir_file, use_external_ir_cb=args[0], hall_type_val=args[1], room_size_val=args[3], diffusion_val=args[4],
            air_absorption_val=args[5], base_early_level=args[6], base_late_level=args[7], dry_wet=args[8], dry_wet_kill_start=args[9], bass_gain=args[10], treble_gain=args[11],
            x_pos=args[12], y_pos=args[13], z_pos=args[14], material=args[2], target_channel_layout=args[15] )
        print(f"  apply_raytrace... zurückgegeben: temp_path={player_path_temp}, metrics='{metrics_str}'")
    except Exception as proc_err: # Fängt Fehler ab, falls apply_raytrace... selbst einen Fehler wirft
        error_msg = f"Fehler während apply_raytrace_convolution_3d: {proc_err}"; print(f"ERROR: {error_msg}"); traceback.print_exc(); gr.Error(error_msg); return None, None, error_msg

    # --- Kopiere Ergebnisdatei für den Player ---
    player_path_final = None; download_path_final = None
    if player_path_temp and os.path.exists(player_path_temp):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="gradio_out_") as temp_for_gradio: gradio_serve_path = temp_for_gradio.name
            shutil.copy2(player_path_temp, gradio_serve_path) # Kopiert Dateiinhalt
            player_path_final = gradio_serve_path; download_path_final = gradio_serve_path
            print(f"  Kopierte Temp-Datei {player_path_temp} nach {gradio_serve_path} für Gradio.")
            try: os.remove(player_path_temp); print(f"  Original-Temp-Datei {player_path_temp} gelöscht.")
            except OSError as e: print(f"  Warnung: Konnte Original-Temp-Datei {player_path_temp} nicht löschen: {e}")
        except Exception as copy_err:
            error_msg_copy = f"Fehler beim Kopieren der Temp-Datei für Gradio: {copy_err}"; print(f"ERROR: {error_msg_copy}"); traceback.print_exc()
            gr.Warning(f"{error_msg_copy}. Versuche Originalpfad (könnte fehlschlagen).")
            player_path_final = player_path_temp; download_path_final = download_path_temp # Fallback
            if isinstance(metrics_str, str): metrics_str += " (Warnung: Player-Fehler möglich!)"
            else: metrics_str = "Verarbeitung OK, aber Player-Fehler möglich!"
    else: print("  Verarbeitung fehlgeschlagen oder kein Pfad zurückgegeben."); player_path_final = None; download_path_final = None

    print(f"--- process_audio_main_v41 gibt zurück: Player={player_path_final}, Download={download_path_final}, Metrics='{metrics_str}' ---")
    return player_path_final, download_path_final, metrics_str


# --- Gradio UI Definition ---
theme = gr.themes.Soft(primary_hue=gr.themes.colors.cyan, secondary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate)

with gr.Blocks(theme=theme, title=f"Audio Raytracing Studio {APP_VERSION}") as demo:

    # --- Tab: Audio-Verarbeitung & Positionierung ---
    with gr.Tab("🎶 Audio-Verarbeitung & Positionierung"):
        gr.Markdown(f"# 🎶 Audio Raytracing Studio {APP_VERSION}")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="🔊 Audio hochladen", type="filepath", show_download_button=False)
                mic_input = gr.Audio(label="🎤 Mikrofonaufnahme", sources=["microphone"], type="filepath", show_download_button=False)
                use_external_ir = gr.Checkbox(label="💡 Externe Stereo IR verwenden?", value=False, info="Überschreibt interne Hallgenerierung.")
                external_ir_input = gr.File(label="📂 Externe IR-Datei (Stereo WAV)", file_types=['.wav'], interactive=False)
            with gr.Column(scale=1):
                target_layout_dropdown = gr.Dropdown(choices=list(CHANNEL_LAYOUTS.keys()), value=DEFAULT_CHANNEL_LAYOUT, label="🎯 Ziel-Layout")
                output_audio = gr.Audio(label="🎧 Ergebnis anhören", type="filepath", interactive=False)
                output_metrics_display = gr.Textbox(label="📊 Ergebnis-Metriken (Gesamt)", value="Noch keine Verarbeitung.", interactive=False, lines=1)
                download = gr.File(label="💾 Download Ergebnis", interactive=False)
        gr.Markdown("*Hinweis: Die 'Ergebnis-Metriken' oben sind für das gesamte Audio nach Verarbeitung.*")

        with gr.Accordion("⚙️ Raum & Hall Charakteristik (Interne Generierung)", open=True):
             with gr.Row():
                 with gr.Column(scale=1):
                      hall_type = gr.Dropdown(choices=["Plate", "Room", "Cathedral"], label="🏩️ Hall-Typ", value=DEFAULT_HALL_TYPE, interactive=True)
                      material_choice = gr.Dropdown(choices=list(material_absorption.keys()), value=DEFAULT_MATERIAL, label="🧱 Material", interactive=True)
                      hall_info_text = gr.Markdown(update_hall_info(DEFAULT_HALL_TYPE), elem_id="hall-info-md")
                 with gr.Column(scale=1):
                      room_size_slider = gr.Slider(10, 1000, value=100, step=10, label="📏 Raumgröße (m³)", interactive=True)
                      diffusion_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="💫 Diffusion", interactive=True)
                      air_absorption_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="💨 Luftabsorption", interactive=True)
             with gr.Row():
                  early_level = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Basis Early Level", interactive=True)
                  late_level = gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="Basis Late Level", interactive=True)

        with gr.Accordion("🔊 Mix & EQ", open=True):
             with gr.Row():
                 with gr.Column(scale=1):
                     dry_wet = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Dry/Wet Mix")
                     dry_wet_kill_start_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Dry Kill Start")
                 with gr.Column(scale=1):
                     bass_gain = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Bass Gain")
                     treble_gain = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Treble Gain")

        with gr.Accordion("📍 3D Positionierung", open=True):
             with gr.Row():
                 with gr.Column(scale=2):
                     gr.Markdown("Klicke für X/Y Position"); surround_image = gr.Image(label="Karte (Klicken für X/Y)", value=BASE_SURROUND_MAP_PATH, interactive=True, type="filepath")
                     surround_output_image = gr.Image(label="🎯 Position (X/Y)", interactive=False, type="filepath")
                 with gr.Column(scale=1):
                     surround_x = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="↔️ X (L/R)")
                     surround_y = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="↕️ Y (F/B)")
                     surround_z = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label=" L Z (U/O)")
                     gr.Markdown("*(X: 0=L↔️1=R)*\n\n*(Y: 0=F↔️1=B)*\n\n*(Z: 0=U↔️1=O)*")
        process_button = gr.Button("➡️ Verarbeiten & Anhören!", variant="primary")

    # --- Tab: Visualizer & Profiler ---
    with gr.Tab("📊 Visualizer & ⚖️ Profiler"):
         with gr.Row():
              with gr.Column(scale=1):
                   gr.Markdown("## 📊 Visualizer (v4)"); input_file_vis = gr.File(label="🔍 Original (Visualizer)", file_types=['audio'])
                   output_file_vis = gr.File(label="🔍 Bearbeitet (Visualizer)", file_types=['audio'])
                   with gr.Row(): load_last_result_vis = gr.Button(" Lade letztes Ergebnis (Bearb.)", scale=1); show_visuals_button = gr.Button("📊 Visualisieren", variant="secondary", scale=1)
                   input_image = gr.Image(label="🔵 Original Vis", interactive=False, type="filepath"); output_image = gr.Image(label="🟠 Bearbeitet Vis", interactive=False, type="filepath")
              with gr.Column(scale=1):
                   gr.Markdown("## ⚖️ Audio-Profiler (v4)"); profiler_input_original = gr.File(label=" Lade Original (Profiler)", file_types=['audio'])
                   profiler_input_processed = gr.File(label=" Lade Bearbeitet (Profiler)", file_types=['audio'])
                   with gr.Row(): load_last_result_prof = gr.Button(" Lade letztes Ergebnis (Bearb.)", scale=1); profiler_analyze_button = gr.Button("🚀 Analysieren!", variant="primary", scale=1)
                   profiler_report_output = gr.Markdown(label="📋 Analysebericht", value="*Bericht wird hier angezeigt...*")

    # --- Tab: Preset-Editor ---
    with gr.Tab("🛠 Preset-Editor (v4)"):
        gr.Markdown("## 🛠 Presets (v4 Format)")
        with gr.Row():
            preset_name_input = gr.Textbox(label="📝 Preset-Name", placeholder="Name für neues Preset...")
            save_preset_button = gr.Button("💾 Speichern", variant="primary")
        save_status = gr.Label(label="Status", value="Bereit.")
        with gr.Row():
            preset_list = gr.Dropdown(label="📂 Presets (v4)", choices=[], interactive=True, allow_custom_value=False)
            with gr.Column(scale=1, min_width=160):
                load_preset_button = gr.Button("📥 Laden")
                refresh_presets_button = gr.Button("🔄 Liste neu laden")
                delete_preset_button = gr.Button("🗑️ Löschen", variant="stop")
        with gr.Row():
            export_presets_button = gr.Button("📦 ZIP Export")
            zip_download = gr.File(label="📦 Download ZIP", interactive=False)

    # --- Tab: Hilfe & Dokumentation ---
    with gr.Tab("ℹ️ Hilfe & Dokumentation"):
         gr.Markdown(f"""
         ## 🎶 Audio Raytracing Studio {APP_VERSION} - Hilfe
         **Neu in v4.1 ('Astonishing Attempt Edition'):**
         *   **Ergebnis-Metriken:** LUFS, True Peak, RMS des Ergebnisses direkt unter dem Player.
         *   **Kanal-Mapper (Erweitert):** Generiert einfache Inhalte für Stereo, 7.1, 5.1.2 aus 5.1 Basis.
         *   **Visualizer/Profiler:** Buttons zum Laden des letzten Ergebnisses.
         *   **Stabilität & Kommentare:** Code verbessert.
         *   **(Aus v4.0):** 3D (Z), Externe IRs, Raumgröße, Diffusion, Luftabsorption, neuer Profiler & Visualizer.
         ---
         **Ziel:** Simulation von Audio in 3D-Raum mit Mehrkanal-Positionierung.
         **Bedienung:**
         1. Audio laden (Upload/Mic). 2. Modus wählen (Intern/Extern IR). 3. Parameter anpassen (Raum, Mix, EQ). 4. Positionieren (X/Y/Z). 5. Ziel-Layout wählen. 6. Verarbeiten. 7. Analyse (Optional). 8. Presets (Optional).
         **Technische Hinweise:** Ausgabe: WAV (PCM16). Benötigt FFmpeg für Nicht-WAV. Bibliotheken: numpy, gradio, scipy, matplotlib, pillow, soundfile, pyloudnorm.
         """)

    # --- Liste aller Preset-steuerbaren Controls (NACHDEM sie im Layout definiert wurden) ---
    ALL_PRESET_CONTROLS_V4 = [
        use_external_ir, hall_type, material_choice, room_size_slider, diffusion_slider, air_absorption_slider,
        early_level, late_level, dry_wet, dry_wet_kill_start_slider, bass_gain, treble_gain,
        surround_x, surround_y, surround_z, target_layout_dropdown
    ]
    NUM_PRESET_CONTROLS_V4 = len(ALL_PRESET_CONTROLS_V4)
    print(f"Definierte {NUM_PRESET_CONTROLS_V4} Steuerelemente für Presets.")

    # --- Event Handlers ---
    hall_type.change(fn=update_hall_info, inputs=[hall_type], outputs=[hall_info_text])

    def toggle_ir_controls_v4(use_external):
        """Schaltet die Interaktivität der Hall-Parameter um."""
        is_external = bool(use_external); is_internal = not is_external
        updates = {ctrl: gr.update(interactive=is_internal) for ctrl in [hall_type, material_choice, room_size_slider, diffusion_slider, air_absorption_slider, early_level, late_level]}
        updates[external_ir_input] = gr.update(interactive=is_external)
        print(f"IR Controls umgeschaltet: External Mode = {is_external}")
        # Gibt Updates in der Reihenfolge der 'interactive_control_outputs' zurück
        return updates[external_ir_input], updates[hall_type], updates[material_choice], updates[room_size_slider], updates[diffusion_slider], updates[air_absorption_slider], updates[early_level], updates[late_level]

    interactive_control_outputs = [external_ir_input, hall_type, material_choice, room_size_slider, diffusion_slider, air_absorption_slider, early_level, late_level]
    use_external_ir.change(fn=toggle_ir_controls_v4, inputs=[use_external_ir], outputs=interactive_control_outputs)

    surround_image.select(fn=update_controls_from_click, inputs=None, outputs=[surround_x, surround_y, surround_output_image])
    surround_x.input(fn=handle_slider_change, inputs=[surround_x, surround_y], outputs=[surround_output_image])
    surround_y.input(fn=handle_slider_change, inputs=[surround_x, surround_y], outputs=[surround_output_image])

    show_visuals_button.click(fn=plot_waveform_and_spectrogram_v4, inputs=[input_file_vis, gr.Textbox("Original", visible=False)], outputs=[input_image])
    show_visuals_button.click(fn=plot_waveform_and_spectrogram_v4, inputs=[output_file_vis, gr.Textbox("Bearbeitet", visible=False)], outputs=[output_image])
    profiler_analyze_button.click(fn=run_audio_profiler_v4, inputs=[profiler_input_original, profiler_input_processed], outputs=[profiler_report_output])
    load_last_result_vis.click(fn=lambda x: x, inputs=[download], outputs=[output_file_vis])
    load_last_result_prof.click(fn=lambda x: x, inputs=[download], outputs=[profiler_input_processed])

    save_preset_button.click(fn=save_current_preset_v4, inputs=[preset_name_input] + ALL_PRESET_CONTROLS_V4, outputs=[save_status, preset_list])
    load_preset_button.click(fn=load_selected_preset_v4, inputs=[preset_list], outputs=ALL_PRESET_CONTROLS_V4
        ).then(fn=handle_slider_change, inputs=[surround_x, surround_y], outputs=[surround_output_image]
        ).then(fn=update_hall_info, inputs=[hall_type], outputs=[hall_info_text]
        ).then(fn=toggle_ir_controls_v4, inputs=[use_external_ir], outputs=interactive_control_outputs
        ).then(lambda p: f"Preset '{p}' geladen." if p else "Kein Preset gewählt.", inputs=[preset_list], outputs=save_status)
    refresh_presets_button.click(fn=list_presets_for_dropdown_v4, inputs=[], outputs=[preset_list]).then(lambda: "Presetliste aktualisiert.", inputs=None, outputs=save_status)
    delete_preset_button.click(fn=delete_selected_preset_v4, inputs=[preset_list], outputs=[save_status, preset_list])
    export_presets_button.click(fn=export_presets_as_zip_v4, inputs=[], outputs=[zip_download]).then(lambda x: gr.update(value="ZIP Export erfolgreich." if x else "Export fehlgeschlagen."), inputs=[zip_download], outputs=save_status)

    # Haupt-Verarbeitungsknopf
    process_button.click(
        fn=process_audio_main_v41, # Diese Funktion beinhaltet die Kopierlogik
        inputs=[audio_input, mic_input, external_ir_input] + ALL_PRESET_CONTROLS_V4,
        outputs=[output_audio, download, output_metrics_display] # Player, Download, Metrik-Text
    )

    # --- App Initialisierung beim Start ---
    def on_start_v41():
        """Initialisiert den App-Zustand beim Start."""
        print(f"App {APP_VERSION} startet, führe on_start aus...")
        ensure_preset_dir()
        if not os.path.exists(BASE_SURROUND_MAP_PATH):
            print(f"Basiskarte '{BASE_SURROUND_MAP_PATH}' nicht gefunden. Erstelle Platzhalter...")
            try:
                 placeholder_img = Image.new('RGB', (300, 200), color=(210, 210, 225)); draw_p = ImageDraw.Draw(placeholder_img)
                 draw_p.text((10, 10), "Surround Map\n(Click for X/Y)", fill=(0,0,0)); placeholder_img.save(BASE_SURROUND_MAP_PATH); print(f"  Platzhalterkarte erstellt.")
            except Exception as e: print(f"  FEHLER: Konnte Platzhalterkarte nicht erstellen: {e}")

        available_presets = list_presets_for_dropdown_v4(); last_preset = load_last_preset()
        defaults = {"use_external_ir": False, "hall_type": DEFAULT_HALL_TYPE, "material": DEFAULT_MATERIAL, "room_size": 100.0, "diffusion": 0.5, "air_absorption": 0.1, "early_level": 0.8, "late_level": 0.6, "dry_wet": 0.5, "dry_wet_kill_start": 0.5, "bass_gain": 1.0, "treble_gain": 1.0, "x_pos": 0.5, "y_pos": 0.5, "z_pos": 0.5, "target_layout": DEFAULT_CHANNEL_LAYOUT}
        keys_in_preset_order = ["use_external_ir", "hall_type", "material", "room_size", "diffusion", "air_absorption", "early_level", "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain", "x_pos", "y_pos", "z_pos", "target_layout"]
        loaded_values_dict = defaults.copy(); preset_to_select = None

        if last_preset:
            preset_path = os.path.join(PRESET_DIR, last_preset)
            if os.path.exists(preset_path):
                try:
                    with open(preset_path, "r", encoding='utf-8') as f: preset_data = json.load(f)
                    print(f"  Lade Werte aus letztem Preset: {last_preset}")
                    for key in keys_in_preset_order:
                         if key in preset_data and preset_data[key] is not None:
                             if key == "use_external_ir": loaded_values_dict[key] = bool(preset_data[key])
                             elif key in ["room_size", "diffusion", "air_absorption", "early_level", "late_level", "dry_wet", "dry_wet_kill_start", "bass_gain", "treble_gain", "x_pos", "y_pos", "z_pos"]:
                                 try: loaded_values_dict[key] = float(preset_data[key])
                                 except (ValueError, TypeError): pass
                             else: loaded_values_dict[key] = preset_data[key]
                    preset_to_select = last_preset; print(f"  Werte erfolgreich geladen.")
                except Exception as e: print(f"Fehler beim Parsen des letzten Presets '{last_preset}': {e}. Nutze Standard."); loaded_values_dict = defaults.copy(); save_last_preset("")
            else: print(f"Letzte Preset-Datei '{last_preset}' nicht gefunden. Nutze Standard."); save_last_preset("")
        else: print("Keine letzte Preset-Datei gefunden. Nutze Standard.")

        print("  Generiere initiales Markerbild..."); initial_marker_path = update_marker_image(loaded_values_dict["x_pos"], loaded_values_dict["y_pos"])
        print(f"  Initialer Markerpfad: {initial_marker_path}"); initial_hall_info = update_hall_info(loaded_values_dict["hall_type"])
        print("  Generiere initiale Interaktivitätszustände..."); initial_interactivity_updates = toggle_ir_controls_v4(loaded_values_dict["use_external_ir"])

        # Erstellt Liste der Rückgabewerte für demo.load (Reihenfolge muss zu on_start_outputs passen!)
        output_updates_list = [gr.update(choices=available_presets, value=preset_to_select)] # 1. Presetliste
        for key in keys_in_preset_order: output_updates_list.append(gr.update(value=loaded_values_dict[key])) # 2. Alle Steuerungswerte
        output_updates_list.append(gr.update(value=BASE_SURROUND_MAP_PATH if os.path.exists(BASE_SURROUND_MAP_PATH) else None)) # 3. Oberes Bild (Basiskarte)
        output_updates_list.append(gr.update(value=initial_marker_path)) # 4. Unteres Bild (Marker)
        output_updates_list.append(gr.update(value=initial_hall_info)) # 5. Hall-Infotext
        output_updates_list.extend(list(initial_interactivity_updates)) # 6. Interaktivitäts-Updates
        output_updates_list.append(gr.update(value="Bereit. Bitte Audio laden.")) # 7. Metrik-Anzeige
        print(f"on_start v4.1: {len(output_updates_list)} Updates vorbereitet. Initialisierung abgeschlossen.")
        return output_updates_list

    # Definiert die Ausgabekomponenten für demo.load (Reihenfolge muss zu on_start passen!)
    on_start_outputs = [preset_list, *ALL_PRESET_CONTROLS_V4, surround_image, surround_output_image, hall_info_text, *interactive_control_outputs, output_metrics_display]
    demo.load(fn=on_start_v41, inputs=[], outputs=on_start_outputs)

# --- App Start ---
if __name__ == "__main__":
    print("-" * 60); print(f"Starte Audio Raytracing Studio {APP_VERSION}..."); print(f"Preset Verzeichnis: {os.path.abspath(PRESET_DIR)}")
    ensure_preset_dir()
    if not os.path.exists(BASE_SURROUND_MAP_PATH):
        print(f"Basiskarte '{BASE_SURROUND_MAP_PATH}' fehlt vor Start. Erstelle Platzhalter...")
        try:
             img = Image.new('RGB', (300, 200), (210, 210, 225)); draw = ImageDraw.Draw(img)
             draw.text((10, 10), "Surround Map\n(Click for X/Y)", fill=(0,0,0)); img.save(BASE_SURROUND_MAP_PATH); print(f"  Platzhalterkarte erstellt.")
        except Exception as e: print(f"  WARNUNG: Konnte Platzhalterkarte nicht erstellen: {e}")
    print("\nStellen Sie sicher, dass FFmpeg installiert/im PATH ist für Nicht-WAV-Input."); print("Benötigte Bibliotheken: numpy, gradio, scipy, matplotlib, pillow, soundfile, pyloudnorm"); print("-" * 60)
    demo.launch(server_name="0.0.0.0", server_port=8861, debug=True, share=False)
