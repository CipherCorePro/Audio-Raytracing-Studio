import streamlit as st
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import soundfile as sf
from typing import Union




# Hilfsfunktion: LUFS-Messung
def measure_lufs(file_path: Union[str, Path]) -> float:
    """Misst die integrierte LUFS-Lautheit mit ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(file_path),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                "-f", "null", "-"
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stderr
        return extract_lufs_from_output(output)
    except Exception as e:
        st.error(f"LUFS-Messung fehlgeschlagen: {e}")
        return np.nan

# Hilfsfunktion: LUFS-Wert extrahieren
def extract_lufs_from_output(ffmpeg_output: str) -> float | None:
    try:
        lines = ffmpeg_output.splitlines()
        for line in lines:
            if "input_i" in line:  # Korrekte Zeile für input integrated loudness
                lufs_str = line.split(":")[-1].replace('"', '').replace(',', '').strip()
                return float(lufs_str)
    except Exception as e:
        print(f"Fehler beim Parsen der LUFS: {e}")
    return None

# Hilfsfunktion: Audio-Analyse
def analyze_audio(file_path: Union[str, Path]) -> dict:
    """Analysiert grundlegende Eigenschaften einer Audiodatei."""
    try:
        with sf.SoundFile(file_path) as f:
            samplerate = f.samplerate
            channels = f.channels
            subtype = f.subtype
            frames = f.frames
        duration = frames / samplerate
        lufs = measure_lufs(file_path)
        return {
            "Pfad": str(file_path),
            "Abtastrate": samplerate,
            "Kanäle": channels,
            "Bittiefe": subtype,
            "Dauer (Sekunden)": round(duration, 2),
            "LUFS": round(lufs, 2) if not np.isnan(lufs) else "Nicht messbar"
        }
    except Exception as e:
        st.error(f"Audio-Analyse fehlgeschlagen: {e}")
        return {}

# Hilfsfunktion: Konvertierung
def convert_audio(file_path: Union[str, Path], output_format: str, bitrate: str) -> Path:
    """Konvertiert eine Audiodatei ins gewünschte Format."""
    try:
        audio = AudioSegment.from_file(file_path)
        output_dir = Path(tempfile.gettempdir())
        output_file = output_dir / f"{Path(file_path).stem}.{output_format}"
        audio.export(output_file, format=output_format, bitrate=f"{bitrate}k")
        return output_file
    except Exception as e:
        st.error(f"Konvertierung fehlgeschlagen: {e}")
        return Path()

# 💥 NEU: Hilfsfunktion zur LUFS-Normalisierung
def normalize_to_lufs(input_path: Union[str, Path], target_lufs: int = -16) -> Path:
    """Normalisiert eine Audiodatei auf den gewünschten LUFS-Wert."""
    try:
        output_dir = Path(tempfile.gettempdir())
        output_path = output_dir / f"{Path(input_path).stem}_normalized.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_path),
                "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
                str(output_path)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return output_path
    except Exception as e:
        st.error(f"LUFS-Normalisierung fehlgeschlagen: {e}")
        return Path()

# Streamlit-Oberfläche
def main():
    st.set_page_config(page_title="Audio Analyzer Studio", page_icon="🎵", layout="wide")
    st.title("🎵 Audio Analyzer Studio")

    with st.sidebar:
        st.header("🌍 Optionen")
        mode = st.radio("Modus auswählen", ["Dateianalyse", "Dateikonvertierung"])
        bitrate = st.selectbox("Bitrate auswählen (nur für Konvertierung)", ["64", "128", "192", "256", "320"], index=3)
        output_format = st.selectbox("Zielformat", ["wav", "mp3", "flac", "aac", "ogg"], index=1)

    if mode == "Dateianalyse":
        st.header("📃 Audiodatei analysieren")
        file = st.file_uploader("Audiodatei hochladen", type=["wav", "mp3", "flac", "aac", "ogg"])

        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = Path(tmp_file.name)
            analysis = analyze_audio(tmp_file_path)

            if analysis:
                st.success("Analyse abgeschlossen!")
                st.json(analysis)

                # LUFS-Normalisierung anbieten
                if st.button("🔊 Auf -16 LUFS normalisieren"):
                    normalized_path = normalize_to_lufs(tmp_file_path, target_lufs=-16)
                    if normalized_path.exists():
                        st.success("Normalisierung abgeschlossen!")
                        with open(normalized_path, "rb") as f:
                            st.download_button("🎯 Normalisierte Datei herunterladen", f, file_name=normalized_path.name)

    elif mode == "Dateikonvertierung":
        st.header("🔄 Audiodatei konvertieren")
        file = st.file_uploader("Audiodatei hochladen", type=["wav", "mp3", "flac", "aac", "ogg"])

        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = Path(tmp_file.name)

            output_file = convert_audio(tmp_file_path, output_format, bitrate)

            if output_file.exists():
                st.success(f"Konvertierung abgeschlossen: {output_file.name}")
                with open(output_file, "rb") as f:
                    st.download_button("Datei herunterladen", f, file_name=output_file.name)

if __name__ == "__main__":
    main()
