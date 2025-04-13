# 🎶 Audio Raytracing Studio 

Ein web-basiertes Werkzeug zur erstellung von Audioquellen in virtuellen akustischen Umgebungen, zur Anwendung von Halleffekten (prozedural oder via Impulsantwort) und zur Positionierung im 3D-Raum mit Mehrkanal-Audioausgabe.

---

## ✨ Hauptfunktionen

*   **🔊 Audio-Eingabe:** Unterstützt das Hochladen gängiger Audiodateiformate (WAV, MP3, etc. - benötigt FFmpeg für Nicht-WAV) sowie 🎤 Mikrofonaufnahmen direkt im Browser.
*   **💡 Zwei Hall-Modi:**
    *   **Interner Hall:** Prozedural generierter Hall mit wählbaren Typen (Plate, Room, Cathedral) und anpassbaren Parametern.
    *   **Externe Impulsantwort (IR):** Verwendung eigener Stereo-WAV-Dateien als Impulsantwort für präzise Hall-Simulationen.
*   **⚙️ Detaillierte Hall-Parameter (Intern):**
    *   **Hall-Typ:** Grundlegender Charakter des Halls (Plate, Room, Cathedral).
    *   **Material:** Auswahl von Oberflächenmaterialien (Stein, Holz, Teppich etc.) zur Beeinflussung der Absorption.
    *   **Raumgröße (m³):** Skaliert Hallparameter wie Dauer und Delay für unterschiedliche Raumdimensionen.
    *   **Diffusion:** Steuert die Dichte und Glättung des Nachhalls.
    *   **Luftabsorption:** Simuliert die frequenzabhängige Dämpfung durch Luft, beeinflusst hohe Frequenzen im Nachhall.
    *   **Early/Late Levels:** Basis-Lautstärken für frühe Reflexionen und den späten Nachhall.
*   **🎚️ Mix & EQ:**
    *   **Dry/Wet Mix:** Stufenlose Mischung zwischen Originalsignal (Dry) und bearbeitetem Signal (Wet).
    *   **Dry Kill Start:** Dynamisches Ausblenden des Dry-Signals bei hohem Wet-Anteil.
    *   **Einfacher EQ:** Bass- und Höhenanhebung/-absenkung auf das gemischte Signal.
*   **📍 3D-Positionierung:**
    *   Interaktive 2D-Karte zum Setzen der X/Y-Position (Links/Rechts, Vorne/Hinten).
    *   Separate Slider für X, Y und Z (Unten/Oben).
    *   Die Position beeinflusst das Panning und interne Hallparameter (z.B. Direktheit, Höhenkanal-Generierung).
*   **🎯 Mehrkanal-Ausgabe:**
    *   Auswahl verschiedener Ziel-Layouts: Stereo, 5.1, 7.1, 5.1.2 (Atmos Light).
    *   **Kanal-Mapping:** Intern wird in 5.1 gearbeitet. Das Ergebnis wird auf das Ziel-Layout gemappt.
        *   **Stereo:** Downmix aus 5.1.
        *   **7.1 / 5.1.2:** Basis-Generierung von Side- bzw. Height-Kanälen durch Verzögerung/Skalierung der Rear-Kanäle.
*   **💾 Ausgabe & Download:**
    *   Das bearbeitete Audio kann direkt angehört werden.
    *   Bereitstellung eines Download-Links für die resultierende WAV-Datei (16-bit PCM).
*   **📊 Analyse & Metriken:**
    *   **Sofort-Metriken:** Anzeige von LUFS (Integrated), True Peak (dBFS) und RMS (dBFS) des Gesamtergebnisses nach der Verarbeitung.
    *   **Visualizer:** Vergleich von Original und bearbeitetem Audio anhand von Wellenform- und Spektrogramm-Plots.
    *   **Profiler:** Detaillierter Bericht zum Vergleich von Original und Ergebnis (Lautheit, Pegel, Stereobreite, Kanalpegel).
*   **📁 Preset-Management:**
    *   Speichern und Laden von kompletten Einstellungen als JSON-Dateien (v4 Format).
    *   Löschen von Presets.
    *   Aktualisieren der Preset-Liste.
    *   Exportieren aller Presets als ZIP-Archiv.
    *   Die Anwendung merkt sich das zuletzt geladene Preset.

---

## 🚀 Erste Schritte

### Voraussetzungen

*   **Python:** Version 3.8 oder höher empfohlen.
*   **pip:** Python Paket-Installer (normalerweise mit Python enthalten).
*   **FFmpeg:** (Optional, aber **stark empfohlen**) Wird von `soundfile` benötigt, um Audioformate außer WAV (z.B. MP3, FLAC, OGG) zu laden. Stellen Sie sicher, dass `ffmpeg` installiert und im Systempfad (`PATH`) verfügbar ist. Sie können es von [ffmpeg.org](https://ffmpeg.org/download.html) herunterladen.

### Abhängigkeiten

Die Anwendung benötigt folgende Python-Bibliotheken:

*   `numpy`
*   `gradio`
*   `scipy`
*   `matplotlib`
*   `Pillow`
*   `soundfile`
*   `pyloudnorm`

### Installation

1.  **Klonen Sie das Repository oder laden Sie den Code herunter.**
2.  **Navigieren Sie im Terminal zum Projektverzeichnis.**
3.  **Installieren Sie die Abhängigkeiten:**
    ```bash
    pip install numpy gradio scipy matplotlib pillow soundfile pyloudnorm
    ```

### Ausführen der Anwendung

Führen Sie das Python-Skript im Terminal aus:

```bash
python test.py
```

Die Anwendung sollte starten und eine lokale URL ausgeben (standardmäßig `http://0.0.0.0:8861` oder `http://127.0.0.1:8861`). Öffnen Sie diese URL in Ihrem Webbrowser.

---

## 🔧 Benutzungsanleitung

1.  **🎶 Audio laden:**
    *   Verwenden Sie die Upload-Komponente ("🔊 Audio hochladen"), um eine Audiodatei von Ihrem Computer auszuwählen.
    *   Oder verwenden Sie die Mikrofon-Komponente ("🎤 Mikrofonaufnahme"), um direkt aufzunehmen.
    *   *Hinweis: Wenn beides bereitgestellt wird, hat der Upload Vorrang.*
2.  **💡 Modus wählen:**
    *   **Interner Hall:** Lassen Sie die Checkbox "💡 Externe Stereo IR verwenden?" deaktiviert. Passen Sie die Parameter im Akkordeon "⚙️ Raum & Hall Charakteristik" an.
    *   **Externe IR:** Aktivieren Sie die Checkbox und laden Sie eine Stereo-WAV-Impulsantwortdatei über "📂 Externe IR-Datei" hoch. Die internen Hallparameter werden ignoriert.
3.  **⚙️ Hall/Raum anpassen (nur interner Modus):**
    *   Wählen Sie den `Hall-Typ` und das `Material`.
    *   Passen Sie `Raumgröße`, `Diffusion` und `Luftabsorption` an.
    *   Stellen Sie die Grundlautstärken mit `Basis Early Level` und `Basis Late Level` ein.
4.  **🔊 Mix & EQ anpassen:**
    *   Regeln Sie das Verhältnis von Original zu Effekt mit `Dry/Wet Mix`.
    *   Bestimmen Sie, ab wann das Originalsignal ausgeblendet wird, mit `Dry Kill Start`.
    *   Optional: Passen Sie `Bass Gain` und `Treble Gain` an.
5.  **📍 3D Positionieren:**
    *   Klicken Sie in die obere Karte ("Karte (Klicken für X/Y)"), um die X- und Y-Position festzulegen.
    *   Verwenden Sie die Slider `↔️ X`, `↕️ Y`, ` L Z`, um die Position fein einzustellen. Die untere Karte ("🎯 Position (X/Y)") zeigt die aktuelle X/Y-Markerposition.
6.  **🎯 Ziel-Layout wählen:**
    *   Wählen Sie das gewünschte Ausgabeformat aus dem Dropdown-Menü (z.B. "5.1 (Standard)", "Stereo").
7.  **➡️ Verarbeiten:**
    *   Klicken Sie auf den Button "➡️ Verarbeiten & Anhören!".
    *   Die Verarbeitung kann je nach Dateilänge und Einstellungen einige Zeit dauern.
    *   Das Ergebnis erscheint im Player "🎧 Ergebnis anhören".
    *   Die berechneten Gesamtmetriken werden unter "📊 Ergebnis-Metriken" angezeigt.
    *   Ein Link zum Herunterladen der resultierenden WAV-Datei erscheint bei "💾 Download Ergebnis".
8.  **📊 Analyse (Optional):**
    *   Wechseln Sie zum Tab "📊 Visualizer & ⚖️ Profiler".
    *   Laden Sie die Originaldatei in "🔍 Original (...)".
    *   Laden Sie die Ergebnisdatei in "🔍 Bearbeitet (...)". Sie können dazu den Button " Lade letztes Ergebnis (Bearb.)" verwenden, um die zuletzt generierte Datei zu laden.
    *   Klicken Sie auf "📊 Visualisieren", um Wellenform/Spektrogramm zu sehen.
    *   Klicken Sie auf "🚀 Analysieren!", um den detaillierten Profiler-Bericht zu erhalten.
9.  **🛠️ Presets (Optional):**
    *   Wechseln Sie zum Tab "🛠 Preset-Editor (v4)".
    *   Geben Sie einen Namen ein und klicken Sie auf "💾 Speichern", um die aktuellen Einstellungen zu sichern.
    *   Wählen Sie ein gespeichertes Preset aus der Liste und klicken Sie auf "📥 Laden", um es anzuwenden.
    *   Verwalten Sie Presets mit "🔄 Liste neu laden" und "🗑️ Löschen".
    *   Exportieren Sie alle Presets mit "📦 ZIP Export".

---

## ⚙️ Parameter-Erklärungen (Auswahl)

*   **Raumgröße:** Beeinflusst indirekt Halldauer, Delay-Zeiten und Reflexionsdichte. Größere Werte simulieren größere Räume.
*   **Diffusion:** Steuert, wie "glatt" oder "körnig" der Nachhall klingt. Höhere Werte führen zu einem dichteren, weniger diskreten Hall.
*   **Luftabsorption:** Simuliert die natürliche Dämpfung hoher Frequenzen in der Luft über Distanz. Höhere Werte führen zu einem dunkleren Nachhall.
*   **Dry Kill Start:** Der Punkt im Dry/Wet-Mix (0-1), ab dem das *trockene* Signal beginnt, linear ausgeblendet zu werden. Bei 1.0 wird das Dry-Signal nie komplett ausgeblendet (nur durch den Dry/Wet-Regler skaliert). Bei 0.0 beginnt das Ausblenden sofort, wenn Wet > 0 ist.
*   **3D Position Z:** Beeinflusst die Balance zwischen vorderen und hinteren Kanälen beim Panning und die Intensität der generierten Höhenkanäle im 5.1.2-Modus.

---

## 📁 Presets

*   Presets werden als `.json`-Dateien im Unterordner `presets_v4` gespeichert.
*   Sie enthalten alle einstellbaren Parameter der Hauptverarbeitungsseite.
*   Die Anwendung speichert, welches Preset zuletzt geladen wurde (in `last_preset_v4.txt`) und lädt dieses beim nächsten Start automatisch.

---

## 🖼️ Surround Map Bild

*   Die Anwendung benötigt eine Bilddatei namens `surround_layout_3d.png` im selben Verzeichnis wie das Skript für die interaktive Positionierungskarte.
*   Wenn die Datei beim Start nicht gefunden wird, wird automatisch ein einfacher Platzhalter erstellt. Sie können diese Datei durch ein eigenes Bild (z.B. mit eingezeichneten Lautsprecherpositionen) ersetzen.

---

## ⚠️ Wichtige Hinweise

*   **FFmpeg:** Stellen Sie sicher, dass FFmpeg korrekt installiert und im Systempfad ist, wenn Sie andere Formate als WAV laden möchten.

---

## 📜 Lizenz



```
