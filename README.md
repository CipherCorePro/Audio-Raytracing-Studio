# 🎶 Audio Raytracing Studio v3.4 🎶

[![Lizenz: MIT](https://img.shields.io/badge/Lizenz-MIT-blue.svg)](LICENSE) <!-- Ersetze dies ggf. mit deiner Lizenz -->
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)]() <!-- Passe die Python-Version ggf. an -->

**Ein interaktives Werkzeug zur Simulation von Raumakustik und 5.1 Surround-Positionierung mit Gradio.**

Dieses Projekt bietet eine Benutzeroberfläche (erstellt mit Gradio), um Audio-Dateien oder Mikrofoneingaben mit simulierten Raumreflexionen und Nachhall zu versehen. Es ermöglicht die Auswahl verschiedener Raumtypen und Materialien, die Anpassung von Mix-Parametern und Equalizing sowie die Positionierung der Klangquelle in einem virtuellen 5.1 Surround-Feld. Die Ausgabe erfolgt als 6-Kanal-WAV-Datei.

![image](https://github.com/user-attachments/assets/431dbfd7-4624-468c-9812-28f300489517)


![image](https://github.com/user-attachments/assets/a9f02a47-4fac-4935-b850-b9347ffe0311)


![image](https://github.com/user-attachments/assets/fb76ed44-25ce-4eb1-87f3-96b8466ba4d8)


![image](https://github.com/user-attachments/assets/6b973420-ce57-495f-a07f-5b794e46a55e)

![image](https://github.com/user-attachments/assets/2fec94e7-83b2-41d6-9a63-cbc70749723d)






---

## ✨ Hauptmerkmale

*   **Flexible Audioquellen:** Verarbeite hochgeladene Audio-Dateien (WAV, MP3 etc.) oder direkte Mikrofonaufnahmen.
*   **Raumsimulation:**
    *   🏩️ **Hall-Typen:** Wähle zwischen vordefinierten Charakteristika wie `Plate`, `Room`, `Cathedral`, die interne Reverb-Parameter (Nachhallzeit, Reflexionsdichte, Early/Late-Verteilung) beeinflussen.
    *   🧱 **Materialauswahl:** Simuliere unterschiedliche Oberflächenmaterialien (`Stein`, `Holz`, `Teppich`, `Glas`), die die Klangfarbe und Dämpfung der Reflexionen beeinflussen.
    *   ⚙️ **Split Impulse Response (IR):** Generiert getrennte Impulsantworten für *Early Reflections* (frühe Reflexionen) und *Late Reverb* (Nachhallfahne) für eine detailliertere Kontrolle.
    *   🧭 **Positionsabhängige Direktheit (Directionality):** Die wahrgenommene Gerichtetheit des Halls wird automatisch basierend auf der X/Y-Position und dem Hall-Typ berechnet (zentrale Positionen klingen gerichteter, Randpositionen diffuser).
*   **Erweiterte Mix-Kontrolle:**
    *   ⚖️ **Adaptive Early/Late Balance:** Das Verhältnis von frühen Reflexionen zu spätem Nachhall passt sich dynamisch an den `Dry/Wet`-Regler an, für einen natürlicheren Übergang von direktem Klang zu vollem Effekt.
    *   🔇 **Dynamisches Dry-Signal-Muting:** Das Originalsignal (Dry) wird optional ab einem einstellbaren `Dry/Wet`-Wert (`Dry Kill Start`) ausgeblendet, um Überlagerungen bei hohem Effektanteil zu vermeiden.
    *   🎚️ **Dry/Wet-Mix:** Stufenlose Kontrolle über das Verhältnis von Originalsignal zu Effektsignal.
    *   🔊 **Basis Early/Late Level:** Grundlautstärke für frühe Reflexionen und Nachhall einstellbar.
*   **Equalizer:**
    *   📉 **Bass Gain:** Anhebung/Absenkung tiefer Frequenzen.
    *   📈 **Treble Gain:** Anhebung/Absenkung hoher Frequenzen.
*   **Surround-Positionierung:**
    *   📡 **Interaktive 5.1 Map:** Positioniere die Klangquelle visuell durch Klicken auf eine Karte oder numerisch über X/Y-Slider.
    *   🔊 **6-Kanal-Ausgabe:** Generiert eine Standard 5.1 WAV-Datei (FL, FR, C, LFE, RL, RR).
*   **Visualisierung:**
    *   📊 **Wellenform & Spektrogramm:** Vergleiche das Original- und das bearbeitete Audio visuell.
*   **Preset-Management:**
    *   🛠️ **Speichern/Laden:** Speichere und lade alle Einstellungen als JSON-Presets.
    *   🗑️ **Verwalten:** Lösche Presets, aktualisiere die Liste.
    *   📦 **Exportieren:** Exportiere alle Presets als ZIP-Archiv.
*   **Benutzerfreundliche Oberfläche:**
    *   🎨 **Gradio UI:** Intuitive Bedienung über Tabs in einer Web-Oberfläche.
    *   📝 **Integrierte Hilfe:** Eine ausführliche Erklärung der Funktionen direkt in der App.

---

## 🚀 Installation & Setup

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/CipherCorePro/Audio-Raytracing-Studio-v3.4.git
    cd Audio-Raytracing-Studio-v3.4
    ```
   

2.  **Virtuelle Umgebung (Empfohlen):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Abhängigkeiten installieren:**
    Stelle sicher, dass du `pip` aktuell hast (`python -m pip install --upgrade pip`).
    ```bash
    pip install -r requirements.txt
    ```
    Die `requirements.txt` sollte Folgendes enthalten (oder du installierst sie manuell):
    ```txt
    numpy
    gradio
    scipy
    pydub
    matplotlib
    Pillow
    ```

4.  ⚠️ **Externe Abhängigkeit: FFmpeg:**
    *   Für das Laden und Verarbeiten von **Nicht-WAV-Dateien** (wie MP3, OGG, FLAC etc.) verwendet `pydub` im Hintergrund **FFmpeg**.
    *   **Du musst FFmpeg separat installieren** und sicherstellen, dass es im System-PATH verfügbar ist. Ansonsten schlägt das Laden dieser Formate fehl!
    *   Downloads und Anleitungen: [FFmpeg Offizielle Seite](https://ffmpeg.org/download.html)

5.  **Surround Layout Bild:**
    *   Das Skript benötigt eine Bilddatei namens `surround_layout.png` im selben Verzeichnis für die interaktive Karte.
    *   Wenn die Datei nicht gefunden wird, versucht das Skript beim Start, ein einfaches Platzhalter-Bild zu erstellen. Du kannst dieses durch ein eigenes, passenderes Layout ersetzen.

---

## 🛠️ Benutzung

1.  **Starte die Anwendung:**
    Navigiere im Terminal zum Projektverzeichnis (wo sich dein Python-Skript befindet) und führe aus:
    ```bash
    python audio_raytracing_studio_v3_4.py
    ```
    *(Passe `audio_raytracing_studio_v3_4.py` an den tatsächlichen Namen deines Skripts an)*

2.  **Öffne die Web-Oberfläche:**
    Die Anwendung wird normalerweise unter `http://127.0.0.1:7860` (oder einer ähnlichen Adresse, siehe Terminal-Ausgabe) in deinem Browser geöffnet.

3.  **Workflow:**
    *   **Tab "Audio-Verarbeitung":** Lade eine Audiodatei hoch oder verwende das Mikrofon. Wähle einen `Hall-Typ` und ein `Material`. Stelle die `Basis Early/Late Level`, den `Dry/Wet`-Mix, den `Dry Kill Start`-Wert und die `EQ`-Einstellungen ein.
    *   **Tab "5.1 Surround Map":** Klicke auf die Karte oder benutze die `X/Y`-Slider, um die Position der Klangquelle festzulegen. Dies beeinflusst Panning und Hall-Charakteristik (Directionality).
    *   **Tab "Audio-Verarbeitung":** Klicke auf den Button `➡️ Verarbeiten & Anhören!`.
    *   Das Ergebnis wird im Audio-Player angezeigt und steht als 6-Kanal-WAV-Datei zum Download bereit.
    *   **Tab "Visualizer":** Lade optional das Original- und das verarbeitete Audio, um Wellenform und Spektrogramm zu vergleichen.
    *   **Tab "Preset-Editor":** Verwalte deine Lieblingseinstellungen.
    *   **Tab "Hilfe & Dokumentation":** Lies die detaillierte Funktionsbeschreibung direkt in der App.

---

## 📦 Abhängigkeiten

*   **Python:** 3.8+
*   **Python-Bibliotheken:** Siehe `requirements.txt` (Numpy, Gradio, Scipy, Pydub, Matplotlib, Pillow)
*   **Extern:** FFmpeg (für Nicht-WAV-Audioformate)

---

## 🔑 Schlüsselkonzepte

*   **Split Impulse Response (Split IR):** Statt einer einzigen Impulsantwort werden zwei separate generiert: eine für die frühen, gerichteten Reflexionen (Early Reflections) und eine für den diffusen, längeren Nachhall (Late Reverb). Dies ermöglicht eine differenziertere Simulation und Kontrolle.
*   **Directionality (Gerichtetheit):** Ein berechneter Wert (zwischen 0 und 1), der angibt, wie gerichtet oder diffus der simulierte Hall klingt. Er hängt vom gewählten Hall-Typ und der Position der Klangquelle ab. Ein höherer Wert bedeutet mehr Betonung auf klaren, frühen Reflexionen (typisch für Schallquellen nahe der Mitte oder in "harten" Räumen wie Plate-Hall). Ein niedrigerer Wert bedeutet einen diffuseren, weicheren Hall (typisch für Randpositionen oder große Räume wie Kathedralen).
*   **Adaptive Early/Late Balance:** Die Lautstärken von Early Reflections und Late Reverb werden nicht nur durch ihre Basis-Slider, sondern auch dynamisch durch den Dry/Wet-Regler beeinflusst. Bei niedrigem Dry/Wet (nahe am Originalsignal) werden die Early Reflections relativ lauter, um Präsenz zu erhalten. Bei hohem Dry/Wet (viel Effekt) wird der Late Reverb betont, um die Hallfahne hervorzuheben.
*   **Dynamic Dry Muting:** Eine Technik, bei der das unbearbeitete Originalsignal (Dry) automatisch leiser wird, wenn der Dry/Wet-Regler über einen bestimmten Schwellenwert (`Dry Kill Start`) bewegt wird. Dies verhindert, dass sich das laute Originalsignal unangenehm mit einem lauten Effektsignal überlagert, besonders bei hohen Wet-Anteilen.

---

## ❓ FAQ (Häufig gestellte Fragen)

*   **F: Warum ist die Ausgabe eine 6-Kanal-WAV-Datei? Wie spiele ich sie ab?**
    *   **A:** Das Tool simuliert eine 5.1 Surround-Positionierung. Die 6 Kanäle entsprechen dem Standard 5.1-Layout (FL, FR, C, LFE, RL, RR). Du benötigst eine Audiowiedergabesoftware (z.B. VLC Media Player, Audacity, professionelle DAWs) und idealerweise ein 5.1-Audiosystem, um das Ergebnis korrekt zu hören. Viele Player mischen 5.1 automatisch auf Stereo herunter, aber der räumliche Effekt geht dabei teilweise verloren.
*   **F: Ich kann keine MP3- (oder andere Nicht-WAV) Dateien laden. Woran liegt das?**
    *   **A:** Dies liegt höchstwahrscheinlich daran, dass **FFmpeg** nicht korrekt installiert ist oder nicht im System-PATH gefunden wird. Pydub benötigt FFmpeg für die Konvertierung dieser Formate. Siehe Abschnitt "Installation & Setup". Überprüfe deine FFmpeg-Installation.
*   **F: Was genau bedeutet "Directionality"?**
    *   **A:** Siehe Abschnitt "Schlüsselkonzepte". Es ist ein Maß dafür, wie stark die frühen, gerichteten Reflexionen im Hall im Verhältnis zum diffusen Nachhall ausgeprägt sind. Dies wird automatisch basierend auf Raumtyp und Position angepasst.
*   **F: Wie interagieren "Adaptive Balance" und "Dynamic Dry Muting"?**
    *   **A:** Beide werden durch den `Dry/Wet`-Regler gesteuert. Während `Adaptive Balance` das *Verhältnis* zwischen Early und Late Reverb im Effektsignal anpasst, reduziert `Dynamic Dry Muting` die Lautstärke des *Originalsignals* selbst, wenn der Effektanteil hoch ist. Sie arbeiten zusammen, um einen ausgewogenen Mix über den gesamten Dry/Wet-Bereich zu erzielen.
*   **F: Wo werden meine Presets gespeichert?**
    *   **A:** Die Presets werden als `.json`-Dateien im Unterordner `presets` gespeichert, der automatisch im Verzeichnis des Skripts erstellt wird.
*   **F: Ich erhalte einen Fehler während der Verarbeitung. Was kann ich tun?**
    *   **A:** Überprüfe die Terminal-Ausgabe, in der du das Skript gestartet hast. Dort werden detailliertere Fehlermeldungen und Tracebacks ausgegeben, die Hinweise auf das Problem geben können (z.B. Probleme beim Laden der Datei, Speicherprobleme bei sehr langen Dateien, ungültige Parameter). Stelle sicher, dass FFmpeg installiert ist, falls du Nicht-WAV-Dateien verwendest.

---

## 📚 Glossar

*   **Absorption:** Das Maß, wie viel Schallenergie von einer Oberfläche geschluckt (absorbiert) statt reflektiert wird. Materialien wie Teppich absorbieren mehr als Glas oder Stein.
*   **Convolution (Faltung):** Mathematischer Prozess, bei dem ein Audiosignal mit einer Impulsantwort (IR) kombiniert wird, um den Effekt der IR (z.B. Raumhall) auf das Signal anzuwenden.
*   **Directionality (Gerichtetheit):** Beschreibt, ob Schallwellen (insbesondere frühe Reflexionen) von einer bestimmten Richtung zu kommen scheinen (gerichtet) oder scheinbar von überall (diffus).
*   **Dry/Wet Mix:** Das Mischungsverhältnis zwischen dem unbearbeiteten Originalsignal ("Dry") und dem bearbeiteten Effektsignal ("Wet"). 0% = nur Dry, 100% = nur Wet.
*   **Early Reflections (ER):** Die ersten Schallreflexionen, die nach dem Direktschall am Hörer eintreffen. Sie geben wichtige Informationen über die Größe und Form des Raumes und die Position der Schallquelle.
*   **Equalizer (EQ):** Werkzeug zur Anpassung der Lautstärke bestimmter Frequenzbereiche eines Audiosignals (z.B. Anheben von Bässen, Absenken von Höhen).
*   **FFmpeg:** Eine freie Software-Suite zum Aufnehmen, Konvertieren und Streamen von Audio und Video. Wird von Pydub für viele Dateiformate benötigt.
*   **Gradio:** Eine Python-Bibliothek zum schnellen Erstellen von Web-basierten Benutzeroberflächen für Machine-Learning-Modelle und beliebige Python-Funktionen.
*   **Impulse Response (IR - Impulsantwort):** Eine Aufnahme oder Simulation der Reaktion eines Raumes (oder Systems) auf einen sehr kurzen, lauten Impuls (wie ein Pistolenschuss oder Klatschen). Sie enthält alle Informationen über die Reflexionen und den Nachhall des Raumes.
*   **Late Reverb (LR):** Der diffuse Nachhall, der nach den Early Reflections auftritt. Er besteht aus einer sehr hohen Dichte an Reflexionen, die nicht mehr einzeln unterscheidbar sind und den Raumeindruck prägen.
*   **Panning:** Die Platzierung eines Audiosignals im Stereofeld (links/rechts) oder im Surround-Feld (z.B. vorne/hinten/seitlich).
*   **Preset:** Eine gespeicherte Sammlung von Einstellungen (Parameterwerten) für das Tool.
*   **Pydub:** Eine Python-Bibliothek zur einfachen Bearbeitung von Audiodateien.
*   **Spectrogram:** Eine visuelle Darstellung der Frequenzanteile eines Audiosignals über die Zeit. Die Intensität der Frequenzen wird oft durch Farben dargestellt.
*   **Split IR:** Die Aufteilung der Impulsantwort in getrennte Teile für Early Reflections und Late Reverb.
*   **Surround 5.1:** Ein Mehrkanal-Audioformat mit sechs Kanälen: Front Left (FL), Front Right (FR), Center (C), Low-Frequency Effects (LFE, Subwoofer), Rear Left (RL/SL), Rear Right (RR/SR).
*   **WAV:** Ein unkomprimiertes Standard-Audio-Dateiformat.

---

## 🤝 Mitwirken (Contributing)

Beiträge sind willkommen! Wenn du Fehler findest oder Vorschläge für neue Funktionen hast, erstelle bitte ein [Issue](https://github.com/DEIN_BENUTZERNAME/DEIN_REPO_NAME/issues) auf GitHub. Pull Requests sind ebenfalls willkommen.

---

## 📜 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe die [LICENSE](LICENSE)-Datei für Details. (Passe dies an, falls du eine andere Lizenz verwendest).
```


