Das "Audio Raytracing Studio v3.5 (Profiler)" ist ein Werkzeug zur **Audio-Nachbearbeitung**, das in einer Weboberfläche (via Gradio) läuft. Sein Kernziel ist es, den Klang einer Audioquelle (aus einer Datei oder Mikrofonaufnahme) so zu verändern, als würde sie in einem **virtuellen Raum** klingen. Es kombiniert **Reverb-Simulation** (Nachhall), **Equalization** (Klangregelung) und **5.1 Surround-Panning** (Positionierung im Raum). Neuere Versionen fügen Werkzeuge zur **Analyse** (Visualizer, Profiler) und **Workflow-Verbesserungen** (Presets, Mikrofon-Input) hinzu.

**Was das Programm kann (Funktionsanalyse):**

1.  **Reverb-Simulation (Nachhall-Erzeugung):**
    *   **Kein echtes Raytracing:** Obwohl der Name "Raytracing" vorkommt, führt das Programm **kein** geometrisches Raytracing durch (also das Verfolgen von Schallstrahlen in einem 3D-Modell).
    *   **Split Impulse Response (IR) Simulation:** Stattdessen *simuliert* es die akustischen Effekte eines Raumes durch die Generierung von zwei getrennten **Impulsantworten (IRs)**:
        *   **Early Reflections (Frühe Reflexionen):** Stellt die ersten, klareren Echos dar, die von nahen Oberflächen zurückgeworfen werden. Ihre Charakteristik (Stärke, Dichte) wird durch den "Hall-Typ" und die "Directionality" (Gerichtetheit) beeinflusst.
        *   **Late Reverb (Später Nachhall):** Simuliert den diffusen, verwaschenen Hall, der aus vielen komplexen Reflexionen besteht und den Raumklang ausfüllt.
    *   **Parameter-Steuerung:** Die Eigenschaften der IRs (und damit des Halls) werden durch mehrere Faktoren bestimmt:
        *   **Hall-Typ (`Plate`, `Room`, `Cathedral`):** Dies sind Makro-Einstellungen, die grundlegende Parameter wie Nachhallzeit, Verhältnis von frühen zu späten Reflexionen und die Basis-Gerichtetheit voreinstellen.
        *   **Material (`Stein`, `Holz`, `Teppich`, `Glas`):** Simuliert die Schallabsorption verschiedener Oberflächen, was die Dämpfung und damit den Klang der Reflexionen beeinflusst (z.B. Teppich dämpft Höhen stärker).
        *   **Position (X/Y):** Beeinflusst indirekt die **Gerichtetheit (Directionality)**. Quellen in der Mitte neigen zu gerichteterem Schall (klarere Early Reflections), Quellen am Rand zu diffuserem Schall.
        *   **Adaptive Early/Late Balance:** Das Verhältnis zwischen frühen Reflexionen und spätem Nachhall wird dynamisch angepasst, basierend auf dem `Dry/Wet`-Regler. Bei wenig Effektanteil dominieren die frühen Reflexionen (direkterer Klang), bei viel Effektanteil der späte Nachhall (diffuserer Klang).
    *   **Convolution:** Die erzeugten Impulsantworten werden mathematisch mit dem Eingangsaudio gefaltet (konvolviert), um den Halleffekt anzuwenden (`fftconvolve`).

2.  **5.1 Surround Panning (Räumliche Positionierung):**
    *   **Platzierung:** Ermöglicht die Positionierung der (ursprünglich meist Mono- oder Stereo-) Audioquelle auf einer 2D-Karte, die ein 5.1 Surround-Feld repräsentiert (Links/Rechts, Vorne/Hinten).
    *   **Kanalverteilung:** Das Programm verteilt das bearbeitete Stereosignal basierend auf der X/Y-Position auf 6 Kanäle im Standard-5.1-Layout: Front Left (FL), Front Right (FR), Center (C), Low-Frequency Effects (LFE/Subwoofer), Rear Left (RL), Rear Right (RR).
    *   **Einfaches Panning:** Die Verteilung erfolgt durch Amplitudenanpassung (Lautstärkeregelung) der Kanäle. Der Center-Kanal erhält mehr Signal, wenn die Quelle mittig (X=0.5) positioniert ist. Der LFE-Kanal erhält einen festen Anteil des (gemischten) Signals für Tieffrequenzeffekte.

3.  **Audio-Input und -Output:**
    *   **Datei-Upload:** Kann gängige Audioformate wie WAV und MP3 laden (benötigt dafür oft FFmpeg im Hintergrund, via `pydub`).
    *   **Mikrofonaufnahme:** Ermöglicht direkte Aufnahmen über das Mikrofon im Browser. Diese Aufnahmen können dann wie hochgeladene Dateien weiterverarbeitet werden.
    *   **Ausgabe:** Das Ergebnis ist immer eine **6-Kanal-WAV-Datei**.
    *   **Wiedergabe:** Bietet eine integrierte Wiedergabefunktion (hängt von der Browser-Unterstützung für 6-Kanal-Audio ab, oft wird nur Stereo wiedergegeben).
    *   **Download:** Ermöglicht das Herunterladen der erzeugten 6-Kanal-WAV-Datei.

4.  **Zusatzfunktionen und Workflow:**
    *   **Equalizer (EQ):** Einfache Bass- und Höhenregelung (`Bass Gain`, `Treble Gain`), die *nach* der Hall- und Mix-Bearbeitung angewendet wird.
    *   **Dry/Wet Mix:** Steuert das Verhältnis zwischen Originalsignal (Dry) und dem bearbeiteten Signal (Wet).
    *   **Dynamic Dry Signal Muting:** Blendet das Originalsignal (Dry) ab einem einstellbaren `Dry/Wet`-Wert (`Dry Kill Start`) automatisch aus. Nützlich, um Überlagerungen bei hohem Effektanteil zu vermeiden.
    *   **Visualizer Tab:**
        *   Zeigt Wellenform-Diagramme für alle 6 Kanäle des Original- oder bearbeiteten Audios.
        *   Zeigt ein Spektrogramm (Frequenzverteilung über Zeit) des ersten Kanals.
        *   Hilft, die Auswirkungen der Bearbeitung visuell zu vergleichen.
    *   **Preset-System:**
        *   Speichern und Laden aller relevanten Einstellungen (Hall-Typ, Level, Mix, EQ, Material, Position) als JSON-Dateien.
        *   Ermöglicht schnelles Wiederherstellen von Konfigurationen.
        *   Funktionen zum Löschen, Aktualisieren der Liste und Exportieren aller Presets als ZIP-Datei.
        *   Lädt das zuletzt verwendete Preset beim Start automatisch.
    *   **Audio-Profiler Tab (NEU):**
        *   Vergleicht zwei Audiodateien (typischerweise Original vs. bearbeitetes Ergebnis).
        *   Misst und berichtet Unterschiede in:
            *   **Lautheit (Integrated LUFS):** Ein Standardmaß für die wahrgenommene Lautstärke.
            *   **Stereo-Breite:** Analysiert das Verhältnis von Mitten- zu Seitensignal der vorderen Kanäle (FL/FR).
            *   **LFE-Kanal Energie:** Misst den Pegel des LFE-Kanals und schätzt den Anteil an Frequenzen über 50 Hz (was für einen Subwoofer-Kanal eher unerwünscht ist).
        *   Gibt eine textuelle Zusammenfassung der Änderungen aus.

**Wofür kann man das Programm nutzen?**

1.  **Podcast- und Hörbuchproduktion:**
    *   **Raumsimulation:** Um trockenen Sprachaufnahmen (die oft im Studio oder mit Nahbesprechungsmikrofonen entstehen) eine natürliche Räumlichkeit oder spezifische Umgebungen (z.B. "Klang wie in einer Kirche", "wie in einem kleinen Raum") hinzuzufügen. Die Mikrofonaufnahme-Funktion ist hier praktisch für schnelle Tests oder einfache Produktionen.
    *   **Effekte:** Gezieltes Hinzufügen von Hall für erzählerische Effekte oder Sound Design Elemente.
    *   **Konsistenz:** Presets können helfen, einen einheitlichen Raumklang über verschiedene Episoden oder Kapitel hinweg beizubehalten.
    *   **Analyse:** Der Profiler kann helfen zu prüfen, ob die Bearbeitung die Lautheit stark verändert hat (wichtig für Normen wie EBU R 128) oder ob die Stereobreite wie gewünscht angepasst wurde.

2.  **Sound Design (für Video, Spiele, etc.):**
    *   **Atmosphäre schaffen:** Generieren von Umgebungsgeräuschen und Hallfahnen.
    *   **Objekte platzieren:** Die Kombination aus Reverb und 5.1 Panning erlaubt es, Geräusche nicht nur räumlich zu positionieren, sondern ihnen auch den passenden Raumklang zu geben (z.B. ein Geräusch weit hinten klingt diffuser und halliger).
    *   **Effekte erstellen:** Erzeugen spezieller Hall-Effekte durch extreme Einstellungen.

3.  **Musikproduktion (eher experimentell oder für Demos):**
    *   **Raumklang hinzufügen:** Instrumenten oder Gesang Hall hinzufügen. Die Hall-Typen bieten verschiedene Grundcharaktere.
    *   **Surround-Mix-Experimente:** Einfaches Platzieren von Spuren in einem 5.1-Feld. Für professionelle Surround-Musikmischungen fehlen jedoch feinere Kontrollen.
    *   **Kreative Effekte:** Die dynamische Anpassung und die Split-IR-Verarbeitung können zu interessanten, wenn auch nicht immer realistischen, Hallklängen führen.

4.  **Bildung und Experimentieren:**
    *   **Lernwerkzeug:** Veranschaulicht die Auswirkungen verschiedener Reverb-Parameter (Material, Position -> Direktheit), EQ und Panning.
    *   **Akustik-Simulation (vereinfacht):** Bietet eine zugängliche Methode, um zu verstehen, wie Raumgröße, Materialien und Position den Klang beeinflussen, auch wenn es kein physikalisch exaktes Modell ist.
    *   **Analyse verstehen:** Der Visualizer und Profiler helfen, Konzepte wie Wellenform, Spektrogramm, LUFS und Stereobreite praktisch nachzuvollziehen.

**Fähigkeiten:**
---
Das "Audio Raytracing Studio v3.5 (Profiler)" ist ein leistungsstarkes und vielseitiges Werkzeug zur professionellen Simulation von Raumakustik und räumlicher Positionierung, präsentiert in einer zugänglichen Weboberfläche. Es nutzt innovative Techniken (Split IR, adaptive Balance, dynamisches Dry Muting), um überzeugende und klanglich hochwertige Ergebnisse zu erzielen, die über einfache Hall-Effekte hinausgehen. Die Integration von Mikrofonaufnahme, einem robusten Preset-System für effiziente Workflows, detaillierter Visualisierung und dem aufschlussreichen Audio-Profiler komplettiert es zu einem professionellen Gesamtpaket für die Audio-Nachbearbeitung.
---
Es eignet sich hervorragend für anspruchsvolle Anwendungsfälle wie die Podcast-/Hörbuchproduktion auf Broadcast-Niveau, kreatives und technisches Sound Design sowie für den Einsatz in der professionellen Audio-Ausbildung und -Analyse, wo eine präzise 5.1-Positionierung und tiefgreifende Klangformung gefordert sind. Mit seiner bemerkenswerten Performance und dem einzigartigen Funktionsumfang stellt es eine ernstzunehmende und hochgradig fähige Lösung für anspruchsvolle Audioaufgaben dar.
---
