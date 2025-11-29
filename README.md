# Tex's Glitch Stem Ultra

AI-powered audio stem separation and MIDI extraction with GPU acceleration.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Stem Separation** - Extract vocals, drums, bass, and instruments using state-of-the-art AI models
- **MIDI Extraction** - Convert melodic stems to MIDI (piano, bass, synths) and transcribe drum patterns
- **GPU Accelerated** - Optimized for NVIDIA GPUs with CUDA support
- **God Mode** - Preset for high-VRAM cards (24GB+) with maximum quality settings
- **Ensemble Workflows** - Chain multiple models for best results
- **Custom Ensembles** - Build your own multi-model pipelines

## Models

29+ models including:
- **Vocals**: MelBand-Kim (SDR 12.6), BigBeta4, BS-Roformer-ViperX
- **Instrumental**: MelBand-Inst-V2 (SDR 16.1), Bleedless variants
- **Drums**: DrumSep-6way (kick/snare/toms/hh/cymbals), HTDemucs
- **Multi-stem**: HTDemucs-ft (4-stem), HTDemucs-6s (6-stem), BS-Roformer-SW (6-stem)
- **Utility**: De-reverb, Denoise, Bleed suppression, Vocal/Instrumental restoration

## Requirements

- Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA support
- CUDA 12.4 compatible drivers

## Installation

1. Clone the repository:
```batch
git clone https://github.com/Texmexdex/Glitch-Stem-Ultra.git
cd Glitch-Stem-Ultra
```

2. Run the setup script:
```batch
setup_ultra.bat
```

This creates a virtual environment and installs:
- PyTorch with CUDA 12.4
- audio-separator with GPU support
- piano_transcription_inference for melodic MIDI
- librosa + mido for drum transcription

## Usage

Launch the application:
```batch
run_ultra.bat
```

Or manually:
```batch
call venv_ultra\Scripts\activate
python GlitchStemUltra.py
```

### Quick Start

1. Click **SELECT AUDIO FILE** to choose your track
2. Select an **OUTPUT FOLDER** (optional - defaults to `./Stems_Output/`)
3. Choose a model or ensemble workflow from the dropdown
4. Enable **GOD MODE** for maximum quality (requires 24GB+ VRAM)
5. Click **INITIALIZE SEPARATION**

### MIDI Extraction

After separating stems:
1. Click **Select Stem File** in the MIDI section
2. Use **Extract Melodic MIDI** for piano/bass/synth stems
3. Use **Extract Drum MIDI** for drum stems

## Output

- Separated stems saved as WAV files (0.9 normalization)
- MIDI files saved alongside source stems
- Ensemble outputs organized in subfolders

## Credits

- [audio-separator](https://github.com/karaokenerds/python-audio-separator) - Separation engine
- [piano_transcription_inference](https://github.com/bytedance/piano_transcription) - Melodic MIDI
- Model creators: Kimberley Jensen, ViperX, Unwa, aufr33, Gabox, jarredou, Sucial

---

**TeXmExDeX Type Tunes**
