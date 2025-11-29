# Tex's Glitch Stem Ultra

AI-powered audio stem separation and MIDI extraction with GPU acceleration.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Stem Separation** - Extract vocals, drums, bass, and instruments using state-of-the-art AI models
- **MIDI Extraction** - Convert melodic stems to MIDI (piano, bass, synths) and transcribe drum patterns
- **GPU Accelerated** - Optimized for NVIDIA GPUs with CUDA support
- **Auto Hardware Detection** - Automatically detects your GPU and recommends optimal settings
- **Hardware Presets** - 6 presets from CPU-only to God Mode for any PC configuration
- **Ensemble Workflows** - Chain multiple models for best results
- **Custom Ensembles** - Build your own multi-model pipelines
- **Interactive Tooltips** - Hover over settings to see hardware impact and tips

## Models

29+ models including:
- **Vocals**: MelBand-Kim (SDR 12.6), BigBeta4, BS-Roformer-ViperX
- **Instrumental**: MelBand-Inst-V2 (SDR 16.1), Bleedless variants
- **Drums**: DrumSep-6way (kick/snare/toms/hh/cymbals), HTDemucs
- **Multi-stem**: HTDemucs-ft (4-stem), HTDemucs-6s (6-stem), BS-Roformer-SW (6-stem)
- **Utility**: De-reverb, Denoise, Bleed suppression, Vocal/Instrumental restoration

## Hardware Presets

The app auto-detects your GPU on startup and recommends the best preset:

| Preset | VRAM | GPU Examples | Settings |
|--------|------|--------------|----------|
| CPU Only | None | No NVIDIA GPU | Slowest, works anywhere |
| Laptop/Low | 2-4GB | GTX 1050, 1650, MX series | Safe, avoids crashes |
| Mid-Range | 6-8GB | GTX 1060-1080, RTX 2060-3060 | Balanced speed/quality |
| High-End | 10-12GB | RTX 3060 Ti/3070/3080, 4070 | Fast, high quality |
| Enthusiast | 16-24GB | RTX 3090, 4080/4090 | Very high quality |
| 🔥 God Mode | 24GB+ | RTX 3090 Ti, 4090, A6000 | Maximum quality |

Hover over parameter labels (Segment Size ⓘ, Overlap ⓘ, Batch Size ⓘ) to see hardware impact details.

## Requirements

- Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA support (optional - CPU mode available)
- CUDA 12.4 compatible drivers (for GPU acceleration)

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
3. Choose a **Hardware Preset** (auto-detected on startup, or select manually)
4. Choose a model or ensemble workflow from the dropdown
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
