@echo off
title GlitchStemUltra - High Performance Setup
color 0b

echo ==================================================
echo       GLITCH STEM ULTRA - 3090 Ti SETUP
echo ==================================================
echo.

echo [1/6] Creating High-Performance Virtual Environment...
python -m venv venv_ultra

echo.
echo [2/6] Activating...
call venv_ultra\Scripts\activate

echo.
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/6] Installing PyTorch with CUDA 12.4 support...
echo       (This enables GPU acceleration for your 3090 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [5/6] Installing Audio Separation Engine...
echo       (audio-separator with GPU support)
pip install "audio-separator[gpu]" customtkinter onnxruntime-gpu

echo.
echo [6/6] Installing MIDI Extraction Tools...
echo       (Piano transcription + Drum transcription)
pip install piano_transcription_inference torchlibrosa mido librosa

echo.
echo ==================================================
echo       SETUP COMPLETE - READY FOR DEPLOYMENT
echo ==================================================
echo.
echo Installed components:
echo   - PyTorch with CUDA 12.4 (GPU acceleration)
echo   - audio-separator (stem separation)
echo   - piano_transcription_inference (melodic MIDI)
echo   - librosa + mido (drum MIDI)
echo   - customtkinter (GUI)
echo.
echo Run 'run_ultra.bat' to launch the application.
echo.
pause
