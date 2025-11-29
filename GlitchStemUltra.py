import customtkinter as ctk
from tkinter import filedialog
import threading
import subprocess
import os
import sys

# Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Drum transcription using librosa (load first as it's used by both)
DRUMS_AVAILABLE = False
try:
    import librosa
    import numpy as np
    from mido import MidiFile, MidiTrack, Message
    DRUMS_AVAILABLE = True
except ImportError:
    pass

# MIDI extraction available flag
MIDI_AVAILABLE = False
PIANO_SAMPLE_RATE = 16000  # piano_transcription expects 16kHz
try:
    from piano_transcription_inference import PianoTranscription
    import torch
    MIDI_AVAILABLE = True
except ImportError:
    pass

# GM Drum Map (standard MIDI drum notes)
DRUM_MAP = {
    'kick': 36,      # Bass Drum 1
    'snare': 38,     # Acoustic Snare
    'hihat': 42,     # Closed Hi-Hat
    'hihat_open': 46, # Open Hi-Hat
    'tom_low': 45,   # Low Tom
    'tom_mid': 47,   # Mid Tom
    'tom_high': 50,  # High Tom
    'crash': 49,     # Crash Cymbal 1
    'ride': 51,      # Ride Cymbal 1
}

# Model database with descriptions and filenames
MODEL_DATABASE = {
    # === 🆕 CUTTING EDGE (2025) ===
    "BS-Roformer-SW-6stem": {
        "file": "BS-Roformer-SW.ckpt",
        "desc": "🆕 6-stem (vocals/bass/drums/guitar/piano/other) - jarredou",
        "category": "multi-stem"
    },
    "BS-Roformer-MaleFemale": {
        "file": "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
        "desc": "🆕 Split vocals into Male/Female (SDR 24.1) - Sucial",
        "category": "vocals"
    },
    "BS-Roformer-VocalsRevive-V3": {
        "file": "bs_roformer_vocals_revive_v3e_unwa.ckpt",
        "desc": "🆕 Enhance/restore degraded vocals - Unwa",
        "category": "utility"
    },
    "BS-Roformer-InstResurrection": {
        "file": "bs_roformer_instrumental_resurrection_unwa.ckpt",
        "desc": "🆕 Restore/enhance instrumentals - Unwa",
        "category": "utility"
    },
    # === VOCALS - TOP TIER ===
    "MelBand-Kim-Vocals": {
        "file": "vocals_mel_band_roformer.ckpt",
        "desc": "Best vocal extraction (SDR 12.6) - Kimberley Jensen",
        "category": "vocals"
    },
    "MelBand-BigBeta4": {
        "file": "melband_roformer_big_beta4.ckpt",
        "desc": "Top-tier vocals (SDR 12.5) - unwa fine-tune",
        "category": "vocals"
    },
    "MelBand-BigBeta5e": {
        "file": "melband_roformer_big_beta5e.ckpt",
        "desc": "Excellent vocals (SDR 12.4) - unwa",
        "category": "vocals"
    },
    "MelBand-Kim-FT-Unwa": {
        "file": "mel_band_roformer_kim_ft_unwa.ckpt",
        "desc": "Kim model fine-tuned (SDR 12.4) - unwa",
        "category": "vocals"
    },
    "MelBand-BigSYHFT-V1": {
        "file": "MelBandRoformerBigSYHFTV1.ckpt",
        "desc": "Big SYHFT vocals (SDR 12.3) - SYH99999",
        "category": "vocals"
    },
    "BS-Roformer-ViperX-1296": {
        "file": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "desc": "BS-Roformer vocals (SDR 12.1) - ViperX",
        "category": "vocals"
    },
    "BS-Roformer-ViperX-1297": {
        "file": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "desc": "Classic BS-Roformer (SDR 11.8) - ViperX",
        "category": "vocals"
    },
    # === INSTRUMENTAL - TOP TIER ===
    "MelBand-Inst-V2": {
        "file": "melband_roformer_inst_v2.ckpt",
        "desc": "Best instrumental (SDR 16.1) - Unwa",
        "category": "instrumental"
    },
    "MelBand-InstVoc-Duality-V2": {
        "file": "melband_roformer_instvox_duality_v2.ckpt",
        "desc": "Balanced vocal/inst (SDR 16.1/11.0) - Unwa",
        "category": "instrumental"
    },
    "MelBand-Inst-Bleedless-V3": {
        "file": "mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
        "desc": "Minimal vocal bleed instrumental - Gabox",
        "category": "instrumental"
    },
    # === DRUMS - SEPARATION ===
    "DrumSep-6way": {
        "file": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
        "desc": "🥁 Splits drums: kick/snare/toms/hh/ride/crash - aufr33",
        "category": "drums"
    },
    "BS-Roformer-DrumBass": {
        "file": "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
        "desc": "Isolate drum+bass from mix (SDR 10.5) - ViperX",
        "category": "drums"
    },
    "Kuielab-Drums-A": {
        "file": "kuielab_a_drums.onnx",
        "desc": "Extract drums from mix (SDR 7.0) - Kuielab",
        "category": "drums"
    },
    "Kuielab-Drums-B": {
        "file": "kuielab_b_drums.onnx",
        "desc": "Extract drums from mix (SDR 7.1) - Kuielab",
        "category": "drums"
    },
    # === MULTI-STEM ===
    "HTDemucs-ft": {
        "file": "htdemucs_ft.yaml",
        "desc": "4-stem (vocals/drums/bass/other) - Meta",
        "category": "multi-stem"
    },
    "HTDemucs-6s": {
        "file": "htdemucs_6s.yaml",
        "desc": "6-stem (+guitar/piano) - Meta",
        "category": "multi-stem"
    },
    # === UTILITY / POST-PROCESSING ===
    # De-Reverb models
    "DeReverb-MelBand-Anvuew": {
        "file": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        "desc": "Remove reverb (SDR 19.2) - anvuew - BEST",
        "category": "utility"
    },
    "DeReverb-SuperBig": {
        "file": "dereverb_super_big_mbr_ep_346.ckpt",
        "desc": "Heavy de-reverb (large model) - Sucial",
        "category": "utility"
    },
    "DeReverb-Echo-V2": {
        "file": "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
        "desc": "Remove reverb + echo combined - Sucial",
        "category": "utility"
    },
    "DeReverb-BS-Roformer": {
        "file": "deverb_bs_roformer_8_384dim_10depth.ckpt",
        "desc": "BS-Roformer de-reverb variant",
        "category": "utility"
    },
    # De-Noise models
    "Denoise-MelBand": {
        "file": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        "desc": "Noise removal (SDR 28.0) - aufr33 - BEST",
        "category": "utility"
    },
    "Denoise-Aggressive": {
        "file": "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
        "desc": "Aggressive noise removal - aufr33",
        "category": "utility"
    },
    "Denoise-Debleed": {
        "file": "mel_band_roformer_denoise_debleed_gabox.ckpt",
        "desc": "Denoise + remove stem bleed - Gabox",
        "category": "utility"
    },
    # Bleed suppression (removes vocal remnants from instrumentals)
    "Bleed-Suppressor": {
        "file": "mel_band_roformer_bleed_suppressor_v1.ckpt",
        "desc": "Remove vocal bleed from instrumentals - unwa",
        "category": "utility"
    },
    # Instrumental enhancement
    "Inst-Fullness-V3": {
        "file": "mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
        "desc": "Enhance instrumental fullness/body - Gabox",
        "category": "utility"
    },
}

# Ensemble presets - run multiple models and average results
ENSEMBLE_PRESETS = {
    "ENSEMBLE: 🥁 Drum Isolation + Split": {
        "desc": "Extract drums from mix → split into kick/snare/hh/toms/cymbals",
        "models": ["HTDemucs-ft"],
        "post_process": "DrumSep-6way",
        "output_stem": "drums"
    },
    "ENSEMBLE: 🎸 Clean Instrumental": {
        "desc": "Best instrumental + de-reverb + bleed removal",
        "models": ["MelBand-Inst-V2"],
        "post_process": "Bleed-Suppressor",
        "output_stem": "instrumental"
    },
    "ENSEMBLE: 🎸 Studio Instrumental": {
        "desc": "Instrumental → denoise → de-reverb (cleanest output)",
        "models": ["MelBand-Inst-Bleedless-V3"],
        "post_process": "Denoise-MelBand",
        "output_stem": "instrumental"
    },
    "ENSEMBLE: Ultimate Instrumental": {
        "desc": "Two top models for comparison/averaging",
        "models": ["MelBand-Inst-V2", "MelBand-Inst-Bleedless-V3"],
        "post_process": None,
        "output_stem": "instrumental"
    },
    "ENSEMBLE: 🎛️ Full Mix Breakdown": {
        "desc": "HTDemucs 4-stem (drums/bass/vocals/other) + denoise",
        "models": ["HTDemucs-ft"],
        "post_process": "Denoise-MelBand",
        "output_stem": "instrumental"
    },
    "ENSEMBLE: Studio Master": {
        "desc": "Vocals + instrumental separation with de-reverb",
        "models": ["MelBand-Kim-Vocals", "MelBand-Inst-V2"],
        "post_process": "DeReverb-MelBand-Anvuew",
        "output_stem": "vocals"
    },
    "ENSEMBLE: Ultimate Vocals": {
        "desc": "Best vocal quality - 2 top models + de-reverb",
        "models": ["MelBand-Kim-Vocals", "MelBand-BigBeta4"],
        "post_process": "DeReverb-MelBand-Anvuew",
        "output_stem": "vocals"
    },
    "⚙️ CUSTOM ENSEMBLE...": {
        "desc": "Build your own workflow - click to configure",
        "models": [],
        "post_process": None,
        "output_stem": "custom",
        "is_custom": True
    },
}

class GlitchStemUltraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tex's Glitch Stem Ultra // 3090 Ti Edition")
        self.geometry("900x1000")
        self.resizable(True, True)
        self.minsize(850, 950)

        # Data
        self.input_file = ""
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Stems_Output"))
        self.separator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv_ultra", "Scripts", "audio-separator"))

        # Layout
        self.grid_columnconfigure(0, weight=1)

        # 1. Title
        self.header = ctk.CTkLabel(self, text="Tex's Glitch Stem Ultra", font=("Roboto Medium", 24, "bold"), text_color="#00e5ff")
        self.header.grid(row=0, column=0, pady=(15, 5))
        
        self.sub_header = ctk.CTkLabel(self, text="Hardware Accelerated Separation Engine", font=("Roboto", 12))
        self.sub_header.grid(row=1, column=0, pady=(0, 10), sticky="n")

        # 2. File Selection Frame
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_select = ctk.CTkButton(self.file_frame, text="SELECT AUDIO FILE", command=self.select_file, fg_color="#222", border_color="#00e5ff", border_width=1)
        self.btn_select.pack(side="left", padx=20, pady=15)
        
        self.lbl_filename = ctk.CTkLabel(self.file_frame, text="No file selected", font=("Consolas", 11))
        self.lbl_filename.pack(side="left", padx=10)

        # 2.5 Output Directory Selection Frame
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_output_dir = ctk.CTkButton(self.output_frame, text="OUTPUT FOLDER", command=self.select_output_dir, 
                                            fg_color="#222", border_color="#4CAF50", border_width=1, width=140)
        self.btn_output_dir.pack(side="left", padx=20, pady=10)
        
        self.lbl_output_dir = ctk.CTkLabel(self.output_frame, text=self.output_dir, font=("Consolas", 10), text_color="#4CAF50")
        self.lbl_output_dir.pack(side="left", padx=10, fill="x", expand=True)

        # 3. Settings Frame
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_columnconfigure(1, weight=1)

        # --- Model Selection with Refresh ---
        model_header_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        model_header_frame.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        ctk.CTkLabel(model_header_frame, text="MODEL SELECTION", font=("Roboto", 14, "bold")).pack(side="left", padx=10)
        
        self.btn_refresh = ctk.CTkButton(model_header_frame, text="🔄 Check Updates", width=120, height=25,
                                         command=self.refresh_models, fg_color="#333", hover_color="#444",
                                         font=("Roboto", 11))
        self.btn_refresh.pack(side="left", padx=10)

        # Build model list with descriptions
        self.model_list = self.build_model_list()
        self.model_var = ctk.StringVar(value=self.model_list[0])
        self.model_combo = ctk.CTkComboBox(self.settings_frame, variable=self.model_var, 
                                           values=self.model_list, width=400, command=self.on_model_change)
        self.model_combo.grid(row=1, column=0, columnspan=2, pady=5)

        # Model description label
        self.model_desc = ctk.CTkLabel(self.settings_frame, text="", font=("Roboto", 11), text_color="#888")
        self.model_desc.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        self.on_model_change(self.model_var.get())  # Set initial description

        # --- Parameters ---
        ctk.CTkLabel(self.settings_frame, text="INFERENCE PARAMETERS", font=("Roboto", 14, "bold")).grid(row=3, column=0, columnspan=2, pady=(10, 5))

        # Segment Size
        ctk.CTkLabel(self.settings_frame, text="Segment Size (Context)").grid(row=4, column=0, sticky="e", padx=10)
        self.seg_size = ctk.CTkSlider(self.settings_frame, from_=256, to=4096, number_of_steps=100, command=self.update_seg_label)
        self.seg_size.set(256)
        self.seg_size.grid(row=4, column=1, sticky="w", padx=10)
        self.seg_size_label = ctk.CTkLabel(self.settings_frame, text="256", font=("Consolas", 12), width=50)
        self.seg_size_label.grid(row=4, column=1, sticky="e", padx=(0, 30))
        
        # Overlap
        ctk.CTkLabel(self.settings_frame, text="Overlap (Smoothness)").grid(row=5, column=0, sticky="e", padx=10)
        self.overlap = ctk.CTkSlider(self.settings_frame, from_=2, to=50, number_of_steps=48, command=self.update_overlap_label)
        self.overlap.set(8)
        self.overlap.grid(row=5, column=1, sticky="w", padx=10)
        self.overlap_label = ctk.CTkLabel(self.settings_frame, text="8", font=("Consolas", 12), width=50)
        self.overlap_label.grid(row=5, column=1, sticky="e", padx=(0, 30))

        # Batch Size
        ctk.CTkLabel(self.settings_frame, text="Batch Size (VRAM)").grid(row=6, column=0, sticky="e", padx=10)
        self.batch_size = ctk.CTkSlider(self.settings_frame, from_=1, to=16, number_of_steps=15, command=self.update_batch_label)
        self.batch_size.set(1)
        self.batch_size.grid(row=6, column=1, sticky="w", padx=10)
        self.batch_size_label = ctk.CTkLabel(self.settings_frame, text="1", font=("Consolas", 12), width=50)
        self.batch_size_label.grid(row=6, column=1, sticky="e", padx=(0, 30))

        # --- Presets ---
        self.chk_god_mode = ctk.CTkCheckBox(self.settings_frame, text="ACTIVATE GOD MODE (3090 Ti Preset)", 
                                            command=self.toggle_god_mode, onvalue=True, offvalue=False,
                                            font=("Roboto", 12, "bold"), text_color="#00e5ff")
        self.chk_god_mode.grid(row=7, column=0, columnspan=2, pady=(15, 10))
        
        # 4. Console Output
        self.console = ctk.CTkTextbox(self, height=180, font=("Consolas", 10), text_color="#bbb")
        self.console.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.console.insert("0.0", "Tex's Glitch Stem Ultra initialized.\nGPU acceleration enabled via PyTorch CUDA.\n")

        # 5. Run Button
        self.btn_run = ctk.CTkButton(self, text="INITIALIZE SEPARATION", command=self.run_separation, 
                                     height=50, fg_color="#00e5ff", text_color="black", font=("Roboto", 16, "bold"))
        self.btn_run.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # 6. MIDI Extraction Section
        self.midi_frame = ctk.CTkFrame(self)
        self.midi_frame.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        self.midi_frame.grid_columnconfigure(0, weight=1)
        self.midi_frame.grid_columnconfigure(1, weight=1)
        
        midi_header = ctk.CTkLabel(self.midi_frame, text="🎹 MIDI EXTRACTION", font=("Roboto", 14, "bold"), text_color="#ff9500")
        midi_header.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        
        # File selection row
        midi_file_frame = ctk.CTkFrame(self.midi_frame, fg_color="transparent")
        midi_file_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.btn_midi_select = ctk.CTkButton(midi_file_frame, text="Select Stem File", width=140,
                                              command=self.select_midi_input, fg_color="#333", hover_color="#444")
        self.btn_midi_select.pack(side="left", padx=5)
        
        self.midi_file_label = ctk.CTkLabel(midi_file_frame, text="No stem selected", font=("Consolas", 10), text_color="#666")
        self.midi_file_label.pack(side="left", padx=10)
        
        # Piano/Melodic extraction (left)
        piano_frame = ctk.CTkFrame(self.midi_frame)
        piano_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(piano_frame, text="🎹 Piano/Melodic", font=("Roboto", 12, "bold")).pack(pady=(10, 5))
        ctk.CTkLabel(piano_frame, text="Best for: piano, bass, vocals, synths", font=("Roboto", 9), text_color="#888").pack()
        
        self.btn_midi_extract = ctk.CTkButton(piano_frame, text="Extract Melodic MIDI", width=160,
                                               command=self.run_midi_extraction, fg_color="#ff9500", 
                                               text_color="black", hover_color="#ffaa33")
        self.btn_midi_extract.pack(pady=10)
        
        # Drum extraction (right)
        drum_frame = ctk.CTkFrame(self.midi_frame)
        drum_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(drum_frame, text="🥁 Drums", font=("Roboto", 12, "bold")).pack(pady=(10, 5))
        ctk.CTkLabel(drum_frame, text="Kick, snare, hi-hat detection", font=("Roboto", 9), text_color="#888").pack()
        
        self.btn_drum_extract = ctk.CTkButton(drum_frame, text="Extract Drum MIDI", width=160,
                                               command=self.run_drum_extraction, fg_color="#e91e63", 
                                               text_color="white", hover_color="#f44336")
        self.btn_drum_extract.pack(pady=10)
        
        self.midi_input_file = ""
        
        if not MIDI_AVAILABLE:
            self.btn_midi_extract.configure(state="disabled")
        if not DRUMS_AVAILABLE:
            self.btn_drum_extract.configure(state="disabled")

        # 7. Footer
        self.footer = ctk.CTkLabel(self, text="TeXmExDeX Type Tunes", font=("Roboto", 11), text_color="#666")
        self.footer.grid(row=8, column=0, pady=(10, 15), sticky="s")

        # Initialize God Mode
        self.chk_god_mode.select()
        self.toggle_god_mode()

    def build_model_list(self):
        """Build display list with categories - BEST/NEWEST FIRST"""
        models = []
        
        # === ENSEMBLES (workflows) ===
        models.append("═══ 🔥 ENSEMBLE WORKFLOWS ═══")
        for name in ENSEMBLE_PRESETS.keys():
            models.append(name)
        
        # === MULTI-STEM (most useful for full separation) ===
        models.append("═══ 🎛️ MULTI-STEM (Full Track) ═══")
        # Ordered best first
        multistem_order = ["BS-Roformer-SW-6stem", "HTDemucs-ft", "HTDemucs-6s"]
        for name in multistem_order:
            if name in MODEL_DATABASE:
                models.append(name)
        
        # === DRUMS ===
        models.append("═══ 🥁 DRUMS ═══")
        drums_order = ["DrumSep-6way", "BS-Roformer-DrumBass", "Kuielab-Drums-A", "Kuielab-Drums-B"]
        for name in drums_order:
            if name in MODEL_DATABASE:
                models.append(name)
        
        # === INSTRUMENTAL ===
        models.append("═══ 🎸 INSTRUMENTAL ═══")
        inst_order = ["MelBand-Inst-V2", "MelBand-Inst-Bleedless-V3", "MelBand-InstVoc-Duality-V2"]
        for name in inst_order:
            if name in MODEL_DATABASE:
                models.append(name)
        
        # === VOCALS ===
        models.append("═══ 🎤 VOCALS ═══")
        vocals_order = ["BS-Roformer-MaleFemale", "MelBand-Kim-Vocals", "MelBand-BigBeta4", 
                       "MelBand-BigBeta5e", "MelBand-Kim-FT-Unwa", "MelBand-BigSYHFT-V1",
                       "BS-Roformer-ViperX-1296", "BS-Roformer-ViperX-1297"]
        for name in vocals_order:
            if name in MODEL_DATABASE:
                models.append(name)
        
        # === UTILITY (cleaning/enhancement) ===
        models.append("═══ 🧹 UTILITY (Clean/Enhance) ═══")
        utility_order = ["BS-Roformer-VocalsRevive-V3", "BS-Roformer-InstResurrection",
                        "DeReverb-MelBand-Anvuew", "DeReverb-SuperBig", "DeReverb-Echo-V2",
                        "DeReverb-BS-Roformer", "Denoise-MelBand", "Denoise-Aggressive",
                        "Denoise-Debleed", "Bleed-Suppressor", "Inst-Fullness-V3"]
        for name in utility_order:
            if name in MODEL_DATABASE:
                models.append(name)
        
        return models

    def on_model_change(self, selection):
        """Update description when model changes"""
        if selection.startswith("───"):
            return  # Category header
        if selection in ENSEMBLE_PRESETS:
            self.model_desc.configure(text=ENSEMBLE_PRESETS[selection]["desc"])
        elif selection in MODEL_DATABASE:
            self.model_desc.configure(text=MODEL_DATABASE[selection]["desc"])
        else:
            self.model_desc.configure(text="")

    def refresh_models(self):
        """Check for new models from audio-separator"""
        self.log("\n>> Checking for model updates...")
        self.btn_refresh.configure(state="disabled", text="Checking...")
        thread = threading.Thread(target=self._refresh_models_thread)
        thread.start()

    def _refresh_models_thread(self):
        try:
            cmd = [self.separator_path, "--list_models", "--list_limit", "200"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            output, _ = process.communicate(timeout=30)
            
            # Count models
            model_count = output.count(".ckpt") + output.count(".onnx") + output.count(".yaml")
            self.log(f">> Found {model_count} models available in audio-separator")
            self.log(">> Tip: Run 'pip install --upgrade audio-separator[gpu]' for newest models")
            
        except Exception as e:
            self.log(f">> Error checking models: {str(e)}")
        
        self.btn_refresh.configure(state="normal", text="🔄 Check Updates")

    def select_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg")])
        if filename:
            self.input_file = filename
            self.lbl_filename.configure(text=os.path.basename(filename))
            self.log(f"Selected: {filename}")

    def select_output_dir(self):
        """Let user choose output directory for processed files"""
        directory = filedialog.askdirectory(initialdir=self.output_dir, title="Select Output Folder")
        if directory:
            self.output_dir = os.path.abspath(directory)
            # Truncate display if path is too long
            display_path = self.output_dir
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            self.lbl_output_dir.configure(text=display_path)
            self.log(f"Output folder: {self.output_dir}")

    def update_seg_label(self, value):
        self.seg_size_label.configure(text=str(int(value)))

    def update_overlap_label(self, value):
        self.overlap_label.configure(text=str(int(value)))

    def update_batch_label(self, value):
        self.batch_size_label.configure(text=str(int(value)))

    def toggle_god_mode(self):
        if self.chk_god_mode.get():
            self.seg_size.set(2048)
            self.overlap.set(12)
            self.batch_size.set(8)
            self.update_seg_label(2048)
            self.update_overlap_label(12)
            self.update_batch_label(8)
            self.log(">> GOD MODE ACTIVE: Max settings for 3090 Ti 24GB")
        else:
            self.seg_size.set(256)
            self.overlap.set(8)
            self.batch_size.set(1)
            self.update_seg_label(256)
            self.update_overlap_label(8)
            self.update_batch_label(1)
            self.log(">> Standard settings loaded.")

    def log(self, message):
        self.console.insert("end", message + "\n")
        self.console.see("end")
        self.update_idletasks()

    def run_separation(self):
        if not self.input_file:
            self.log("ERROR: No file selected.")
            return

        selection = self.model_var.get()
        if selection.startswith("═══"):
            self.log("ERROR: Please select a model, not a category header.")
            return

        # Check for custom ensemble
        if selection == "⚙️ CUSTOM ENSEMBLE...":
            self.open_custom_ensemble_dialog()
            return

        self.btn_run.configure(state="disabled", text="PROCESSING...")
        
        if selection in ENSEMBLE_PRESETS:
            thread = threading.Thread(target=self.process_ensemble, args=(selection,))
        else:
            thread = threading.Thread(target=self.process_single, args=(selection,))
        thread.start()

    def open_custom_ensemble_dialog(self):
        """Open dialog to build custom ensemble workflow"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Custom Ensemble Builder")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()
        
        # Get list of actual models (not headers)
        available_models = [name for name in MODEL_DATABASE.keys()]
        
        ctk.CTkLabel(dialog, text="⚙️ BUILD CUSTOM WORKFLOW", font=("Roboto", 16, "bold")).pack(pady=15)
        ctk.CTkLabel(dialog, text="Select models to run in sequence:", font=("Roboto", 11)).pack()
        
        # Model selection frame
        model_frame = ctk.CTkFrame(dialog)
        model_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        ctk.CTkLabel(model_frame, text="Step 1: Primary Model", font=("Roboto", 12, "bold")).pack(pady=(10,5))
        self.custom_model1 = ctk.CTkComboBox(model_frame, values=available_models, width=350)
        self.custom_model1.set(available_models[0])
        self.custom_model1.pack(pady=5)
        
        ctk.CTkLabel(model_frame, text="Step 2: Secondary Model (optional)", font=("Roboto", 12, "bold")).pack(pady=(15,5))
        self.custom_model2 = ctk.CTkComboBox(model_frame, values=["(None)"] + available_models, width=350)
        self.custom_model2.set("(None)")
        self.custom_model2.pack(pady=5)
        
        ctk.CTkLabel(model_frame, text="Step 3: Post-Processing (optional)", font=("Roboto", 12, "bold")).pack(pady=(15,5))
        utility_models = ["(None)"] + [name for name, data in MODEL_DATABASE.items() if data["category"] == "utility"]
        self.custom_post = ctk.CTkComboBox(model_frame, values=utility_models, width=350)
        self.custom_post.set("(None)")
        self.custom_post.pack(pady=5)
        
        ctk.CTkLabel(model_frame, text="Apply post-processing to:", font=("Roboto", 11)).pack(pady=(15,5))
        self.custom_stem = ctk.CTkComboBox(model_frame, values=["vocals", "instrumental", "drums", "other"], width=200)
        self.custom_stem.set("vocals")
        self.custom_stem.pack(pady=5)
        
        # Run button
        ctk.CTkButton(dialog, text="RUN CUSTOM WORKFLOW", height=40, fg_color="#00e5ff", text_color="black",
                     font=("Roboto", 14, "bold"), command=lambda: self.run_custom_ensemble(dialog)).pack(pady=20)

    def run_custom_ensemble(self, dialog):
        """Execute custom ensemble workflow"""
        models = [self.custom_model1.get()]
        if self.custom_model2.get() != "(None)":
            models.append(self.custom_model2.get())
        
        post_process = self.custom_post.get() if self.custom_post.get() != "(None)" else None
        output_stem = self.custom_stem.get()
        
        dialog.destroy()
        
        # Store custom config
        self.custom_ensemble_config = {
            "models": models,
            "post_process": post_process,
            "output_stem": output_stem
        }
        
        self.log(f"\n>> CUSTOM WORKFLOW:")
        self.log(f"   Models: {', '.join(models)}")
        if post_process:
            self.log(f"   Post-process: {post_process} → {output_stem}")
        
        self.btn_run.configure(state="disabled", text="PROCESSING...")
        thread = threading.Thread(target=self.process_custom_ensemble)
        thread.start()

    def process_custom_ensemble(self):
        """Process custom ensemble workflow"""
        config = self.custom_ensemble_config
        models = config["models"]
        post_process = config.get("post_process")
        output_stem = config.get("output_stem", "vocals")
        
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        ensemble_dir = os.path.join(self.output_dir, f"{base_name}_custom")
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)
        
        self.log(f"\n{'='*50}")
        self.log(f"CUSTOM ENSEMBLE")
        self.log(f"{'='*50}\n")
        
        all_success = True
        for i, model_name in enumerate(models, 1):
            self.log(f"\n[{i}/{len(models)}] Processing with {model_name}...")
            model_output_dir = os.path.join(ensemble_dir, f"pass_{i}_{model_name}")
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            
            success = self.run_model(model_name, self.input_file, model_output_dir)
            if not success:
                all_success = False
                self.log(f">> WARNING: {model_name} failed")
        
        # Post-processing
        if post_process and all_success:
            self.log(f"\n[POST] Applying {post_process}...")
            pass1_dir = os.path.join(ensemble_dir, f"pass_1_{models[0]}")
            post_dir = os.path.join(ensemble_dir, "post_processed")
            if not os.path.exists(post_dir):
                os.makedirs(post_dir)
            
            for f in os.listdir(pass1_dir):
                if not f.endswith(".wav"):
                    continue
                if output_stem.lower() in f.lower():
                    target_file = os.path.join(pass1_dir, f)
                    self.log(f">> Processing: {os.path.basename(target_file)}")
                    self.run_model(post_process, target_file, post_dir)
                    break
        
        self.log(f"\n{'='*50}")
        self.log(f">> CUSTOM ENSEMBLE COMPLETE")
        self.log(f">> Output: {ensemble_dir}")
        self.log(f"{'='*50}")
        
        self.btn_run.configure(state="normal", text="INITIALIZE SEPARATION")

    def run_model(self, model_name, input_file, output_dir, suffix=""):
        """Run a single model and return output files"""
        if model_name not in MODEL_DATABASE:
            self.log(f"ERROR: Unknown model {model_name}")
            return None
            
        model_data = MODEL_DATABASE[model_name]
        model_filename = model_data["file"]
        
        cmd = [
            self.separator_path,
            input_file,
            "--model_filename", model_filename,
            "--output_dir", output_dir,
            "--output_format", "wav",
            "--normalization", "0.9",
            "--use_autocast"
        ]
        
        # Add architecture-specific params
        if "htdemucs" in model_filename:
            cmd.extend(["--demucs_shifts", "4", "--demucs_overlap", "0.25"])
        else:
            cmd.extend([
                "--mdxc_segment_size", str(int(self.seg_size.get())),
                "--mdxc_overlap", str(int(self.overlap.get())),
                "--mdxc_batch_size", str(int(self.batch_size.get()))
            ])
        
        self.log(f"Running: {model_name}")
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                       text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.log(line)
            process.wait()
            return process.returncode == 0
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            return False

    def process_single(self, model_name):
        """Process with a single model"""
        output_path = self.output_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        self.log(f"\n{'='*50}")
        self.log(f"Starting separation: {model_name}")
        self.log(f"{'='*50}")
        
        success = self.run_model(model_name, self.input_file, output_path)
        
        if success:
            self.log(f"\n>> SEPARATION COMPLETE")
            self.log(f">> Output: {output_path}")
        else:
            self.log("\n>> ERROR: Separation failed")
        
        self.btn_run.configure(state="normal", text="INITIALIZE SEPARATION")

    def process_ensemble(self, preset_name):
        """Process with ensemble (multiple models)"""
        preset = ENSEMBLE_PRESETS[preset_name]
        models = preset["models"]
        post_process = preset.get("post_process")
        
        # Create temp directory for ensemble processing
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        ensemble_dir = os.path.join(self.output_dir, f"{base_name}_ensemble")
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)
        
        self.log(f"\n{'='*50}")
        self.log(f"ENSEMBLE MODE: {preset_name}")
        self.log(f"Models: {', '.join(models)}")
        if post_process:
            self.log(f"Post-processing: {post_process}")
        self.log(f"{'='*50}\n")
        
        # Run each model
        all_success = True
        for i, model_name in enumerate(models, 1):
            self.log(f"\n[{i}/{len(models)}] Processing with {model_name}...")
            model_output_dir = os.path.join(ensemble_dir, f"pass_{i}_{model_name}")
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            
            success = self.run_model(model_name, self.input_file, model_output_dir)
            if not success:
                all_success = False
                self.log(f">> WARNING: {model_name} failed, continuing...")
        
        # Post-processing pass (de-reverb, denoise, drum split, etc.)
        if post_process and all_success:
            self.log(f"\n[POST] Applying {post_process}...")
            pass1_dir = os.path.join(ensemble_dir, f"pass_1_{models[0]}")
            post_dir = os.path.join(ensemble_dir, "post_processed")
            if not os.path.exists(post_dir):
                os.makedirs(post_dir)
            
            output_stem = preset.get("output_stem", "vocals")
            
            # Find the right stem file(s) based on preset type
            target_files = []
            for f in os.listdir(pass1_dir):
                if not f.endswith(".wav"):
                    continue
                f_lower = f.lower()
                if output_stem == "drums" and "drum" in f_lower:
                    target_files.append(os.path.join(pass1_dir, f))
                elif output_stem == "vocals" and "vocal" in f_lower:
                    target_files.append(os.path.join(pass1_dir, f))
                elif output_stem == "instrumental" and ("instrument" in f_lower or "other" in f_lower):
                    target_files.append(os.path.join(pass1_dir, f))
                elif output_stem == "both" and "vocal" in f_lower:
                    # For "both", apply post-processing to vocals
                    target_files.append(os.path.join(pass1_dir, f))
            
            if target_files:
                for target_file in target_files:
                    self.log(f">> Processing: {os.path.basename(target_file)}")
                    self.run_model(post_process, target_file, post_dir)
            else:
                self.log(f">> WARNING: Could not find {output_stem} stem for post-processing")
        
        if all_success:
            self.log(f"\n{'='*50}")
            self.log(f">> ENSEMBLE COMPLETE")
            self.log(f">> Output directory: {ensemble_dir}")
            self.log(f">> Tip: Compare outputs from each pass, or average them in your DAW")
            self.log(f"{'='*50}")
        else:
            self.log("\n>> ENSEMBLE COMPLETED WITH ERRORS")
        
        self.btn_run.configure(state="normal", text="INITIALIZE SEPARATION")

    def select_midi_input(self):
        """Select a stem file for MIDI extraction"""
        # Start in the output directory if it exists
        initial_dir = self.output_dir if os.path.exists(self.output_dir) else None
        filename = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
        )
        if filename:
            self.midi_input_file = filename
            self.midi_file_label.configure(text=os.path.basename(filename))
            self.log(f"MIDI input selected: {os.path.basename(filename)}")

    def run_midi_extraction(self):
        """Run MIDI extraction on selected stem"""
        if not self.midi_input_file:
            self.log("ERROR: No stem file selected for MIDI extraction")
            return
        
        if not MIDI_AVAILABLE:
            self.log("ERROR: piano_transcription_inference not available")
            return
        
        self.btn_midi_extract.configure(state="disabled", text="Extracting...")
        thread = threading.Thread(target=self._midi_extraction_thread)
        thread.start()

    def _midi_extraction_thread(self):
        """MIDI extraction worker thread"""
        try:
            self.log(f"\n{'='*50}")
            self.log("MIDI EXTRACTION")
            self.log(f"{'='*50}")
            self.log(f"Input: {os.path.basename(self.midi_input_file)}")
            self.log("Loading audio...")
            
            # Load audio using librosa (compatible with all versions)
            audio, sr = librosa.load(self.midi_input_file, sr=PIANO_SAMPLE_RATE, mono=True)
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log(f"Using device: {device.upper()}")
            
            # Initialize transcriber with explicit checkpoint path (fixes Windows path issue)
            self.log("Initializing piano transcription model...")
            checkpoint_path = os.path.join(os.path.expanduser('~'), 'piano_transcription_inference_data', 'note_F1=0.9677_pedal_F1=0.9186.pth')
            transcriptor = PianoTranscription(device=device, checkpoint_path=checkpoint_path)
            
            # Generate output path
            input_dir = os.path.dirname(self.midi_input_file)
            input_name = os.path.splitext(os.path.basename(self.midi_input_file))[0]
            midi_output = os.path.join(input_dir, f"{input_name}_transcribed.mid")
            
            self.log("Transcribing to MIDI...")
            
            # Run transcription
            transcriptor.transcribe(audio, midi_output)
            
            self.log(f"\n>> MIDI EXTRACTION COMPLETE")
            self.log(f">> Output: {midi_output}")
            self.log(f"{'='*50}")
            
        except Exception as e:
            self.log(f"MIDI ERROR: {str(e)}")
        
        self.btn_midi_extract.configure(state="normal", text="Extract Melodic MIDI")

    def run_drum_extraction(self):
        """Run drum transcription on selected stem"""
        if not self.midi_input_file:
            self.log("ERROR: No stem file selected for drum extraction")
            return
        
        if not DRUMS_AVAILABLE:
            self.log("ERROR: librosa/mido not available for drum transcription")
            return
        
        self.btn_drum_extract.configure(state="disabled", text="Extracting...")
        thread = threading.Thread(target=self._drum_extraction_thread)
        thread.start()

    def _drum_extraction_thread(self):
        """Drum transcription worker thread using onset detection + frequency analysis"""
        try:
            self.log(f"\n{'='*50}")
            self.log("🥁 DRUM TRANSCRIPTION")
            self.log(f"{'='*50}")
            self.log(f"Input: {os.path.basename(self.midi_input_file)}")
            
            # Load audio
            self.log("Loading audio...")
            y, sr = librosa.load(self.midi_input_file, sr=44100, mono=True)
            
            # Detect onsets
            self.log("Detecting drum hits...")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            self.log(f"Found {len(onset_times)} potential drum hits")
            
            # Analyze frequency content at each onset to classify drum type
            self.log("Classifying drum hits (kick/snare/hihat)...")
            
            drum_hits = []
            hop_length = 512
            
            for onset_frame in onset_frames:
                # Get a small window around the onset
                start_sample = max(0, onset_frame * hop_length - 1024)
                end_sample = min(len(y), onset_frame * hop_length + 4096)
                segment = y[start_sample:end_sample]
                
                if len(segment) < 512:
                    continue
                
                # Compute spectrum
                spectrum = np.abs(np.fft.rfft(segment))
                freqs = np.fft.rfftfreq(len(segment), 1/sr)
                
                # Frequency band energies
                low_energy = np.sum(spectrum[(freqs >= 20) & (freqs < 150)])      # Kick range
                mid_energy = np.sum(spectrum[(freqs >= 150) & (freqs < 1000)])    # Snare body
                high_energy = np.sum(spectrum[(freqs >= 3000) & (freqs < 12000)]) # Hi-hat/cymbals
                snare_crack = np.sum(spectrum[(freqs >= 1000) & (freqs < 5000)])  # Snare crack
                
                total_energy = low_energy + mid_energy + high_energy + snare_crack + 1e-10
                
                # Classify based on frequency distribution
                low_ratio = low_energy / total_energy
                high_ratio = high_energy / total_energy
                snare_ratio = snare_crack / total_energy
                
                onset_time = librosa.frames_to_time(onset_frame, sr=sr)
                
                # Classification logic
                if low_ratio > 0.4 and high_ratio < 0.2:
                    drum_hits.append((onset_time, 'kick'))
                elif snare_ratio > 0.25 and low_ratio > 0.15:
                    drum_hits.append((onset_time, 'snare'))
                elif high_ratio > 0.35:
                    drum_hits.append((onset_time, 'hihat'))
                else:
                    # Default to hihat for ambiguous hits
                    drum_hits.append((onset_time, 'hihat'))
            
            # Count hits by type
            kick_count = sum(1 for _, t in drum_hits if t == 'kick')
            snare_count = sum(1 for _, t in drum_hits if t == 'snare')
            hihat_count = sum(1 for _, t in drum_hits if t == 'hihat')
            
            self.log(f"Classified: {kick_count} kicks, {snare_count} snares, {hihat_count} hi-hats")
            
            # Create MIDI file
            self.log("Creating MIDI file...")
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            
            # Set tempo (120 BPM default, can be detected)
            tempo = 500000  # microseconds per beat (120 BPM)
            track.append(Message('program_change', program=0, time=0))
            
            # Sort hits by time
            drum_hits.sort(key=lambda x: x[0])
            
            # Convert to MIDI events
            ticks_per_beat = mid.ticks_per_beat
            prev_tick = 0
            
            for time_sec, drum_type in drum_hits:
                # Convert time to ticks
                tick = int(time_sec * ticks_per_beat * 2)  # Assuming 120 BPM
                delta = max(0, tick - prev_tick)
                
                note = DRUM_MAP.get(drum_type, 42)
                velocity = 100
                
                # Note on
                track.append(Message('note_on', note=note, velocity=velocity, time=delta, channel=9))
                # Note off (short duration for drums)
                track.append(Message('note_off', note=note, velocity=0, time=50, channel=9))
                
                prev_tick = tick + 50
            
            # Save MIDI
            input_dir = os.path.dirname(self.midi_input_file)
            input_name = os.path.splitext(os.path.basename(self.midi_input_file))[0]
            midi_output = os.path.join(input_dir, f"{input_name}_drums.mid")
            
            mid.save(midi_output)
            
            self.log(f"\n>> DRUM TRANSCRIPTION COMPLETE")
            self.log(f">> Output: {midi_output}")
            self.log(f">> Total hits: {len(drum_hits)}")
            self.log(f"{'='*50}")
            
        except Exception as e:
            self.log(f"DRUM ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        
        self.btn_drum_extract.configure(state="normal", text="Extract Drum MIDI")


if __name__ == "__main__":
    app = GlitchStemUltraApp()
    app.mainloop()
