# app.py
"""
AutoDub completo con:
- Transcripción (Whisper)
- (Opcional) Diarización (pyannote si tienes HF_TOKEN)
- Traducción (M2M100 offline si está, fallback googletrans online)
- Detección de género por voz (librosa -> yin -> median f0)
- TTS por género: usa Coqui TTS si está disponible y pasas MALE_VOICE/FEMALE_VOICE,
  si no hay Coqui, usa gTTS (sin control de género).
- Reemplazo de pista de audio con ffmpeg (sin moviepy).
"""

import os
import tempfile
import subprocess
import traceback
from typing import List, Optional, Dict

import gradio as gr
import whisper
from pydub import AudioSegment
import numpy as np
import librosa

# Optional heavy deps (use if available)
USE_PYANNOTE = False
diarization_pipeline = None
try:
    from pyannote.audio import Pipeline
    USE_PYANNOTE = True
except Exception:
    USE_PYANNOTE = False

USE_COQUI = False
tts_coqui = None
try:
    from TTS.api import TTS
    USE_COQUI = True
except Exception:
    USE_COQUI = False

# gTTS fallback
from gtts import gTTS

# Try offline translator (M2M100). If missing sentencepiece, fallback online.
OFFLINE_TRANSLATION = False
translation_model = None
tokenizer = None
translator_online = None
try:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
except Exception:
    M2M100ForConditionalGeneration = None
    M2M100Tokenizer = None

try:
    from googletrans import Translator
except Exception:
    Translator = None

# -----------------------
# CONFIG
# -----------------------
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")  # small by default
TRANSLATION_MODEL = "facebook/m2m100_418M"

# mapping visible in UI
LANGS = {
    "español": "es",
    "inglés": "en",
    "francés": "fr",
    "alemán": "de",
    "chino": "zh",
    "portugués": "pt",
    "italiano": "it",
    "japonés": "ja",
    "coreano": "ko"
}

GTTS_MAP = {
    "es": "es",
    "en": "en",
    "fr": "fr",
    "de": "de",
    "zh": "zh-CN",
    "pt": "pt",
    "it": "it",
    "ja": "ja",
    "ko": "ko"
}

# Voice mapping for Coqui: you can set env vars MALE_VOICE / FEMALE_VOICE to control which
# speaker names Coqui uses. If empty, code will try TTS default.
MALE_VOICE = os.environ.get("MALE_VOICE", None)
FEMALE_VOICE = os.environ.get("FEMALE_VOICE", None)

# Pitch thresholds (Hz) to classify male/female
# male typical median_f0 ~ 85-180, female ~165-255. We use threshold ~165 Hz.
GENDER_F0_THRESHOLD = 165.0

# -----------------------
# Utilities
# -----------------------
def run_cmd(cmd: List[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode(errors="ignore")
    err = proc.stderr.decode(errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out, err

def ensure_ffmpeg():
    try:
        out = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if out.returncode != 0:
            raise FileNotFoundError()
    except Exception:
        raise RuntimeError("ffmpeg no encontrado. Instala ffmpeg y asegúrate que esté en PATH.")

def get_video_path(video_input):
    """Normalize Gradio input to a file path"""
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        for k in ("name", "tmp_path", "file"):
            if k in video_input:
                return video_input[k]
        for v in video_input.values():
            if isinstance(v, str) and os.path.exists(v):
                return v
    if hasattr(video_input, "name"):
        return video_input.name
    raise ValueError(f"Formato de entrada de video no soportado: {type(video_input)}")

# -----------------------
# Model init
# -----------------------
print("Cargando Whisper...")
whisper_model = whisper.load_model(WHISPER_SIZE)

# Translation: try M2M100 offline; otherwise try googletrans
try:
    if M2M100Tokenizer is None or M2M100ForConditionalGeneration is None:
        raise ImportError("M2M100 unavailable")
    print("Cargando traductor M2M100 (puede descargar ~400MB)...")
    tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL)
    translation_model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL)
    OFFLINE_TRANSLATION = True
    print("Traducción: modo OFFLINE (M2M100).")
except Exception as e:
    OFFLINE_TRANSLATION = False
    print("No se pudo cargar M2M100 (fallback online). Error:", e)
    if Translator is not None:
        translator_online = Translator()
        print("Traducción: modo ONLINE (googletrans).")
    else:
        translator_online = None
        print("No hay traductor disponible; se usará texto original.")

# pyannote diarization (optional)
if USE_PYANNOTE:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        try:
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
            print("pyannote cargado (diarización).")
        except Exception as e:
            print("pyannote no se pudo cargar:", e)
            diarization_pipeline = None
    else:
        print("HF_TOKEN no definido — pyannote no se usará.")
        diarization_pipeline = None

# Coqui TTS (optional)
if USE_COQUI:
    try:
        # El modelo por defecto puede ser cambiado; si tu modelo soporta 'speaker' pasa MALE_VOICE/FEMALE_VOICE.
        tts_coqui = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
        print("Coqui TTS cargado.")
    except Exception as e:
        print("Coqui TTS no inicializado (fallback gTTS). Error:", e)
        tts_coqui = None

# -----------------------
# Audio utility functions
# -----------------------
def extract_audio(video_path: str, out_wav: str):
    """Extract WAV mono 16k from video using ffmpeg"""
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        out_wav
    ]
    run_cmd(cmd)

def diarize_audio(audio_path: str):
    """Return list of (start, end, speaker) if diarization available, else []"""
    if diarization_pipeline is None:
        return []
    try:
        diar = diarization_pipeline({"audio": audio_path})
        segs = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segs.append((turn.start, turn.end, speaker))
        # map to max 2 labels
        mapping = {}
        next_id = 0
        mapped = []
        for s,e,sp in segs:
            if sp not in mapping:
                if next_id < 2:
                    mapping[sp] = f"SPEAKER_{next_id}"
                    next_id += 1
                else:
                    mapping[sp] = "SPEAKER_0"
            mapped.append((s,e,mapping[sp]))
        return mapped
    except Exception as e:
        print("Error diarization:", e)
        return []

def whisper_transcribe(audio_path: str):
    res = whisper_model.transcribe(audio_path, word_timestamps=False)
    lang = res.get("language", None)
    segs = []
    for s in res["segments"]:
        segs.append({"start": s["start"], "end": s["end"], "text": s["text"].strip()})
    return lang, segs

def align_segments(whisper_segs, diarization_segs):
    if not diarization_segs:
        return [{"start": s["start"], "end": s["end"], "text": s["text"], "speaker": "SPEAKER_0"} for s in whisper_segs]
    assigned = []
    for seg in whisper_segs:
        s_start, s_end = seg["start"], seg["end"]
        best_sp = "SPEAKER_0"
        best_overlap = 0.0
        for d_start, d_end, sp in diarization_segs:
            overlap = max(0, min(s_end, d_end) - max(s_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sp = sp
        assigned.append({"start": s_start, "end": s_end, "text": seg["text"], "speaker": best_sp})
    return assigned

def translate_text(text: str, src_lang: Optional[str], tgt_lang: str):
    if not text.strip():
        return ""
    # offline M2M100 if available
    if OFFLINE_TRANSLATION and tokenizer is not None and translation_model is not None:
        try:
            if src_lang:
                try:
                    tokenizer.src_lang = src_lang
                except Exception:
                    pass
            encoded = tokenizer(text, return_tensors="pt")
            forced_bos_id = tokenizer.get_lang_id(tgt_lang)
            generated = translation_model.generate(**encoded, forced_bos_token_id=forced_bos_id, max_length=512)
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return translated
        except Exception as e:
            print("Error traducción offline:", e)
            # fallthrough to online
    if translator_online is not None:
        try:
            res = translator_online.translate(text, src=src_lang or "auto", dest=tgt_lang)
            return res.text
        except Exception as e:
            print("googletrans falló:", e)
            return text
    return text

# -----------------------
# Gender detection (librosa -> yin)
# -----------------------
def detect_gender_from_wav(wav_path: str) -> str:
    """
    Estimate F0 using librosa.yin and return 'male' / 'female' / 'unknown'
    based on median F0 threshold.
    """
    try:
        # librosa.load returns float32 array
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) < sr // 10:
            # extremely short -> unknown
            return "unknown"
        # use yin for f0 estimation
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=256)
        # f0 contains np.nan where unvoiced
        voiced = f0[~np.isnan(f0)]
        if len(voiced) == 0:
            return "unknown"
        median_f0 = float(np.median(voiced))
        # debug print
        print(f"detect_gender_from_wav: median_f0={median_f0:.1f} Hz")
        if median_f0 < GENDER_F0_THRESHOLD:
            return "male"
        else:
            return "female"
    except Exception as e:
        print("Error detectando género (librosa):", e)
        return "unknown"

# -----------------------
# TTS synthesis (gender-aware)
# -----------------------
def synthesize_tts(text: str, tgt_lang: str, out_wav: str, gender: str = "unknown"):
    """
    Use Coqui TTS with MALE_VOICE / FEMALE_VOICE if available.
    Fallback to gTTS otherwise (no gender control).
    """
    if not text.strip():
        AudioSegment.silent(duration=10).export(out_wav, format="wav")
        return out_wav

    # Try Coqui if available
    if tts_coqui is not None:
        try:
            # If user set voice names in env, prefer them
            if gender == "male" and MALE_VOICE:
                tts_coqui.tts_to_file(text=text, file_path=out_wav, speaker=MALE_VOICE)
                return out_wav
            if gender == "female" and FEMALE_VOICE:
                tts_coqui.tts_to_file(text=text, file_path=out_wav, speaker=FEMALE_VOICE)
                return out_wav
            # If no specific voices configured, try default tts call (model may or may not support 'speaker')
            try:
                tts_coqui.tts_to_file(text=text, file_path=out_wav)
                return out_wav
            except TypeError:
                # older/newer API variations: try without speaker param
                tts_coqui.tts_to_file(text=text, file_path=out_wav)
                return out_wav
        except Exception as e:
            print("Coqui TTS falló (se intentará gTTS). Error:", e)

    # Fallback gTTS -> create mp3 then convert to wav mono 16k
    try:
        gtts_lang = GTTS_MAP.get(tgt_lang, "en")
        tts = gTTS(text=text, lang=gtts_lang)
        tmp_mp3 = out_wav + ".mp3"
        tts.save(tmp_mp3)
        seg = AudioSegment.from_file(tmp_mp3, format="mp3")
        seg = seg.set_frame_rate(16000).set_channels(1)
        seg.export(out_wav, format="wav")
        os.remove(tmp_mp3)
        # NOTE: gTTS has no gender control; if you want gendered voices, install Coqui and configure MALE_VOICE/FEMALE_VOICE.
        return out_wav
    except Exception as e:
        raise RuntimeError(f"Error sintetizando TTS: {e}")

# -----------------------
# Compose final audio (overlay by timestamps)
# -----------------------
def compose_audio(original_wav: str, segments: List[dict], sample_rate: int = 16000):
    """
    segments: list of dicts with keys 'start' (s), 'tts_wav' (path)
    returns path to final wav
    """
    orig = AudioSegment.from_wav(original_wav)
    total_ms = len(orig)
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=sample_rate)
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        tts_wav = seg["tts_wav"]
        if not os.path.exists(tts_wav):
            continue
        frag = AudioSegment.from_wav(tts_wav)
        frag = frag.set_frame_rate(sample_rate).set_channels(1)
        canvas = canvas.overlay(frag, position=start_ms)
    out_final = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    canvas.export(out_final, format="wav")
    return out_final

def mux_audio_video(original_video: str, new_audio_wav: str, out_path: str):
    ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", new_audio_wav,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:a", "aac",
        out_path
    ]
    run_cmd(cmd)

# -----------------------
# Main pipeline
# -----------------------
def process_video(video_input, target_language_name):
    try:
        if video_input is None:
            return None, "No se subió video."
        video_path = get_video_path(video_input)
        if not os.path.exists(video_path):
            return None, f"Archivo no encontrado: {video_path}"

        tgt_code = LANGS.get(target_language_name, "es")

        # 1) extract audio
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        extract_audio(video_path, tmp_audio)

        # 2) diarization (optional)
        diarization_segs = []
        if diarization_pipeline is not None:
            try:
                diarization_segs = diarize_audio(tmp_audio)
            except Exception as e:
                print("Warning diarization:", e)
                diarization_segs = []

        # 3) whisper transcription
        detected_lang, whisper_segs = whisper_transcribe(tmp_audio)
        print("Whisper detected language:", detected_lang)

        # 4) align to speakers
        aligned = align_segments(whisper_segs, diarization_segs)

        # 5) for each segment: translate, detect gender, synthesize
        tts_segments = []
        # load original audio as pydub to extract fragments for gender detection
        orig_audio = AudioSegment.from_wav(tmp_audio)
        for seg in aligned:
            text = seg["text"]
            if not text.strip():
                continue
            translated = translate_text(text, detected_lang, tgt_code)

            # extract fragment wav for gender detection
            frag_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            # guard against bounds
            audio_fragment = orig_audio[start_ms:end_ms] if end_ms > start_ms else orig_audio[start_ms:start_ms+50]
            audio_fragment.export(frag_wav, format="wav")

            gender = detect_gender_from_wav(frag_wav)
            # cleanup fragment? keep for debug
            print(f"Segment [{seg['start']:.2f}-{seg['end']:.2f}] speaker={seg.get('speaker')} gender={gender}")

            # synthesize with gender preference
            wav_seg = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            synthesize_tts(translated, tgt_code, wav_seg, gender=gender)
            tts_segments.append({"start": seg["start"], "tts_wav": wav_seg, "speaker": seg.get("speaker", "SPEAKER_0"), "gender": gender})

        # 6) compose final audio
        final_wav = compose_audio(tmp_audio, tts_segments)

        # 7) mux into video
        out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        mux_audio_video(video_path, final_wav, out_mp4)

        # mode info
        mode = "OFFLINE (M2M100)" if OFFLINE_TRANSLATION else ("ONLINE googletrans" if translator_online else "NO TRADUCCIÓN")
        return out_mp4, f"Éxito — idioma detectado: {detected_lang} — segmentos: {len(tts_segments)} — modo traducción: {mode}"

    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR process_video:", e, tb)
        return None, f"Error: {e}\n\nTraceback:\n{tb}"

# -----------------------
# Gradio UI
# -----------------------
title = "🎙️ AutoDub — Doblaje con detección de género"
desc = "Sube un video → detecto idioma → traduzco → detecto género por fragmento → sintetizo voz masculina/femenina (si Coqui disponible) → sincronizo."

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}\n{desc}")
    with gr.Row():
        video_in = gr.Video(label="Sube tu video")
        lang_dropdown = gr.Dropdown(list(LANGS.keys()), label="Idioma destino", value="español")
    run_btn = gr.Button("Procesar")
    output_video = gr.Video(label="Video doblado")
    status = gr.Textbox(label="Estado / Mensajes", lines=6)

    def run_pipeline(video, lang_name):
        out, msg = process_video(video, lang_name)
        if out:
            return out, msg
        else:
            return None, msg

    run_btn.click(run_pipeline, inputs=[video_in, lang_dropdown], outputs=[output_video, status])

if __name__ == "__main__":
    demo.launch()
