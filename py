# Instalar las bibliotecas necesarias
!pip install torch torchaudio transformers moviepy pydub

import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from pydub import AudioSegment
import moviepy.editor as mp

# Configurar el modelo de traducción
model_name = "Helsinki-NLP/opus-mt-en-es"  # Ejemplo de modelo de traducción de inglés a español
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Función para traducir texto
def traducir_texto(texto, modelo, tokenizer):
    inputs = tokenizer(texto, return_tensors="pt")
    translated_tokens = modelo.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Función para procesar el audio del video
def procesar_audio(video_path, idioma_destino):
    # Extraer audio del video
    clip = mp.VideoFileClip(video_path)
    audio_path = "audio.mp3"
    clip.audio.write_audiofile(audio_path)

    # Cargar audio con pydub
    audio = AudioSegment.from_mp3(audio_path)

    # Dividir audio en segmentos (por ejemplo, cada 5 segundos)
    segmentos = [audio[i:i+5000] for i in range(0, len(audio), 5000)]

    # Procesar cada segmento
    textos_traducidos = []
    for i, segmento in enumerate(segmentos):
        # Guardar segmento temporalmente
        temp_path = f"segmento_{i}.mp3"
        segmento.export(temp_path, format="mp3")

        # Usar un modelo de reconocimiento de voz para transcribir el audio a texto
        recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")
        texto = recognizer(temp_path)["text"]

        # Traducir el texto
        texto_traducido = traducir_texto(texto, model, tokenizer)
        textos_traducidos.append(texto_traducido)

        # Eliminar archivo temporal
        os.remove(temp_path)

    return textos_traducidos

# Función para generar audio con voz sintética utilizando un modelo de código abierto
def generar_audio_sintetico(texto, idioma_destino):
    from gtts import gTTS
    tts = gTTS(text=texto, lang=idioma_destino)
    audio_path = f"audio_sintetico_{idioma_destino}.mp3"
    tts.save(audio_path)
    return audio_path

# Función principal para doblar el video
def doblar_video(video_path, idioma_destino):
    # Procesar audio y obtener textos traducidos
    textos_traducidos = procesar_audio(video_path, idioma_destino)

    # Generar audio sintético para cada segmento
    segmentos_audio = []
    for texto in textos_traducidos:
        audio_path = generar_audio_sintetico(texto, idioma_destino)
        segmentos_audio.append(AudioSegment.from_mp3(audio_path))

    # Combinar todos los segmentos de audio en uno solo
    audio_final = sum(segmentos_audio)

    # Guardar audio final
    audio_final.export("audio_final.mp3", format="mp3")

    # Cargar video original y reemplazar audio
    clip = mp.VideoFileClip(video_path)
    final_clip = clip.set_audio("audio_final.mp3")
    final_clip.write_videofile("video_doblado.mp4")
