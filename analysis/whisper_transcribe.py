
import whisper
import librosa
import numpy as np
#np.NAN = np.nan

from pyannote.audio import Pipeline
import xml.etree.ElementTree as ET
import os

def diarize_audio(audio_path):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")
    
    diarization = pipeline(audio_path)
    return diarization

def transcribe_segments(audio_path, diarization):
    print("Loading Whisper model...")
    model = whisper.load_model("medium.en")
    print("Transcribing segments...")

    transcripts = []

    # Manually load and resample audio to 16 kHz
    audio, sr = librosa.load(audio_path, sr=16000)  
    audio = np.array(audio, dtype=np.float32)  

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) > 0:
            result = model.transcribe(segment_audio)
            transcript = result['text']
            transcripts.append((speaker, segment.start, segment.end, transcript))
            print(f"Transcribed segment from {segment.start:.1f}s to {segment.end:.1f}s for speaker {speaker}.")
        else:
            print(f"No audio to transcribe from {segment.start:.1f}s to {segment.end:.1f}s for speaker {speaker}.")
            transcripts.append((speaker, segment.start, segment.end, "[No audio in segment]"))

    return transcripts

def main(audio_path, txt_path):
    diarization = diarize_audio(audio_path)
    transcripts = transcribe_segments(audio_path, diarization)

    with open(txt_path, 'w', encoding='utf-8') as f:
        for speaker, start, end, transcript in transcripts:
            #f.write(f"Speaker {speaker}: {transcript} (from {start:.1f}s to {end:.1f}s)\n")
            f.write(f"{transcript} ")
    print(f"Transcripts written to {txt_path}")

if __name__ == "__main__":
   # audio_file_path = '/mnt/dv/wid/projects6/Hawkins-neuro-tangrams/Jesse_neurotanagrams/new-cleaned-audio/cleaned/sessid002/sessid002-PreTest.wav'  
   # file path to your audio file
    audio_path = input("Enter the path to the audio file: ")
    txt_path = input("Enter the path to the output text file: ")
    main(audio_path,txt_path)
