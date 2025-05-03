
# import tempfile
# from collections import defaultdict

# import matplotlib.pyplot as plt
# import seaborn as sns
# from dotenv import load_dotenv
# from pyannote.audio import Pipeline
# from pydub import AudioSegment
# from transformers import pipeline as hf_pipeline

# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from snownlp import SnowNLP
# import os



# # ---------------------------------------------------------------------------
# # Setup
# # ---------------------------------------------------------------------------
# load_dotenv()  # expects HUGGINGFACE_TOKEN in .env
# HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("Please set HUGGINGFACE_TOKEN in your .env")

# # download VADER lexicon if first run
# nltk.download("vader_lexicon", quiet=True)
# sia = SentimentIntensityAnalyzer()

# # ---------------------------------------------------------------------------
# # Helper: ensure 16 kHz mono WAV
# # ---------------------------------------------------------------------------
# def ensure_wav16mono(path: str) -> str:
#     """
#     Convert <path> to 16 kHz mono WAV if not already, return new path.
#     """
#     base, ext = os.path.splitext(path)
#     wav_path = base + "_16k_mono.wav"
#     if os.path.exists(wav_path):
#         return wav_path

#     audio = AudioSegment.from_file(path)
#     audio = audio.set_frame_rate(16000).set_channels(1)
#     audio.export(wav_path, format="wav")
#     return wav_path

# # ---------------------------------------------------------------------------
# # 1) Speaker Diarization (force 2 speakers)
# # ---------------------------------------------------------------------------
# def perform_diarization(audio_path: str) -> dict:
#     wav = ensure_wav16mono(audio_path)
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization@2.1",
#         use_auth_token=HF_TOKEN
#     )
#     diarization = pipeline(wav, min_speakers=2, max_speakers=2)
#     speaker_data = defaultdict(list)
#     for turn, _, spk in diarization.itertracks(yield_label=True):
#         speaker_data[spk].append({
#             "start":    turn.start,
#             "end":      turn.end,
#             "duration": turn.duration
#         })
#     return speaker_data

# # ---------------------------------------------------------------------------
# # 2) ASR via Wav2Vec2
# # ---------------------------------------------------------------------------
# asr = hf_pipeline(
#     "automatic-speech-recognition",
#     model="facebook/wav2vec2-large-960h",
#     chunk_length_s=30,
# )

# def transcribe_segment(segment: AudioSegment) -> str:
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#         segment.export(tmp.name, format="wav")
#         res = asr(tmp.name)
#     os.remove(tmp.name)
#     return res.get("text", "").strip()

# # ---------------------------------------------------------------------------
# # 3) Offline Multilingual Sentiment
# # ---------------------------------------------------------------------------
# def analyze_sentiment(text: str, lang: str):
#     """
#     Returns (label, score) with score in [-1,1].
#     """
#     text = text.strip()
#     if not text:
#         return "NEUTRAL", 0.0

#     if lang == "english":
#         c = sia.polarity_scores(text)["compound"]
#         if c >= 0.05:
#             return "POSITIVE", c
#         elif c <= -0.05:
#             return "NEGATIVE", c
#         else:
#             return "NEUTRAL", c

#     # Mandarin/Cantonese → SnowNLP
#     s = SnowNLP(text).sentiments  # [0..1]
#     v = (s - 0.5) * 2             # normalize to [-1..1]
#     if v >= 0.2:
#         return "POSITIVE", v
#     elif v <= -0.2:
#         return "NEGATIVE", v
#     else:
#         return "NEUTRAL", v

# # ---------------------------------------------------------------------------
# # 4) Plotting Helpers
# # ---------------------------------------------------------------------------
# def plot_speaker_distribution(durations):
#     total = sum(durations.values())
#     plt.figure(figsize=(6,6))
#     plt.pie(
#         durations.values(),
#         labels=[f"{spk} ({dur:.1f}s)" for spk, dur in durations.items()],
#         autopct=lambda p: f"{p:.1f}% ({p*total/100:.1f}s)",
#         startangle=90
#     )
#     plt.title("Speaking Time Distribution")
#     plt.axis("equal")
#     plt.savefig("speaker_pie_chart.png")
#     plt.close()

# def plot_speaker_timeline(speaker_data):
#     plt.figure(figsize=(12,4))
#     for i, (spk, segments) in enumerate(speaker_data.items()):
#         for seg in segments:
#             plt.hlines(i, seg["start"], seg["end"], linewidth=8)
#     plt.yticks(range(len(speaker_data)), speaker_data.keys())
#     plt.xlabel("Time (s)")
#     plt.title("Speaker Turn Timeline")
#     plt.tight_layout()
#     plt.savefig("speaker_timeline.png")
#     plt.close()

# def plot_word_count_distribution(word_counts):
#     plt.figure(figsize=(8,5))
#     sns.barplot(x=list(word_counts.keys()), y=list(word_counts.values()))
#     plt.xlabel("Speaker")
#     plt.ylabel("Estimated Word Count")
#     plt.title("Word Count Distribution")
#     plt.savefig("word_count_distribution.png")
#     plt.close()

# def plot_sentiment(sentiments):
#     speakers = list(sentiments.keys())
#     vals, colors = [], []
#     for label, score in sentiments.values():
#         lab = label.upper()
#         if lab == "POSITIVE":
#             vals.append(score);   colors.append("#2ca02c")
#         elif lab == "NEGATIVE":
#             vals.append(-score);  colors.append("#d62728")
#         else:
#             vals.append(0.0);     colors.append("#7f7f7f")
#     plt.figure(figsize=(8,4))
#     plt.bar(speakers, vals, color=colors)
#     plt.axhline(0, color="black", linewidth=0.8)
#     plt.ylabel("Sentiment Score (-1 … +1)")
#     plt.title("Speaker-Level Sentiment")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("speaker_sentiment.png")
#     plt.close()

# # ---------------------------------------------------------------------------
# # 5) Main Pipeline
# # ---------------------------------------------------------------------------
# def analyze_conversation(audio_path: str, lang: str):
#     # 5.1 Diarization
#     speaker_data = perform_diarization(audio_path)

#     # 5.2 Durations & word counts
#     durations   = {spk: sum(seg["duration"] for seg in segs) 
#                    for spk, segs in speaker_data.items()}
#     word_counts = {spk: int(120 * durations[spk] / 60) 
#                    for spk in durations}

#     # 5.3 Transcribe per speaker
#     audio = AudioSegment.from_file(audio_path)
#     texts = defaultdict(str)
#     for spk, segs in speaker_data.items():
#         for seg in segs:
#             start, end = int(seg["start"]*1000), int(seg["end"]*1000)
#             chunk = audio[start:end]
#             texts[spk] += " " + transcribe_segment(chunk)

#     # 5.4 Sentiment per speaker
#     sentiments = {spk: analyze_sentiment(txt, lang) 
#                   for spk, txt in texts.items()}

#     # 5.5 Plot outputs
#     plot_speaker_distribution(durations)
#     plot_speaker_timeline(speaker_data)
#     plot_word_count_distribution(word_counts)
#     plot_sentiment(sentiments)

#     # 5.6 Text report
#     with open("conversation_report.txt", "w", encoding="utf-8") as f:
#         f.write("Conversation Analytics Report\n\n")
#         f.write(f"Total Speakers: {len(speaker_data)}\n\n")
#         f.write("Speaking Time (s):\n")
#         for spk, dur in durations.items():
#             f.write(f"- {spk}: {dur:.1f}s\n")
#         f.write("\nEstimated Word Counts:\n")
#         for spk, wc in word_counts.items():
#             f.write(f"- {spk}: {wc} words\n")
#         f.write("\nSentiment Analysis:\n")
#         for spk, (lab, score) in sentiments.items():
#             f.write(f"- {spk}: {lab} ({score:.2f})\n")

#     print("✅ Done. Generated:")
#     print(" • speaker_pie_chart.png")
#     print(" • speaker_timeline.png")
#     print(" • word_count_distribution.png")
#     print(" • speaker_sentiment.png")
#     print(" • conversation_report.txt")

# # ---------------------------------------------------------------------------
# # Entry Point
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Diarization & Sentiment (Eng/Mandarin/Cantonese)"
#     )
#     parser.add_argument("audio", help="Path to audio file (wav/mp3)")
#     parser.add_argument(
#         "--lang",
#         choices=["english", "mandarin", "cantonese"],
#         default="english",
#         help="Language for sentiment analysis"
#     )
#     args = parser.parse_args()

#     analyze_conversation(args.audio, args.lang)
    
    
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import pipeline as hf_pipeline

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from snownlp import SnowNLP
import os

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()  # expects HUGGINGFACE_TOKEN in .env
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_TOKEN in your .env")

# download VADER lexicon if first run
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------------------------------------------------------------------------
# Helper: ensure 16 kHz mono WAV
# ---------------------------------------------------------------------------
def ensure_wav16mono(path: str) -> str:
    """
    Convert <path> to 16 kHz mono WAV if not already, return new path.
    """
    base, ext = os.path.splitext(path)
    wav_path = base + "_16k_mono.wav"
    if os.path.exists(wav_path):
        return wav_path

    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    return wav_path

# ---------------------------------------------------------------------------
# 1) Speaker Diarization (auto-detect speakers)
# ---------------------------------------------------------------------------
def perform_diarization(audio_path: str) -> dict:
    wav = ensure_wav16mono(audio_path)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=HF_TOKEN
    )
    # let the model infer speaker count
    diarization = pipeline(wav)
    speaker_data = defaultdict(list)
    for turn, _, spk in diarization.itertracks(yield_label=True):
        speaker_data[spk].append({
            "start":    turn.start,
            "end":      turn.end,
            "duration": turn.duration
        })
    return speaker_data

# ---------------------------------------------------------------------------
# 2) ASR via Wav2Vec2
# ---------------------------------------------------------------------------
asr = hf_pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-large-960h",
    chunk_length_s=30,
)

def transcribe_segment(segment: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        segment.export(tmp.name, format="wav")
        res = asr(tmp.name)
    os.remove(tmp.name)
    return res.get("text", "").strip()

# ---------------------------------------------------------------------------
# 3) Offline Multilingual Sentiment
# ---------------------------------------------------------------------------
def analyze_sentiment(text: str, lang: str):
    """
    Returns (label, score) with score in [-1,1].
    """
    text = text.strip()
    if not text:
        return "NEUTRAL", 0.0

    if lang == "english":
        c = sia.polarity_scores(text)["compound"]
        if c >= 0.05:
            return "POSITIVE", c
        elif c <= -0.05:
            return "NEGATIVE", c
        else:
            return "NEUTRAL", c

    # Mandarin/Cantonese → SnowNLP
    s = SnowNLP(text).sentiments  # [0..1]
    v = (s - 0.5) * 2             # normalize to [-1..1]
    if v >= 0.2:
        return "POSITIVE", v
    elif v <= -0.2:
        return "NEGATIVE", v
    else:
        return "NEUTRAL", v

# ---------------------------------------------------------------------------
# 4) Plotting Helpers
# ---------------------------------------------------------------------------
def plot_speaker_distribution(durations):
    total = sum(durations.values())
    plt.figure(figsize=(6,6))
    plt.pie(
        durations.values(),
        labels=[f"{spk} ({dur:.1f}s)" for spk, dur in durations.items()],
        autopct=lambda p: f"{p:.1f}% ({p*total/100:.1f}s)",
        startangle=90
    )
    plt.title("Speaking Time Distribution")
    plt.axis("equal")
    plt.savefig("speaker_pie_chart.png")
    plt.close()


def plot_speaker_timeline(speaker_data):
    plt.figure(figsize=(12,4))
    for i, (spk, segments) in enumerate(speaker_data.items()):
        for seg in segments:
            plt.hlines(i, seg["start"], seg["end"], linewidth=8)
    plt.yticks(range(len(speaker_data)), speaker_data.keys())
    plt.xlabel("Time (s)")
    plt.title("Speaker Turn Timeline")
    plt.tight_layout()
    plt.savefig("speaker_timeline.png")
    plt.close()


def plot_word_count_distribution(word_counts):
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(word_counts.keys()), y=list(word_counts.values()))
    plt.xlabel("Speaker")
    plt.ylabel("Estimated Word Count")
    plt.title("Word Count Distribution")
    plt.savefig("word_count_distribution.png")
    plt.close()


def plot_sentiment(sentiments):
    speakers = list(sentiments.keys())
    vals, colors = [], []
    for label, score in sentiments.values():
        lab = label.upper()
        if lab == "POSITIVE":
            vals.append(score);   colors.append("#2ca02c")
        elif lab == "NEGATIVE":
            vals.append(-score);  colors.append("#d62728")
        else:
            vals.append(0.0);     colors.append("#7f7f7f")
    plt.figure(figsize=(8,4))
    plt.bar(speakers, vals, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Sentiment Score (-1 … +1)")
    plt.title("Speaker-Level Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("speaker_sentiment.png")
    plt.close()

# ---------------------------------------------------------------------------
# 5) Main Pipeline
# ---------------------------------------------------------------------------
def analyze_conversation(audio_path: str, lang: str):
    # 5.1 Diarization
    speaker_data = perform_diarization(audio_path)

    # 5.2 Durations & word counts
    durations   = {spk: sum(seg["duration"] for seg in segs) 
                   for spk, segs in speaker_data.items()}
    word_counts = {spk: int(120 * durations[spk] / 60) 
                   for spk in durations}

    # 5.3 Transcribe per speaker
    audio = AudioSegment.from_file(audio_path)
    texts = defaultdict(str)
    for spk, segs in speaker_data.items():
        for seg in segs:
            start, end = int(seg["start"]*1000), int(seg["end"]*1000)
            chunk = audio[start:end]
            texts[spk] += " " + transcribe_segment(chunk)

    # 5.4 Sentiment per speaker
    sentiments = {spk: analyze_sentiment(txt, lang) 
                  for spk, txt in texts.items()}

    # 5.5 Plot outputs
    plot_speaker_distribution(durations)
    plot_speaker_timeline(speaker_data)
    plot_word_count_distribution(word_counts)
    plot_sentiment(sentiments)

    # 5.6 Text report
    with open("conversation_report.txt", "w", encoding="utf-8") as f:
        f.write("Conversation Analytics Report\n\n")
        f.write(f"Total Speakers: {len(speaker_data)}\n\n")
        f.write("Speaking Time (s):\n")
        for spk, dur in durations.items():
            f.write(f"- {spk}: {dur:.1f}s\n")
        f.write("\nEstimated Word Counts:\n")
        for spk, wc in word_counts.items():
            f.write(f"- {spk}: {wc} words\n")
        f.write("\nSentiment Analysis:\n")
        for spk, (lab, score) in sentiments.items():
            f.write(f"- {spk}: {lab} ({score:.2f})\n")

    print("✅ Done. Generated:")
    print(" • speaker_pie_chart.png")
    print(" • speaker_timeline.png")
    print(" • word_count_distribution.png")
    print(" • speaker_sentiment.png")
    print(" • conversation_report.txt")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diarization & Sentiment (Eng/Mandarin/Cantonese)"
    )
    parser.add_argument("audio", help="Path to audio file (wav/mp3)")
    parser.add_argument(
        "--lang",
        choices=["english", "mandarin", "cantonese"],
        default="english",
        help="Language for sentiment analysis"
    )
    args = parser.parse_args()

    analyze_conversation(args.audio, args.lang)