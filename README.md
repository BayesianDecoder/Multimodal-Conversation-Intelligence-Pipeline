# 🎙️ Multimodal Conversation Intelligence Pipeline

## This project provides a full pipeline to analyze spoken conversations from audio files using Automatic Speech Recognition (ASR), LLM correction, cleaning logic, summarization, and speaker-level analytics.


## 🧩 Node Descriptions

### 🔹 `node1.py`: ASR + Correction
- Converts audio (MP3/WAV) into text using iFLYTEK.
- Each chunk is corrected with `deepseek-r1:1.5b` via Ollama.
- Output: `corrected_transcript.txt`

### 🔹 `node2.py`: Cleaning
- Removes `<think>...</think>` blocks from the transcript.
- Output: `final_clean_output.txt`

### 🔹 `node3.py`: Summarization
- Uses `qwen:7b-chat` to generate meeting summary & action items.
- Output: `meeting_summary_YYYYMMDD_HHMM.md`

### 🔹 `dashboard.py`: Analytics
- Uses `pyannote` for speaker diarization
- Transcribes each segment using `wav2vec2`
- Computes sentiment: `VADER` (English), `SnowNLP` (Chinese).
- Output:
  - `conversation_report.txt`
  - `speaker_pie_chart.png`
  - `speaker_timeline.png`
  - `speaker_sentiment.png`
  - `word_count_distribution.png`

---

## ✅ Requirements
Install dependencies (Python 3.9–3.12):


``` 
pip install -r requirements.txt

```

Your .env file should contain:

```
IFLY_APPID=your_iflytek_appid
IFLY_APIKEY=your_iflytek_apikey
IFLY_APISECRET=your_iflytek_apisecret
HUGGINGFACE_TOKEN=your_hf_token_for_pyannote
```

Run ollama 

```
ollama serve
```

Run the pipeline 

```
python run_pipeline.py /path/to/audio.mp3

```
## 📊 Outputs
- corrected_transcript.txt: Clean LLM-corrected speech.
- final_clean_output.txt: Final version without explanatory notes.
- meeting_summary_*.md: Summary of the meeting.
-Visuals and stats saved as .png and .txt.

## 📌 Notes
- For Chinese sentiment, SnowNLP requires jieba and proper font support.
- Wav2Vec2 is used for speaker-level ASR due to better accuracy in long audio segments.



