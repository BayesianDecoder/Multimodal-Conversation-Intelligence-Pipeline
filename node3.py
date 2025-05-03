
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
import ollama
from ollama._types import ResponseError

# Load environment variables (if needed for Ollama setup)
load_dotenv()

def analyze_transcript(transcript: str) -> str:
    """Send transcript to local qwen:7b-chat model via Ollama for a concise summary only"""
    prompt = f"""Summarize this conversation transcript into a concise meeting summary with action points. Output ONLY the summary—no explanations or extra commentary.

Transcript:
{transcript}"""

    try:
        response = ollama.chat(
            model="qwen:7b-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert meeting summarization assistant. Output only a concise summary with action items."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={"temperature": 0.3}
        )
        return response["message"]["content"].strip()
    except ResponseError as e:
        print(f" LLM error: {e}")
        raise

def process_transcript(file_path: str):
    """Main processing workflow"""
    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    print("⏳ Generating concise summary with qwen:7b-chat…")
    summary = analyze_transcript(transcript)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"meeting_summary_{timestamp}.md"
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(summary)
    
    print(f"Summary saved to {output_file}")
    print("\n" + summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a concise meeting summary using qwen:7b-chat"
    )
    parser.add_argument("file", help="Path to transcript text file")
    args = parser.parse_args()

    process_transcript(args.file)
