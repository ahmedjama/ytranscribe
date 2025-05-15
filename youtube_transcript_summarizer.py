import sys
import requests
import re
import os
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

# Defaults
DEFAULT_OLLAMA_MODEL = "gemma3"
DEFAULT_PROMPT = (
    "You are an assistant summarizing a YouTube transcript. "
    "Please provide a concise summary of the video and list 5–7 key takeaways as bullet points. "
    "Here is the transcript:"
)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Folder paths
TRANSCRIPTS_DIR = "transcripts"
SUMMARIES_DIR = "summaries"

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

def get_video_title(video_url):
    try:
        with YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return sanitize_filename(info.get('title', 'unknown_title'))
    except Exception as e:
        print(f"⚠️ Failed to fetch video title: {e}")
        return "unknown_title"

def save_transcript(filepath, transcript):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for snippet in transcript:
            f.write(f"{snippet.text.strip()} ")
    print(f"📄 Transcript saved as {filepath}")

def load_prompt(prompt_file):
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Failed to load prompt file: {e}")
        print("Using default prompt instead.")
        return DEFAULT_PROMPT

def summarize_with_ollama(prompt, transcript_text, model):
    full_prompt = prompt + "\n\n" + transcript_text
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": full_prompt,
        "stream": False
    })

    if response.status_code == 200:
        result = response.json()
        return result.get("response", "").strip()
    else:
        raise RuntimeError(f"Ollama request failed: {response.text}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_transcript_summarizer.py <video_id> [prompt_file] [ollama_model]")
        sys.exit(1)

    video_id = sys.argv[1]
    prompt_file = sys.argv[2] if len(sys.argv) >= 3 else None
    ollama_model = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_OLLAMA_MODEL

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        print(f"🔍 Fetching title for video: {video_url}")
        video_title = get_video_title(video_url)
        print(f"🎬 Video Title: {video_title.replace('_', ' ')}")

        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_title}_transcript.txt")

        if os.path.exists(transcript_path):
            print(f"📁 Found existing transcript: {transcript_path}")
        else:
            print("⏳ Fetching transcript...")
            ytt_api = YouTubeTranscriptApi()
            fetched_transcript = ytt_api.fetch(video_id)
            print(f"✅ Retrieved {len(fetched_transcript)} transcript snippets.")
            save_transcript(transcript_path, fetched_transcript)

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        print("\n📥 Loading prompt...")
        prompt = load_prompt(prompt_file) if prompt_file else DEFAULT_PROMPT

        print(f"\n🧠 Sending transcript to Ollama ({ollama_model}) for summarization...")
        summary = summarize_with_ollama(prompt, transcript_text, ollama_model)
        print("\n📝 Summary:\n")
        print(summary)

        summary_path = os.path.join(SUMMARIES_DIR, f"{video_title}_summary_{ollama_model}.txt")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n✅ Summary saved as {summary_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
