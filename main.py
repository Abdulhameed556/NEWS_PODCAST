from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import torch
import torchaudio
import base64
from io import BytesIO
from transformers import AutoModelForCausalLM
import sys
import subprocess
from datetime import datetime, timedelta

app = FastAPI(title="Nigerian TTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize necessary directories
os.makedirs("audio_files", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Check if YarnGPT is installed, if not install it
try:
    import yarngpt
    from yarngpt.audiotokenizer import AudioTokenizerV2
except ImportError:
    print("Installing YarnGPT and dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/saheedniyi02/yarngpt.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "outetts", "uroman", "transformers", "torchaudio"])
    from yarngpt.audiotokenizer import AudioTokenizerV2

# Model configuration
tokenizer_path = "saheedniyi/YarnGPT2"

# Check if model files exist, if not download them
wav_tokenizer_config_path = "./models/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
wav_tokenizer_model_path = "./models/wavtokenizer_large_speech_320_24k.ckpt"

if not os.path.exists(wav_tokenizer_config_path):
    print("Downloading model config file...")
    subprocess.check_call([
        "wget", "-O", wav_tokenizer_config_path,
        "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    ])

if not os.path.exists(wav_tokenizer_model_path):
    print("Downloading model checkpoint file...")
    subprocess.check_call([
        "wget", "-O", wav_tokenizer_model_path,
        "https://drive.google.com/uc?id=1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt&export=download"
    ])

print("Loading YarnGPT model and tokenizer...")
audio_tokenizer = AudioTokenizerV2(
    tokenizer_path, wav_tokenizer_model_path, wav_tokenizer_config_path
)
model = AutoModelForCausalLM.from_pretrained(tokenizer_path, torch_dtype="auto").to(audio_tokenizer.device)
print("Model loaded successfully!")

# Available voices and languages
AVAILABLE_VOICES = {
    "female": ["zainab", "idera", "regina", "chinenye", "joke", "remi"],
    "male": ["jude", "tayo", "umar", "osagie", "onye", "emma"]
}
AVAILABLE_LANGUAGES = ["english", "yoruba", "igbo", "hausa"]

# Input validation model
class TTSRequest(BaseModel):
    text: str
    language: str = "english"
    voice: str = "idera"
    
# Output model with base64-encoded audio
class TTSResponse(BaseModel):
    audio_base64: str  # Base64-encoded audio data
    audio_url: str     # Keep for backward compatibility
    text: str
    voice: str
    language: str

@app.get("/")
async def root():
    """API health check and info"""
    return {
        "status": "ok",
        "message": "Nigerian TTS API is running",
        "available_languages": AVAILABLE_LANGUAGES,
        "available_voices": AVAILABLE_VOICES
    }


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Convert text to Nigerian-accented speech"""
    
    # Validate inputs
    if request.language not in AVAILABLE_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language must be one of {AVAILABLE_LANGUAGES}")
    
    all_voices = AVAILABLE_VOICES["female"] + AVAILABLE_VOICES["male"]
    if request.voice not in all_voices:
        raise HTTPException(status_code=400, detail=f"Voice must be one of {all_voices}")
    
    # Generate unique filename
    audio_id = str(uuid.uuid4())
    output_path = f"audio_files/{audio_id}.wav"
    
    try:
        # Create prompt and generate audio
        prompt = audio_tokenizer.create_prompt(request.text, lang=request.language, speaker_name=request.voice)
        input_ids = audio_tokenizer.tokenize_prompt(prompt)
        
        output = model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000,
        )
        
        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)
        
        # Save audio file
        torchaudio.save(output_path, audio, sample_rate=24000)
        
        # Read the file and encode as base64
        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Clean up old files after a while
        background_tasks.add_task(cleanup_old_files)
        
        return TTSResponse(
            audio_base64=audio_base64,
            audio_url=f"/audio/{audio_id}.wav",  # Keep for compatibility
            text=request.text,
            voice=request.voice,
            language=request.language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

# File serving endpoint for direct audio access
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = f"audio_files/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    def iterfile():
        with open(file_path, "rb") as audio_file:
            yield from audio_file
    
    return StreamingResponse(iterfile(), media_type="audio/wav")

# Endpoint to stream audio directly from base64 (useful for debugging)
@app.post("/stream-audio")
async def stream_audio(request: TTSRequest):
    """Stream audio directly without saving to disk"""
    try:
        # Create prompt and generate audio
        prompt = audio_tokenizer.create_prompt(request.text, lang=request.language, speaker_name=request.voice)
        input_ids = audio_tokenizer.tokenize_prompt(prompt)
        
        output = model.generate(
            input_ids=input_ids,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4000,
        )
        
        codes = audio_tokenizer.get_codes(output)
        audio = audio_tokenizer.get_audio(codes)
        
        # Create BytesIO object
        buffer = BytesIO()
        torchaudio.save(buffer, audio, sample_rate=24000, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

# Cleanup function to remove old files
def cleanup_old_files():
    """Delete audio files older than 6 hours to manage disk space"""
    try:
        now = datetime.now()
        audio_dir = "audio_files"
        
        for filename in os.listdir(audio_dir):
            if not filename.endswith(".wav"):
                continue
                
            file_path = os.path.join(audio_dir, filename)
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Delete files older than 6 hours
            if now - file_mod_time > timedelta(hours=6):
                os.remove(file_path)
                print(f"Deleted old audio file: {filename}")
    except Exception as e:
        print(f"Error cleaning up old files: {e}")

# For running locally with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)