# Nigerian Text-to-Speech API

This FastAPI backend generates Nigerian-accented speech using YarnGPT. It supports multiple Nigerian languages and various voices.

## Features

- Text-to-speech conversion with Nigerian accents
- Supports English, Yoruba, Igbo, and Hausa
- Multiple male and female voices
- Base64 encoded audio responses
- Direct audio streaming

## API Endpoints

### `GET /`
Returns API status and available voices/languages.

### `POST /tts`
Converts text to speech.

**Request:**
```json
{
  "text": "Hello, this is a test.",
  "language": "english",
  "voice": "idera"
}