import streamlit as st
import requests
import base64
from io import BytesIO
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Nigerian Text-to-Speech",
    page_icon="🎙️",
    layout="wide"
)

# Define the available voices and languages
AVAILABLE_VOICES = {
    "Female": ["zainab", "idera", "regina", "chinenye", "joke", "remi"],
    "Male": ["jude", "tayo", "umar", "osagie", "onye", "emma"]
}
AVAILABLE_LANGUAGES = ["english", "yoruba", "igbo", "hausa"]

# IMPORTANT: Replace this with the ngrok URL shown in your Colab notebook
# Example: API_BASE_URL = "https://a1b2-34-56-78-90.ngrok.io"
API_BASE_URL = st.text_input(
    "Enter the ngrok URL from Colab (e.g., https://a1b2-34-56-78-90.ngrok.io)",
    value="",
    key="api_url"
)

# Derive the TTS endpoint from the base URL
if API_BASE_URL:
    API_TTS_ENDPOINT = f"{API_BASE_URL}/tts"
    
    # Test connection to backend
    try:
        health_check = requests.get(f"{API_BASE_URL}")
        if health_check.status_code == 200:
            st.success(f"✅ Connected to backend API successfully!")
        else:
            st.warning(f"⚠️ Backend API returned status code {health_check.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to backend API: {str(e)}")
else:
    st.warning("⚠️ Please enter the ngrok URL from your Colab notebook to continue")

# App title and description
st.title("Nigerian Text-to-Speech")
st.markdown("""
Convert text to speech with authentic Nigerian accents. This app uses YarnGPT, a text-to-speech model 
that generates natural Nigerian-accented speech in English, Yoruba, Igbo, and Hausa.
""")

# Create tabs for different functions
tab1, tab2, tab3 = st.tabs(["Basic TTS", "Batch Processing", "About"])

# Tab 1: Basic TTS
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input
        text_input = st.text_area(
            "Enter text to convert to speech",
            "Welcome to Nigeria, the giant of Africa. Our diverse cultures and languages make us unique.",
            height=150
        )
        
        # Generate button
        generate_button = st.button("Generate Audio", type="primary", disabled=not API_BASE_URL)
    
    with col2:
        # Options
        language = st.selectbox("Language", AVAILABLE_LANGUAGES)
        
        gender = st.radio("Gender", ["Female", "Male"])
        voice = st.selectbox("Voice", AVAILABLE_VOICES[gender])
        
        st.info(f"Selected voice: **{voice}** ({gender.lower()})")

    # Generate audio when button is clicked
    if generate_button and text_input and API_BASE_URL:
        with st.spinner("Generating audio... (This may take a minute as the audio is processed through Colab)"):
            try:
                # Call the API with timeout increased
                response = requests.post(
                    API_TTS_ENDPOINT,
                    json={"text": text_input, "language": language, "voice": voice},
                    timeout=100000  # Increase timeout to 2 minutes
                )
                
                if response.status_code == 200:
                    # Get response data
                    audio_data = response.json()
                    
                    # Save info in session state
                    st.session_state.last_text = text_input
                    st.session_state.last_voice = voice
                    st.session_state.last_language = language
                    
                    # Display success and audio player
                    st.success("Audio generated successfully!")
                    st.markdown(f"Voice: **{voice}** | Language: **{language}**")
                    
                    # Handle base64-encoded audio
                    if "audio_base64" in audio_data:
                        audio_bytes = base64.b64decode(audio_data["audio_base64"])
                        audio_stream = BytesIO(audio_bytes)
                        
                        # Play audio directly from the stream
                        st.audio(audio_stream, format="audio/wav")
                    else:
                        # Fall back to URL method (legacy support)
                        audio_url = f"{API_BASE_URL}{audio_data['audio_url']}"
                        st.warning("Using legacy URL-based audio (may not work)")
                        st.code(audio_url, language="text")
                        st.audio(audio_url, format="audio/wav")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")
                st.info(f"Make sure the backend API is running and accessible at {API_BASE_URL}")

# Tab 2: Batch Processing
with tab2:
    st.header("Batch Text-to-Speech Conversion")
    st.markdown("""
    Process multiple text entries at once. Upload a CSV file with the following columns:
    - `text`: The text to convert to speech
    - `language` (optional): Language for the text (english, yoruba, igbo, hausa)
    - `voice` (optional): Voice name to use
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file and API_BASE_URL:
        # Process the file
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV file must contain a 'text' column")
            else:
                st.dataframe(df.head())
                
                # Default values
                default_language = st.selectbox("Default language", AVAILABLE_LANGUAGES)
                default_voice = st.selectbox("Default voice", AVAILABLE_VOICES["Female"] + AVAILABLE_VOICES["Male"])
                
                if st.button("Process Batch", disabled=not API_BASE_URL):
                    # Create a container for audio files
                    audio_container = st.container()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each row
                    results = []
                    audio_files = []  # Store audio data for playback
                    
                    for i, row in enumerate(df.itertuples()):
                        # Update progress
                        progress = int((i + 1) / len(df) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing item {i+1} of {len(df)}...")
                        
                        # Get text and parameters
                        text = row.text
                        lang = getattr(row, 'language', default_language) if hasattr(row, 'language') else default_language
                        voice_name = getattr(row, 'voice', default_voice) if hasattr(row, 'voice') else default_voice
                        
                        try:
                            # Make API call with increased timeout
                            response = requests.post(
                                API_TTS_ENDPOINT,
                                json={"text": text, "language": lang, "voice": voice_name},
                                timeout=120  # Increase timeout to 2 minutes
                            )
                            
                            if response.status_code == 200:
                                audio_data = response.json()
                                
                                # Handle base64-encoded audio
                                if "audio_base64" in audio_data:
                                    audio_bytes = base64.b64decode(audio_data["audio_base64"])
                                    audio_files.append({
                                        "index": i,
                                        "bytes": audio_bytes,
                                        "text": text,
                                        "voice": voice_name,
                                        "language": lang
                                    })
                                    
                                    status = "Success"
                                else:
                                    # Fall back to URL method (legacy support)
                                    audio_url = f"{API_BASE_URL}{audio_data['audio_url']}"
                                    status = "Success (URL mode)"
                                
                                # Add to results
                                results.append({
                                    "text": text[:50] + "..." if len(text) > 50 else text,
                                    "language": lang,
                                    "voice": voice_name,
                                    "status": status
                                })
                            else:
                                results.append({
                                    "text": text[:50] + "..." if len(text) > 50 else text,
                                    "language": lang,
                                    "voice": voice_name,
                                    "status": f"Error: {response.status_code}"
                                })
                        except Exception as e:
                            results.append({
                                "text": text[:50] + "..." if len(text) > 50 else text,
                                "language": lang,
                                "voice": voice_name,
                                "status": f"Error: {str(e)}"
                            })
                    
                    # Show results
                    st.success("Batch processing completed!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Display audio players for successful generations
                    with audio_container:
                        st.subheader("Generated Audio Files")
                        for audio_item in audio_files:
                            st.markdown(f"**{audio_item['index']+1}. {audio_item['text'][:50]}...** ({audio_item['voice']}, {audio_item['language']})")
                            audio_stream = BytesIO(audio_item["bytes"])
                            st.audio(audio_stream, format="audio/wav")
                            st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    elif not API_BASE_URL:
        st.warning("Please enter the ngrok URL first to enable batch processing")

# Tab 3: About
with tab3:
    st.header("About YarnGPT")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Features
        - 🗣️ 12 preset voices (6 male, 6 female)
        - 🎯 Trained on 2000+ hours of Nigerian audio
        - 🔊 24kHz high-quality audio output
        - 📝 Support for long-form text
        
        ### Model Details
        - Base: HuggingFaceTB/SmolLM2-360M
        - Training: 5 epochs on A100 GPU
        - Data: Nigerian movies, podcasts, and open-source audio
        """)
    
    with col2:
        st.markdown("""
        ### Available Voices
        - **Female**: zainab, idera, regina, chinenye, joke, remi
        - **Male**: jude, tayo, umar, osagie, onye, emma
        
        ### Limitations
        - English to Nigerian-accented English primarily
        - May not capture all Nigerian accent variations
        - Training data includes auto-generated content
        """)
    
    st.markdown("""
    ### Credits
    - YarnGPT was created by Saheed Abdulrahman, a Unilag student
    - Model is available as open source on [GitHub](https://github.com/saheedniyi02/yarngpt)
    - Web demo: [https://yarngpt.co/](https://yarngpt.co/)
    """)

# Footer
st.markdown("---")
st.markdown("Developed for a Nigerian News App Podcaster API | Powered by YarnGPT")