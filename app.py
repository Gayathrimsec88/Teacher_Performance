# teacher_performance_app.py
import os
import torch
import numpy as np
import librosa
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tempfile
import base64
import time
import logging
import pandas as pd
import streamlit as st
from datetime import datetime

# Set up the app
st.set_page_config(page_title="Teacher Performance Analyzer", layout="wide")

# Initialize models (similar to your Flask app)
@st.cache_resource
def load_models():
    try:
        logger = logging.getLogger(__name__)
        logger.info("Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda' if torch.cuda.is_available() else 'cpu')
        model.config.forced_decoder_ids = None
        logger.info("Whisper model loaded successfully!")
        
        logger.info("Loading sentiment analysis model...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        return processor, model, sentiment_analyzer
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        raise

# Load models
whisper_processor, whisper_model, sentiment_analyzer = load_models()

# Define your AudioAnalysisModel class (same as before)
class AudioAnalysisModel(nn.Module):
    def __init__(self, mfcc_dim=40, text_dim=768, hidden_dim=256):
        super().__init__()
        self.mfcc_proj = nn.Linear(mfcc_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
            nn.Sigmoid()
        )
        
    def forward(self, mfcc_features, text_features):
        mfcc = self.mfcc_proj(mfcc_features)
        text = self.text_proj(text_features)
        combined = (mfcc + text) / 2
        return self.output(combined) * 10

model = AudioAnalysisModel().to('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit UI
st.title("AI Teacher Performance Analysis System")
st.write("Upload an audio recording of a teaching session for analysis")

# File uploader
audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'm4a', 'ogg', 'webm'])
teacher_id = st.text_input("Teacher ID", "1")

if audio_file is not None:
    if st.button("Analyze"):
        with st.spinner("Processing audio..."):
            try:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
                    tmp.write(audio_file.getvalue())
                    temp_path = tmp.name
                
                # Process audio (similar to your Flask version)
                y, sr = librosa.load(temp_path, sr=16000)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                mfcc_features = np.mean(mfccs, axis=1)
                mfcc_features = (mfcc_features - np.mean(mfcc_features)) / (np.std(mfcc_features) + 1e-8)
                
                # Generate transcription
                input_features = whisper_processor(y, sampling_rate=16000, return_tensors="pt").input_features.to('cuda' if torch.cuda.is_available() else 'cpu')
                predicted_ids = whisper_model.generate(input_features, task="transcribe", language=None)
                transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                # Language detection and sentiment analysis
                language = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
                language = language.split("<|")[1].split("|>")[0] if "|>" in language else "unknown"
                
                sentiment_score = 5.0  # Default
                if language == 'en':
                    try:
                        chunks = [transcription[i:i+500] for i in range(0, len(transcription), 500)]
                        results = [sentiment_analyzer(chunk)[0] for chunk in chunks if chunk.strip()]
                        if results:
                            avg_score = sum(r['score'] for r in results) / len(results)
                            sentiment_score = avg_score * 10 if results[0]['label'] == 'POSITIVE' else (1 - avg_score) * 10
                    except Exception as e:
                        st.warning(f"Sentiment analysis failed: {str(e)}")
                
                # Get scores from your model
                mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
                text_embedding = torch.randn(1, 768, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                with torch.no_grad():
                    outputs = model(mfcc_tensor, text_embedding)
                    speech_score = outputs[0, 0].item()
                    engagement_score = outputs[0, 1].item()
                
                # Calculate final score
                final_score = (speech_score + sentiment_score + engagement_score) / 3
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Speech Score", f"{speech_score:.1f}/10")
                    st.metric("Sentiment Score", f"{sentiment_score:.1f}/10")
                    st.metric("Engagement Score", f"{engagement_score:.1f}/10")
                    st.metric("Overall Score", f"{final_score:.1f}/10")
                
                with col2:
                    # Performance level
                    if final_score >= 8: level = "Excellent"
                    elif final_score >= 7: level = "Very Good"
                    elif final_score >= 6: level = "Good"
                    elif final_score >= 5: level = "Average"
                    else: level = "Needs Improvement"
                    
                    st.metric("Performance Level", level)
                    st.text_area("Transcription", transcription, height=200)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics = ['Speech', 'Sentiment', 'Engagement', 'Overall']
                scores = [speech_score, sentiment_score, engagement_score, final_score]
                colors = ['#4285f4', '#34a853', '#fbbc05', '#ea4335']
                ax.bar(metrics, scores, color=colors)
                ax.set_ylim(0, 10)
                ax.set_title('Teaching Performance Metrics')
                ax.set_ylabel('Score (0-10)')
                st.pyplot(fig)
                
                # Suggestions
                st.subheader("Suggestions for Improvement")
                if speech_score < 6:
                    st.info("🗣️ **Speech Clarity**: Practice clearer enunciation and pacing")
                if sentiment_score < 6 and language == 'en':
                    st.info("😊 **Emotional Expression**: Use more varied vocal tones")
                if engagement_score < 6:
                    st.info("🎯 **Engagement**: Add more interactive elements to your teaching")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)