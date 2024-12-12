import streamlit as st
import requests
import base64
from PIL import Image
import io
import zipfile
import os
import tempfile
import time
from datetime import datetime

# Page config
st.set_page_config(page_title="Flux Training Helper", layout="wide")

# Initialize session states
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

def clear_session():
    st.session_state.processed_files = set()
    st.session_state.current_batch = []
    st.session_state.current_page = 1
    st.session_state.processing_status = {}

# Function to generate T5 prompt from image or text
def generate_t5_prompt(input_data, input_type, api_key):
    if input_type == "text":
        prompt = f"""
        Create a detailed, 512-token prompt for T5 text-to-image model based on this description: {input_data}
        
        Consider and describe in detail:
        1. Composition and layout
        2. Subject matter details
        3. Setting and environment
        4. Lighting and atmosphere
        5. Color palette
        6. Style and technique
        7. Mood and emotion
        8. Technical aspects
        9. Textures and materials
        10. Perspective and viewpoint
        
        Make it detailed, cohesive, and flowing in a single paragraph.
        Focus on positive descriptions.
        Include specific visual details.
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000
        }

    else:  # input_type == "image"
        encoded_image = base64.b64encode(input_data).decode('utf-8')
        
        prompt = """
        Analyze this image and create a detailed, 512-token prompt for T5 text-to-image model.
        
        Describe in detail:
        1. Composition and layout
        2. Subject matter details
        3. Setting and environment
        4. Lighting and atmosphere
        5. Color palette
        6. Style and technique
        7. Mood and emotion
        8. Technical aspects
        9. Textures and materials
        10. Perspective and viewpoint
        
        Make it detailed, cohesive, and flowing in a single paragraph.
        Focus on positive descriptions.
        Include specific visual details.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Function to convert T5 prompt to CLIP prompt
def convert_to_clip(t5_prompt, api_key):
    prompt = f"""
    Convert this T5 prompt to a CLIP prompt of maximum 70 tokens:
    {t5_prompt}
    
    Guidelines:
    - Be specific and concise
    - Prioritize key elements
    - Use descriptive adjectives
    - Include artistic references if present
    - Include keywords like "detailed", "high quality", "4k"
    - Focus on what should be present
    - Maximum 70 tokens
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# [Previous LoRA captioning functions remain the same]
# Include all the functions from the previous version here

# Main app
st.title("Flux Training Helper")

# Mode selection
mode = st.radio("Select Mode", ["LoRA Captioning", "Prompt Optimization"])

if mode == "LoRA Captioning":
    # [Previous LoRA captioning code remains the same]
    # Include all the LoRA captioning code here
    pass

else:  # Prompt Optimization mode
    st.header("Prompt Optimization")
    
    # API key input
    api_key = st.text_input("OpenAI API Key", type="password")
    
    # Input method selection
    input_method = st.radio("Input Method", ["Text", "Image"])
    
    if input_method == "Text":
        user_input = st.text_area("Enter your prompt:", height=100)
        process_button = st.button("Generate Optimized Prompts")
        
        if process_button and api_key and user_input:
            with st.spinner("Generating prompts..."):
                try:
                    # Generate T5 prompt
                    t5_prompt = generate_t5_prompt(user_input, "text", api_key)
                    
                    # Convert to CLIP prompt
                    clip_prompt = convert_to_clip(t5_prompt, api_key)
                    
                    # Display results
                    st.subheader("T5 Prompt (512 tokens)")
                    st.text_area("", t5_prompt, height=200)
                    st.button("Copy T5 Prompt", key="copy_t5")
                    
                    st.subheader("CLIP Prompt (70 tokens)")
                    st.text_area("", clip_prompt, height=100)
                    st.button("Copy CLIP Prompt", key="copy_clip")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    else:  # Image input
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'webp'])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            process_button = st.button("Generate Optimized Prompts")
            
            if process_button and api_key:
                with st.spinner("Analyzing image and generating prompts..."):
                    try:
                        # Generate T5 prompt from image
                        image_bytes = uploaded_image.read()
                        t5_prompt = generate_t5_prompt(image_bytes, "image", api_key)
                        
                        # Convert to CLIP prompt
                        clip_prompt = convert_to_clip(t5_prompt, api_key)
                        
                        # Display results
                        st.subheader("T5 Prompt (512 tokens)")
                        st.text_area("", t5_prompt, height=200)
                        st.button("Copy T5 Prompt", key="copy_t5")
                        
                        st.subheader("CLIP Prompt (70 tokens)")
                        st.text_area("", clip_prompt, height=100)
                        st.button("Copy CLIP Prompt", key="copy_clip")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    st.sidebar.markdown("""
    ### Instructions
    1. Enter your OpenAI API key
    2. Choose input method (Text or Image)
    3. Enter prompt or upload image
    4. Click Generate to get optimized prompts
    5. Copy the prompts you need
    
    ### Note
    - T5 prompts are detailed (512 tokens)
    - CLIP prompts are concise (70 tokens)
    - Processing takes a few seconds
    """)
