import streamlit as st
import requests
import base64
from PIL import Image
import io
import zipfile
import os
import tempfile
import time

st.set_page_config(page_title="LoRA Image Captioner", layout="wide")

st.title("LoRA Image Captioner")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    trigger_word = st.text_input("Trigger Word", placeholder="e.g., JenniePink")
    
    st.markdown("""
    ### Instructions
    1. Enter your OpenAI API key
    2. Set your trigger word
    3. Upload your images
    4. Wait for processing
    5. Download your captioned dataset
    
    ### Note
    - Processing takes 2-3 seconds per image
    - Cost is approximately $0.01-0.02 per image
    """)

def generate_caption(image_bytes, api_key, trigger_word):
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
    Analyze this image and create a caption for LoRA training following this exact structure:
    1. Start with demographic details (age, ethnicity if visible)
    2. Describe hair color and length
    3. Always include "front-facing"
    4. Detail the outfit and any accessories
    5. Describe the environment/background
    6. End with mood or theme
    
    Rules:
    - Be concise but detailed
    - Use commas to separate elements
    - Avoid subjective terms like "beautiful" or "amazing"
    - Focus on clear visual traits
    - Keep descriptions professional and neutral
    
    Example format:
    "young asian woman, long black hair, front-facing, wearing a white blouse with lace details, standing in a sunlit garden with blooming flowers, serene and peaceful"
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
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
        "max_tokens": 500
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        caption = response.json()['choices'][0]['message']['content'].strip()
        return f"{trigger_word}, {caption}"
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Main upload section
uploaded_files = st.file_uploader(
    "Upload Images", 
    accept_multiple_files=True,
    type=['png', 'jpg', 'jpeg', 'webp']
)

if uploaded_files and api_key and trigger_word:
    if st.button("Process Images"):
        # Create progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                total_files = len(uploaded_files)
                
                # Process each image
                for idx, uploaded_file in enumerate(uploaded_files, 1):
                    # Update progress
                    progress_text.text(f"Processing image {idx}/{total_files}")
                    progress_bar.progress(idx/total_files)
                    
                    # Read image
                    image_bytes = uploaded_file.read()
                    
                    # Generate caption
                    caption = generate_caption(image_bytes, api_key, trigger_word)
                    
                    # Save files
                    base_name = f"{idx:04d}"
                    
                    # Save image
                    image_ext = os.path.splitext(uploaded_file.name)[1]
                    image_path = os.path.join(temp_dir, f"{base_name}{image_ext}")
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Save caption
                    txt_path = os.path.join(temp_dir, f"{base_name}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                
                # Create zip file
                zip_path = os.path.join(temp_dir, "lora_dataset.zip")
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for file in os.listdir(temp_dir):
                        if file != "lora_dataset.zip":
                            file_path = os.path.join(temp_dir, file)
                            zf.write(file_path, file)
                
                # Read zip file for download
                with open(zip_path, "rb") as f:
                    zip_data = f.read()
                
                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()
                
                # Create download button
                st.success("Processing complete! Click below to download your dataset.")
                st.download_button(
                    label="Download Captioned Dataset",
                    data=zip_data,
                    file_name="lora_dataset.zip",
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_text.empty()
            progress_bar.empty()
            
elif st.button("Process Images"):
    if not api_key:
        st.error("Please enter your OpenAI API key")
    if not trigger_word:
        st.error("Please enter a trigger word")
    if not uploaded_files:
        st.error("Please upload some images")
