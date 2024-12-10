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
    trigger_word = st.text_input("Trigger Word", placeholder="e.g., JenniePink, RedCar, AnimeStyle")
    
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
    Create an extremely detailed caption for LoRA training, analyzing every aspect of the image. Follow this structure:

    1. Main Subject Identification:
       - Type (person, object, style, character, etc.)
       - Primary distinguishing features
       - Position and orientation in frame
    
    2. Visual Details (Based on subject type):
    
    For People/Characters:
       - Hair: color, length, style, texture, parting, movement
       - Face: expression, features, makeup, angles
       - Pose: body position, gesture, interaction with camera
       - Clothing: style, fit, materials, colors, patterns
       - Accessories: jewelry, props, additional elements
    
    For Objects/Products:
       - Shape and form
       - Materials and textures
       - Colors and patterns
       - Design elements
       - Functional features
       - Scale and proportion
    
    For Styles/Artistic Elements:
       - Artistic techniques
       - Color schemes
       - Patterns and motifs
       - Stylistic influences
       - Unique characteristics
    
    3. Environmental Elements:
       - Background description
       - Setting context
       - Lighting conditions and effects
       - Shadows and highlights
       - Depth and perspective
    
    4. Technical Aspects:
       - Camera angle
       - Shot type (close-up, full body, etc.)
       - Composition elements
       - Focus points
       - Image quality characteristics
    
    5. Mood and Atmosphere:
       - Overall feeling
       - Emotional tone
       - Stylistic mood
       - Environmental atmosphere
    
    Rules:
    - Use commas to separate elements
    - Be extremely detailed and specific
    - Focus on objective, visual characteristics
    - Include all relevant technical details
    - Describe lighting and atmospheric effects
    - Be precise with color descriptions
    - Don't use quotation marks
    - Avoid subjective terms (beautiful, pretty, etc.)
    - Use technical and descriptive language
    - Maintain consistent detail level throughout
    
    Example formats:

    Person: long black hair sleek and straight center-parted, front-facing pose with deliberate hand placement, structured black garment with architectural details, precise lighting highlighting facial contours, studio setting with gradient background, professional editorial atmosphere

    Object: metallic red sports car, aggressive front-facing stance, carbon fiber hood with prominent air intakes, low-profile design with aerodynamic elements, showroom lighting creating reflective highlights, modern industrial setting, dynamic and powerful presence

    Style: vibrant anime-inspired artwork, bold cel-shading technique, exaggerated facial features with large expressive eyes, dynamic action pose with motion lines, saturated color palette with strong contrasts, detailed background with speed effects, energetic and dramatic mood
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
        caption = response.json()['choices'][0]['message']['content'].strip()
        # Remove any quotation marks from the caption
        caption = caption.replace('"', '').replace('"', '')
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
