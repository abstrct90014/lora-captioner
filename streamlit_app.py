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

# Sidebar for API key and settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    trigger_word = st.text_input("Trigger Word", placeholder="e.g., JenniePink")
    
    training_type = st.selectbox(
        "Training Type",
        ["Character", "Style", "Concept"],
        help="""
        Character: Specific person or character (real or animated)
        Style: Art style, time period, or aesthetic
        Concept: Products, objects, clothing, poses, etc.
        """
    )
    
    st.markdown("""
    ### Instructions
    1. Enter your OpenAI API key
    2. Set your trigger word
    3. Select training type
    4. Upload your images
    5. Wait for processing
    6. Download your captioned dataset
    
    ### Note
    - Processing takes 2-3 seconds per image
    - Cost is approximately $0.01-0.02 per image
    """)

def get_prompt_by_type(training_type):
    base_rules = """
    Rules:
    - Create ONE continuous line with elements separated by commas
    - Don't use bullet points or numbered lists
    - Don't use quotation marks
    - Be extremely detailed but keep it flowing
    - Focus on visual and technical aspects
    - Use precise, descriptive language
    - No subjective terms like beautiful/pretty
    """
    
    if training_type == "Character":
        return f"""
        Create a detailed, single-line caption focusing on the character's unique traits and appearance.
        
        Required elements in order:
        1. Character identity and demographic details
        2. Hair details (color, length, style, texture, parting)
        3. Facial features and expression
        4. Pose and body positioning
        5. Clothing and accessories specific to the character
        6. Environmental context and lighting
        7. Character's mood and presence
        
        {base_rules}
        
        Example format:
        young asian woman with distinctive features, long black hair sleek and straight parted in the middle, sharp facial features with defined cheekbones, front-facing pose with arms elegantly crossed in front, minimalistic black strapless outfit, adorned with signature gold jewelry, clean white background with professional lighting, poised and confident presence
        """
    
    elif training_type == "Style":
        return f"""
        Create a detailed, single-line caption focusing on the distinctive elements of the style.
        
        Required elements in order:
        1. Style category and era/origin
        2. Key stylistic elements and techniques
        3. Color palette and patterns
        4. Composition and arrangement
        5. Texture and material qualities
        6. Lighting and atmospheric elements
        7. Overall mood and aesthetic impact
        
        {base_rules}
        
        Example format:
        minimalist contemporary style, clean geometric lines with architectural influence, monochromatic palette dominated by stark blacks and whites, balanced asymmetrical composition, smooth matte surfaces contrasting with metallic accents, diffused studio lighting creating subtle shadows, sophisticated and refined aesthetic
        """
    
    else:  # Concept
        return f"""
        Create a detailed, single-line caption focusing on the primary concept/object.
        
        Required elements in order:
        1. Product/object identification and key features
        2. Materials, textures, and construction details
        3. Color and pattern specifics
        4. Positioning and presentation
        5. Supporting elements (model, props, etc.)
        6. Environmental context and lighting
        7. Overall presentation style
        
        {base_rules}
        
        Example format:
        classic white polo shirt with signature crocodile emblem, premium cotton pique fabric with ribbed collar and cuffs, pristine white colorway with tonal stitching, displayed on athletic male model front-facing stance, styled with dark jeans and minimal accessories, studio setting with soft directional lighting, clean and premium product presentation
        """

def generate_caption(image_bytes, api_key, trigger_word, training_type):
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = get_prompt_by_type(training_type)

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
        # Remove any quotation marks and ensure no numbered lists or bullet points
        caption = caption.replace('"', '').replace('"', '')
        # Remove any numbered list formatting if present
        caption = ' '.join([line.strip() for line in caption.split('\n')])
        # Remove any bullet points if present
        caption = caption.replace('- ', '')
        # Remove any numbering if present
        import re
        caption = re.sub(r'^\d+\.\s*', '', caption)
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
                    caption = generate_caption(image_bytes, api_key, trigger_word, training_type)
                    
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
                
                # Create zip file with trigger word in filename
                zip_filename = f"{trigger_word}_lora_dataset.zip"
                zip_path = os.path.join(temp_dir, zip_filename)
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for file in os.listdir(temp_dir):
                        if file != zip_filename:
                            file_path = os.path.join(temp_dir, file)
                            zf.write(file_path, file)
                
                # Read zip file for download
                with open(zip_path, "rb") as f:
                    zip_data = f.read()
                
                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()
                
                # Create download button with trigger word in filename
                st.success("Processing complete! Click below to download your dataset.")
                st.download_button(
                    label="Download Captioned Dataset",
                    data=zip_data,
                    file_name=zip_filename,
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
