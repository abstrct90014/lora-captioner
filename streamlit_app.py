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
st.set_page_config(page_title="LoRA Image Captioner", layout="wide")

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()  # Just store processed file hashes
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = []  # Store current batch of files to process
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}  # Store processing status and progress

def clear_session():
    """Clear all session state"""
    st.session_state.processed_files = set()
    st.session_state.current_batch = []
    st.session_state.current_page = 1
    st.session_state.processing_status = {}

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
        caption = caption.replace('"', '').replace('"', '')
        caption = ' '.join([line.strip() for line in caption.split('\n')])
        caption = caption.replace('- ', '')
        import re
        caption = re.sub(r'^\d+\.\s*', '', caption)
        return f"{trigger_word}, {caption}"
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Main app title
st.title("LoRA Image Captioner")

# Sidebar
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
    
    if st.button("Clear All"):
        clear_session()
        st.experimental_rerun()
    
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

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader
    new_files = st.file_uploader(
        "Upload Images", 
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg', 'webp'],
        key="file_uploader"
    )

    if new_files:
        # Add only new files to current batch
        st.session_state.current_batch = []
        for file in new_files:
            file_hash = hash(file.name + str(file.size))
            if file_hash not in st.session_state.processed_files:
                st.session_state.current_batch.append({
                    'file': file,
                    'hash': file_hash
                })

    # Display file status
    if st.session_state.current_batch or st.session_state.processing_status:
        st.subheader("Current Batch Status")
        
        # Pagination
        items_per_page = 15
        total_files = len(st.session_state.current_batch)
        total_pages = max(1, (total_files - 1) // items_per_page + 1)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("← Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col2:
            st.write(f"Page {st.session_state.current_page} of {total_pages}")
        with col3:
            if st.button("Next →") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1

        # Display current batch files
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        for file_info in st.session_state.current_batch[start_idx:end_idx]:
            status = st.session_state.processing_status.get(file_info['hash'], {'status': 'queued', 'progress': 0})
            col1, col2, col3 = st.columns([3, 6, 2])
            with col1:
                st.text(file_info['file'].name)
            with col2:
                st.progress(status['progress'] / 100)
            with col3:
                status_color = {
                    'queued': '🟡',
                    'processing': '🔵',
                    'completed': '🟢',
                    'error': '🔴'
                }
                st.write(f"{status_color.get(status['status'], '⚪')} {status['status']}")

with col2:
    if st.session_state.current_batch and api_key and trigger_word:
        if st.button("Process Current Batch"):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    total_files = len(st.session_state.current_batch)
                    
                    for idx, file_info in enumerate(st.session_state.current_batch, 1):
                        file = file_info['file']
                        file_hash = file_info['hash']
                        
                        try:
                            # Update status
                            st.session_state.processing_status[file_hash] = {
                                'status': 'processing',
                                'progress': (idx-1)/total_files * 100
                            }
                            
                            # Process file
                            image_bytes = file.read()
                            caption = generate_caption(image_bytes, api_key, trigger_word, training_type)
                            
                            # Save files
                            base_name = f"{idx:04d}"
                            image_ext = os.path.splitext(file.name)[1]
                            image_path = os.path.join(temp_dir, f"{base_name}{image_ext}")
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                            
                            txt_path = os.path.join(temp_dir, f"{base_name}.txt")
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(caption)
                            
                            # Update status
                            st.session_state.processing_status[file_hash] = {
                                'status': 'completed',
                                'progress': 100
                            }
                            st.session_state.processed_files.add(file_hash)
                            
                        except Exception as e:
                            st.session_state.processing_status[file_hash] = {
                                'status': 'error',
                                'progress': 0
                            }
                            st.error(f"Error processing {file.name}: {str(e)}")
                            continue
                        
                        progress_text.text(f"Processing image {idx}/{total_files}")
                        progress_bar.progress(idx/total_files)
                    
                    # Create zip file
                    zip_filename = f"{trigger_word}_lora_dataset.zip"
                    zip_path = os.path.join(temp_dir, zip_filename)
                    with zipfile.ZipFile(zip_path, "w") as zf:
                        for file in os.listdir(temp_dir):
                            if file != zip_filename:
                                file_path = os.path.join(temp_dir, file)
                                zf.write(file_path, file)
                    
                    # Prepare download
                    with open(zip_path, "rb") as f:
                        zip_data = f.read()
                    
                    # Clear current batch
                    st.session_state.current_batch = []
                    
                    # Clear progress indicators
                    progress_text.empty()
                    progress_bar.empty()
                    
                    # Show download button
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

# Error handling
if st.button("Process Images") and not (api_key and trigger_word and st.session_state.current_batch):
    if not api_key:
        st.error("Please enter your OpenAI API key")
    if not trigger_word:
        st.error("Please enter a trigger word")
    if not st.session_state.current_batch:
        st.error("Please upload some images")
