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
import pyperclip

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
if 't5_prompt' not in st.session_state:
    st.session_state.t5_prompt = ""
if 'clip_prompt' not in st.session_state:
    st.session_state.clip_prompt = ""

# Helper functions
def clear_session():
    st.session_state.processed_files = set()
    st.session_state.current_batch = []
    st.session_state.current_page = 1
    st.session_state.processing_status = {}
    st.session_state.t5_prompt = ""
    st.session_state.clip_prompt = ""

def copy_to_clipboard(text, button_name):
    if st.button(button_name):
        pyperclip.copy(text)
        st.success(f"✅ {button_name} copied to clipboard!")

def display_prompts():
    if st.session_state.t5_prompt or st.session_state.clip_prompt:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("T5 Prompt (512 tokens)")
            st.text_area("", st.session_state.t5_prompt, height=200, key="t5_display")
            copy_to_clipboard(st.session_state.t5_prompt, "Copy T5 Prompt")
        
        with col2:
            st.subheader("CLIP Prompt (70 tokens)")
            st.text_area("", st.session_state.clip_prompt, height=100, key="clip_display")
            copy_to_clipboard(st.session_state.clip_prompt, "Copy CLIP Prompt")

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

def generate_t5_prompt(input_data, input_type, api_key):
    if input_type == "text":
        prompt = """
        Generate a detailed, 512-token T5 prompt describing:
        1. Composition and layout
        2. Subject matter details
        3. Setting and environment
        4. Lighting and atmosphere
        5. Color palette
        6. Style and technique
        7. Mood and emotion
        
        Make it detailed, cohesive, and flowing in a single paragraph.
        """
        content = prompt + "\n\nDescription to convert: " + input_data
        
    else:  # input_type == "image"
        encoded_image = base64.b64encode(input_data).decode('utf-8')
        prompt = """
        Create a detailed, 512-token T5 prompt based on this image.
        Describe composition, subject, setting, lighting, colors, style, and mood.
        Make it detailed, cohesive, and flowing in a single paragraph.
        """
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            }
        ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
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

def convert_to_clip(t5_prompt, api_key):
    prompt = """
    Convert this T5 prompt to a CLIP prompt (max 70 tokens).
    Be specific, concise, and include key elements only.
    Add "detailed", "high quality", "4k" where appropriate.
    
    Original prompt:
    """ + t5_prompt

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
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

# Main app interface
st.title("Flux Training Helper")

# Mode selection
mode = st.radio("Select Mode", ["LoRA Captioning", "Prompt Optimization"])

if mode == "LoRA Captioning":
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        trigger_word = st.text_input("Trigger Word", placeholder="e.g., JenniePink")
        
        training_type = st.selectbox(
            "Training Type",
            ["Character", "Style", "Concept"],
            help="""
            Character: Specific person or character
            Style: Art style, time period, aesthetic
            Concept: Products, objects, poses
            """
        )
        
        if st.button("Clear All"):
            clear_session()
            st.experimental_rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        new_files = st.file_uploader(
            "Upload Images", 
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'webp']
        )

        if new_files:
            st.session_state.current_batch = []
            for file in new_files:
                file_hash = hash(file.name + str(file.size))
                if file_hash not in st.session_state.processed_files:
                    st.session_state.current_batch.append({
                        'file': file,
                        'hash': file_hash
                    })

        # Process files
        if st.session_state.current_batch and api_key and trigger_word:
            if st.button("Process Images"):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        total_files = len(st.session_state.current_batch)
                        processed_files = []
                        
                        for idx, file_info in enumerate(st.session_state.current_batch, 1):
                            file = file_info['file']
                            file_hash = file_info['hash']
                            
                            progress_text.text(f"Processing image {idx}/{total_files}")
                            progress_bar.progress(idx/total_files)
                            
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
                            
                            processed_files.append(file_hash)
                            st.session_state.processed_files.add(file_hash)
                        
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
                        
                        # Clear progress indicators
                        progress_text.empty()
                        progress_bar.empty()
                        
                        # Show download button
                        st.success("Processing complete!")
                        st.download_button(
                            "Download Dataset",
                            data=zip_data,
                            file_name=zip_filename,
                            mime="application/zip"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

else:  # Prompt Optimization mode
    st.header("Prompt Optimization")
    
    # API key input
    api_key = st.text_input("OpenAI API Key", type="password")
    
    # Input method selection
    input_method = st.radio("Input Method", ["Text", "Image"])
    
    if input_method == "Text":
        user_input = st.text_area("Enter your prompt:", height=100)
        if st.button("Generate Prompts") and api_key and user_input:
            with st.spinner("Generating prompts..."):
                try:
                    st.session_state.t5_prompt = generate_t5_prompt(user_input, "text", api_key)
                    st.session_state.clip_prompt = convert_to_clip(st.session_state.t5_prompt, api_key)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            
            display_prompts()
    
    else:  # Image input
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'webp'])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            if st.button("Generate Prompts") and api_key:
                with st.spinner("Analyzing image and generating prompts..."):
                    try:
                        image_bytes = uploaded_image.read()
                        st.session_state.t5_prompt = generate_t5_prompt(image_bytes, "image", api_key)
                        st.session_state.clip_prompt = convert_to_clip(st.session_state.t5_prompt, api_key)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                
                display_prompts()
