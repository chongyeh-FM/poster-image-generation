import streamlit as st
import numpy as np
from PIL import Image
import os
from rembg import remove
from openai import OpenAI
import requests
from io import BytesIO

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

def removebg(input_image):
    try:
        # Remove background from image
        image_without_bg = remove(input_image)
        return image_without_bg
    except Exception as e:
        st.error(f"An error occurred while removing the background: {e}")
        return None

def add_frame(image, frame):
    try:
        frame = frame.convert("RGBA")
        image = image.convert("RGBA")
        
        # Resize frame to match image size
        frame = frame.resize(image.size, Image.LANCZOS)
        
        # Composite the image and the frame
        combined = Image.alpha_composite(image, frame)
        return combined
    except Exception as e:
        st.error(f"An error occurred while adding the frame: {e}")
        return image

def object_detection_image():
    st.title('Poster Image Generation')
    st.subheader("Please scroll down to see the processed image.")

    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    prompt = st.text_input('Enter prompt for image generation', 'Create a anime theme background with Japanese cherry blossoms')
    
    # Load frame images
    frames = {
        'frame1.png': Image.open('frame1.png'),
        'frame2.png': Image.open('frame2.png'),
        'frame3.png': Image.open('frame3.png')
    }

    # Display radio buttons above each frame
    st.markdown("### Choose a frame:")
    frame_option = st.radio(
        "Select Frame",
        options=list(frames.keys()),
        format_func=lambda x: x.replace(".png", "").capitalize()
    )

    # Display frame images
    st.markdown("### Available Frames:")
    cols = st.columns(3)
    for idx, (frame_key, frame_img) in enumerate(frames.items()):
        with cols[idx]:
            st.image(frame_img, caption=frame_key, use_column_width=True)
    
    if file is not None:
        if 'prompt' not in st.session_state or st.session_state.prompt != prompt:
            st.session_state.prompt = prompt
            try:
                img1 = Image.open(file)
                st.image(img1, caption="Uploaded Image")
                my_bar = st.progress(0)

                # Remove background
                img_no_bg = removebg(img1)
                if img_no_bg is None:
                    return

                # Resize the image while maintaining the aspect ratio
                max_size = (1024, 1024)
                img_no_bg.thumbnail(max_size, Image.LANCZOS)

                # Create a new image with a white background
                new_image = Image.new("RGBA", max_size, (255, 255, 255, 0))
                new_image.paste(img_no_bg, ((max_size[0] - img_no_bg.size[0]) // 2, (max_size[1] - img_no_bg.size[1]) // 2))

                # Convert to RGBA to save as a PNG
                rgba_image = new_image.convert('RGBA')
                resized_image_path = "resized_image.png"
                rgba_image.save(resized_image_path, format="PNG")

                # Verify the file size
                image_size = os.path.getsize(resized_image_path)
                if image_size > 4 * 1024 * 1024:
                    raise ValueError("The image file size exceeds 4 MB")

                print(f"Resized image saved at {resized_image_path} with size {image_size} bytes.")

                # Use OpenAI to edit the image
                with open(resized_image_path, "rb") as image_file:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        n=1,
                        quality="hd",
                        size="1024x1024"
                    )

                image_url = response.data[0].url
                print("Edited image URL:", image_url)

                # Download the processed image
                response = requests.get(image_url)
                processed_image = Image.open(BytesIO(response.content))

                # Save the processed image to session state to avoid regenerating
                st.session_state.generated_image = processed_image

                my_bar.progress(100)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                my_bar.progress(0)
        else:
            processed_image = st.session_state.generated_image

    # If processed image exists in session state, allow frame application
    if 'generated_image' in st.session_state:
        processed_image = st.session_state.generated_image
        processed_image = add_frame(processed_image, Image.open("resized_image.png"))
        final_image = add_frame(processed_image, frames[frame_option])
        st.image(final_image, caption="Processed Image with Frame")

def main():
    st.markdown('<p style="font-size: 42px;">Welcome to Poster Image Generation App!</p>', unsafe_allow_html=True)
    st.markdown("""
        This project was built using Streamlit
        to demonstrate Poster Image Generation for food images.
    """)
    
    choice = st.sidebar.selectbox("MODE", ("About", "Image"))
    
    if choice == "Image":
        object_detection_image()
    elif choice == "About":
        st.markdown("""
            This app uses OpenAI DALL-E to generate background for food item images.
            Upload an image and see the prediction in action!
        """)

if __name__ == '__main__':
    main()
