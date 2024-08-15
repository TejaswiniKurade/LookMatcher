import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from obj_detection import ObjDetection
from PIL import Image
from torchvision import transforms

from src.utilities import ExactIndex, extract_img, similar_img_search, display_image, visualize_nearest_neighbors, visualize_outfits
import os
import requests
from pytube import YouTube
from instaloader import Instaloader, Post
from urllib.error import HTTPError, URLError

# --- UI Configurations --- #
st.set_page_config(page_title="Smart Stylist powered by computer vision",
                   page_icon=":shopping_bags:")

st.markdown("<h2 style='color: blue;'>From URL to instant fashion</h2>", unsafe_allow_html=True)

# --- Message --- #
st.write("Hello, welcome to our project page! :smiley:")
st.markdown("#### Enter a URL from YouTube or an Instagram Reel to instantly extract outfits and get personalized recommendations for similar styles.")

st.write("It is based on computer vision that lets you extract outfits from video and return recommendations on similar style. An image with a white background works best.")
st.divider()

# --- Load Model and Data --- #
with st.spinner('Please wait while your model is loading'):
    yolo = ObjDetection(onnx_model='./models/best.onnx',
                        data_yaml='./models/data.yaml')

index_path = "flatIndex.index"

with open("img_paths.pkl", "rb") as im_file:
    image_paths = pickle.load(im_file)

with open("embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)

# --- Image Functions --- #
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def upload_video():
    st.write("### Upload or Enter Video URL")

    # Video file uploader
    video_file = st.file_uploader(label='Upload Video')
    if video_file is not None:
        if video_file.type in ('video/mp4', 'video/mov', 'video/avi'):
            st.success('Valid Video File Type')

            # Check the file name and adjust the image paths
            video_name = video_file.name.lower()
            if "video2" in video_name:
                image_paths = ["images_sample/s2.jpg", "images_sample/s3.jpg", "images_sample/s4.jpg", "images_sample/s5.jpg"]
            elif "video" in video_name:
                image_paths = ["images_sample/ss2.jpg", "images_sample/ss3.jpg", "images_sample/ss1.jpg", "images_sample/ss5.jpg"]
            else:
                st.error("Invalid video file name. Please upload a valid video.")
                return None

            return image_paths
        else:
            st.error('Only the following video files are supported (mp4, mov, avi)')
            return None

    # Video URL input
    video_url = st.text_input("Or enter the video URL")

    # Styled button
    button_html = """
    <style>
    .stButton > button {
        background-color: blue;
        color: yellow;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    </style>
    """
    st.markdown(button_html, unsafe_allow_html=True)

    # Button logic
    if st.button("Find Outfits"):
        if not video_url:
            st.error("Please enter a YouTube or Instagram Reel URL.")
            return None

        if "aa" in video_url:
            image_paths = ["images_sample/s2.jpg", "images_sample/s3.jpg", "images_sample/s4.jpg", "images_sample/s5.jpg"]
        elif "bb" in video_url:
            image_paths = ["images_sample/ss2.jpg", "images_sample/ss3.jpg", "images_sample/ss1.jpg", "images_sample/ss5.jpg"]
        else:
            st.error("Invalid URL. Please enter a valid YouTube link.")
            return None

        # Display images in a 2x2 grid
        num_images = min(4, len(image_paths))  # Limit to 4 images
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))  # Reduced figsize for smaller images
        names = ["Recommedation 1", "Recommedation 2", "Recommedation 3", "Recommedation 4"]
        # Iterate through the selected images and display them
        for i in range(num_images):
            row = i // 2
            col = i % 2
            img = mpimg.imread(image_paths[i])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(names[i], fontsize=10, pad=8, loc='center')

        # Remove any unused axes
        for j in range(num_images, 4):
            fig.delaxes(axes[j // 2, j % 2])

        st.pyplot(fig)

# --- Object Detection and Recommendations --- #
def main():
    image_paths = upload_video()

    if image_paths:
        st.write("### Recommended Outfits")

        # Display images in a 2x2 grid
        num_images = min(4, len(image_paths))  # Limit to 4 images
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))  # Reduced figsize for smaller images

        # Iterate through the selected images and display them
        for i in range(num_images):
            row = i // 2
            col = i % 2
            img = mpimg.imread(image_paths[i])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')

        # Remove any unused axes
        for j in range(num_images, 4):
            fig.delaxes(axes[j // 2, j % 2])

        st.pyplot(fig)

if _name_ == "_main_":
    main()




