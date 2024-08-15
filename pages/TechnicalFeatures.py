import streamlit as st



st.markdown("# :blue[Technical Features] ")
st.divider()

st.write("The system takes in an image or video as input and outputs similar fashion items based on visual similarity attributes. The overall system is structured as follows:")

# System overview
st.image('flowchart.jpg')
st.caption("When a user uploads a video or image, the system detects fashion objects to obtain their individual vector embeddings. It then performs a similarity search in the vector index to find the most visually similar fashion items available in the catalog.")

# # Vector index creation
# st.image('images/flowcharts/vector_index.png')
# st.caption("To create the vector index, the embedding model extracts latent features from every item in the catalog and stores them in a vector index based on their similarities. For this project, a Flat Index with L2 distance was used as the similarity measure.")

# Technical features
st.divider()
st.markdown("#### Technical Features:")
st.markdown("* *Object Detection Model:* Utilized YOLOv5, a state-of-the-art model trained on fashion images, to detect and segment fashion objects in images or videos.")
st.markdown("* *Feature Extraction:* Employed a Convolutional Autoencoder implemented with PyTorch to extract and encode the latent features of detected fashion objects.")
st.markdown("* *Vector Index and Similarity Search Algorithm:* Implemented the FAISS library to construct a vector index and perform efficient similarity searches based on visual attributes of the fashion items.")

# More information
st.divider()
st.markdown("#### Model Training and Evaluation:")
st.markdown("For detailed information on how the model was trained and the evaluation process, please refer to the following posts:")
st.markdown("[Object Detection Model with YOLOv5](https://pyimagesearch.com/2022/06/20/training-the-yolov5-object-detector-on-a-custom-dataset/) ")
st.markdown("[Visual Search Engine with FAISS](https://blog.roboflow.com/clip-image-search-faiss/)")