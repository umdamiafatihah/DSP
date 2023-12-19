import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
from widgets.general import normal_text, ordered_list
from util.custom_theme import load_css
import os

load_css()
st.markdown(
        """
        <style>
           [data-testid="stSidebarNav"]::before {
            content: "Traffic Sign Recognition Against Adversarial Attack";
            display: block;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            position: absolute;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
st.title("Exploratory Data Analysis (EDA)")
normal_text("This section provides a detailed examination of the dataset, featuring the distribution of classes, an analysis of image size variability, and a visual gallery of sample images.", style="text-align:justify")
st.subheader("Class Distribution")
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

df = pd.read_csv('data/class_distribution.csv')

# # Extract class labels and number of images into separate lists
class_num = df['Class Label'].tolist()
train_number = df['Number of Images'].tolist()

# # Create a DataFrame for the class distribution
df = pd.DataFrame({
    'Class Label': class_num,
    'Number of Images': train_number
})

# Create a Plotly chart object
fig = px.bar(df, x='Class Label', y='Number of Images')
fig.update_layout(
    xaxis=dict(tickangle=-90),  # Rotate x-axis labels
    xaxis_title="Class Label",
    yaxis_title="Number of Images",
    font=dict(size=10)  # Adjust font size if necessary
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

# st.markdown("**Explanation:** The class distribution diagram represents the frequency of each traffic sign class within the dataset. A wide variation in the number of samples per class can be observed, with certain traffic sign classes such as 'Speed limit (50km/h)' being more heavily represented than others like 'Dangerous curve to the left'. This variation reflects real-world frequencies where common signs are naturally encountered more often, and hence, are more abundantly captured in datasets.")

st.subheader("Image Size Distribution")

widths = np.load('data/widths.npy')
heights = np.load('data/heights.npy')

# Create two columns to display plots side by side
col1, col2 = st.columns(2)

# Create the histogram for image widths in the first column
with col1:
    fig_width = px.histogram(widths, nbins=30, labels={'value': 'Width'})
    fig_width.update_layout(yaxis_title="Frequency", showlegend=False)
    fig_width.update_traces(hovertemplate='Width: %{x}<br>Count: %{y}')
    st.plotly_chart(fig_width, use_container_width=True)

# Create the histogram for image heights in the second column
with col2:
    fig_height = px.histogram(heights, nbins=30, labels={'value': 'Height'})
    fig_height.update_layout(yaxis_title="Frequency", showlegend=False)
    fig_height.update_traces(hovertemplate='Height: %{x}<br>Count: %{y}')
    st.plotly_chart(fig_height, use_container_width=True)

# st.markdown("**Explanation:** The histograms above display the distribution of image widths and heights in the dataset. The left histogram shows that image widths predominantly cluster at the lower end, indicating a majority of narrower images. The right histogram similarly indicates that most images are shorter, with heights concentrated at lower values. These patterns highlight the dataset's skew towards smaller image dimensions, which is important for configuring image processing models, particularly when considering the architecture and preprocessing requirements.")

st.subheader("Sample Images")
# Set the path to your folder of images
image_path = 'data/samples'

# Get a list of image paths from the folder
image_paths = [os.path.join(image_path, filename) for filename in os.listdir(image_path) if filename.endswith(('.png'))]

# Sort the list to maintain order
image_paths.sort()

# Load images using PIL
images = [Image.open(img_path) for img_path in image_paths]

# Define the fixed width for each image
fixed_width = 200

# Define your captions for each image
captions = ["Speed limit (30km/h) - Low Light Condition", "Speed limit (60km/h) - Highly Blurred", "General caution - Silhouetted by Strong Rear Illumination", "Road work - Overexposed", "Traffic signals - Partially Obstructed", "Pedestrians - Standard Lighting Condition"]  # etc.

# Define the fixed width for each image
fixed_width = 200

# Display images in two rows of three, with the same width
for i in range(0, len(images), 3):
    cols = st.columns(3)  # Create three columns
    images_slice = images[i:i+3]
    captions_slice = captions[i:i+3]
    for col, img, caption in zip(cols, images_slice, captions_slice):
        with col:
            # Display each image with the same width
            st.image(img, width=fixed_width)
            # Use Markdown with HTML to display the caption in white
            st.markdown(f"<p style='color:white; font-size:14px'>{caption}</p>", unsafe_allow_html=True)