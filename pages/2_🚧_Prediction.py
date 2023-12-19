import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from classify import generate_adversarial_example, predict, predict_adv
from widgets.general import normal_text
from util.custom_theme import load_css
from PIL import Image

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

load_css()

sign_names = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed + passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'}

st.title("Traffic Sign Recognition Against Adversarial Attack")

plotly_placeholder = st.empty()

def create_plotly_figure():
    nb_correct_original = np.load('data/nb_correct_original.npy')
    nb_correct_robust_madry = np.load('data/nb_correct_robust_madry_3.npy')
    nb_correct_robust_trades = np.load('data/nb_correct_robust_trades_3.npy')
    nb_correct_robust_awp = np.load('data/nb_correct_robust_awp_4.npy')

    # Generate the epsilon range which should be the same as used when saving the arrays
    eps_range = np.linspace(0, 1, 256)  # make sure this range is correct

    # Create a Plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps_range, y=nb_correct_original, mode='lines', name='Standard'))
    fig.add_trace(go.Scatter(x=eps_range, y=nb_correct_robust_madry, mode='lines', name='Madry'))
    fig.add_trace(go.Scatter(x=eps_range, y=nb_correct_robust_trades, mode='lines', name='TRADES'))
    fig.add_trace(go.Scatter(x=eps_range, y=nb_correct_robust_awp, mode='lines', name='AWP'))

    # Update the layout
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Perturbation Size (eps)',
        yaxis_title='Classification Accuracy',
        legend_title='Defence Method'
    )
    return fig

def load_test_data():
    y_test = pd.read_csv('data/Test.csv')
    return y_test

plotly_figure = create_plotly_figure()
plotly_placeholder.plotly_chart(plotly_figure)

y_test = load_test_data()

# Add a slider for selecting epsilon value
epsilon = st.slider('Select epsilon value for the adversarial attack:', min_value=0.000, max_value=1.000, value=0.03, step=0.001, format="%f")

uploaded_file = st.file_uploader("Choose an image...", type="png")

# Reset session state related to the adversarial image if a new file is uploaded
if 'last_uploaded_file' in st.session_state and uploaded_file != st.session_state['last_uploaded_file']:
    st.session_state['adv_img_generated'] = False
    if 'adversarial_image' in st.session_state:
        del st.session_state['adversarial_image']
    # Reset predictions as well
    for key in ['standard_prediction', 'adv_standard_prediction',
                'robust_1_prediction', 'robust_2_prediction', 'robust_3_prediction']:
        if key in st.session_state:
            del st.session_state[key]

st.session_state['last_uploaded_file'] = uploaded_file

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Extract the filename (assuming the filename is the last part of the path in 'Path' column of y_test)
    uploaded_filename = 'Test/' + uploaded_file.name
    # Look up the filename in the CSV to find the corresponding ClassId
    true_label_row = y_test[y_test['Path'].str.contains(uploaded_filename)]
    if not true_label_row.empty:
        true_label = true_label_row.iloc[0]['ClassId']
        st.session_state['true_label'] = true_label  # Store the true label in the session state
        true_label_name = sign_names[true_label]
        # st.write(f"The true label is: {true_label_name}")
    else:
        st.error("The uploaded image's filename does not match any entry in the test dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)

        if st.button('Predict (Standard Model)'):
            label, confidence = predict(uploaded_file)
            st.session_state['standard_prediction'] = (label, confidence * 100)

        if 'standard_prediction' in st.session_state:
            label, confidence = st.session_state['standard_prediction']
            name = sign_names.get(label)
            # Check if the prediction is correct
            is_correct = name == true_label_name

            # Set the color based on correctness
            color = "#39FF14" if is_correct else "red"
            normal_text(f"**Prediction:** {name}", style=f"color: {color};")
            normal_text(f"**Confidence:** {confidence:.2f}%", style=f"color: {color};")

        if st.button('Generate Adversarial Image'):
            adversarial_image = generate_adversarial_example(image, 'standard', epsilon)
            st.session_state['adv_img_generated'] = True
            st.session_state['adversarial_image'] = adversarial_image

    if 'adv_img_generated' in st.session_state and st.session_state['adv_img_generated']:
        with col2:
            st.image(st.session_state['adversarial_image'], caption='Adversarial Image', use_column_width=True)

            if st.button('Predict with Standard Model'):
                adversarial_image = st.session_state['adversarial_image']
                label, confidence = predict_adv(adversarial_image, 'standard')
                st.session_state['adv_standard_prediction'] = (label, confidence * 100)

            if 'adv_standard_prediction' in st.session_state:
                label, confidence = st.session_state['adv_standard_prediction']
                name = sign_names.get(label)
                # Check if the prediction is correct
                is_correct = name == true_label_name

                # Set the color based on correctness
                color = "#39FF14" if is_correct else "red"
                normal_text(f"**Prediction:** {name}", style=f"color: {color};")
                normal_text(f"**Confidence:** {confidence:.2f}%", style=f"color: {color};")

            if st.button('Predict with Madry'):
                adversarial_image_1 = generate_adversarial_example(image, 'madry', epsilon)
                label, confidence = predict_adv(adversarial_image_1, 'madry')
                st.session_state['robust_1_prediction'] = (label, confidence * 100)

            if 'robust_1_prediction' in st.session_state:
                label, confidence = st.session_state['robust_1_prediction']
                name = sign_names.get(label)
                # Check if the prediction is correct
                is_correct = name == true_label_name

                # Set the color based on correctness
                color = "#39FF14" if is_correct else "red"
                normal_text(f"**Prediction:** {name}", style=f"color: {color};")
                normal_text(f"**Confidence:** {confidence:.2f}%", style=f"color: {color};")

            if st.button('Predict with TRADES'):
                adversarial_image_2 = generate_adversarial_example(image, 'trades', epsilon)
                label, confidence = predict_adv(adversarial_image_2, 'trades')
                st.session_state['robust_2_prediction'] = (label, confidence * 100)

            if 'robust_2_prediction' in st.session_state:
                label, confidence = st.session_state['robust_2_prediction']
                name = sign_names.get(label)
                # Check if the prediction is correct
                is_correct = name == true_label_name

                # Set the color based on correctness
                color = "#39FF14" if is_correct else "red"
                normal_text(f"**Prediction:** {name}", style=f"color: {color};")
                normal_text(f"**Confidence:** {confidence:.2f}%", style=f"color: {color};")

            if st.button('Predict with AWP'):
                adversarial_image_3 = generate_adversarial_example(image, 'awp', epsilon)
                label, confidence = predict_adv(adversarial_image_3, 'awp')
                st.session_state['robust_3_prediction'] = (label, confidence * 100)

            if 'robust_3_prediction' in st.session_state:
                label, confidence = st.session_state['robust_3_prediction']
                name = sign_names.get(label)
                # Check if the prediction is correct
                is_correct = name == true_label_name

                # Set the color based on correctness
                color = "#39FF14" if is_correct else "red"
                normal_text(f"**Prediction:** {name}", style=f"color: {color};")
                normal_text(f"**Confidence:** {confidence:.2f}%", style=f"color: {color};")
