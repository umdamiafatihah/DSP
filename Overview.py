import streamlit as st
from widgets.general import normal_text, ordered_list
from util.custom_theme import load_css

# Main app logic
def main():
    st.set_page_config(
        page_title="Traffic Sign Recognition Against Adversarial Attack",
        page_icon="🚗",
    )
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
    st.title("Overview")
    st.subheader("Introduction")
    normal_text("Traffic signs are essential tools that provide crucial information and instructions to drivers. Despite advancements in navigation technology, such as Google Maps and Waze, the reliance on physical traffic signs persists due to their critical role in road safety. This project aims to enhance road safety by focusing on traffic sign recognition technology, a key component of Advanced Driver Assistance Systems (ADAS) employed in autonomous vehicles. Traffic sign recognition involves the ability to identify and interpret various traffic signs using camera-captured images, a vital function in autonomous driving systems. Deep learning models, renowned for their ability to learn complex patterns from massive datasets, are widely used in traffic sign recognition. However, they are vulnerable to adversarial attacks, where slight, often imperceptible perturbations in input images can lead to incorrect predictions. This vulnerability poses a significant challenge to the reliability of these systems.", style='text-align: justify')
    st.subheader("Problem Statement")
    normal_text("The project addresses the critical issue of adversarial attacks in traffic sign recognition systems, specifically focusing on the Fast Gradient Sign Method (FGSM). FGSM generates adversarial examples by first calculating the gradient of the loss function with respect to the input image. Then, it takes the sign of this gradient, capturing the direction of each pixel's adjustment. Finally, it multiplies this sign matrix by a small factor, epsilon (ε), which controls the magnitude of the perturbation.", style='text-align: justify')
    st.subheader("Objectives")
    objectives = ["To develop an adversarially robust traffic sign recognition model.", "To evaluate the effectiveness of the model in mitigating the impact of the FGSM attack.", "To implement a dashboard to demonstrate the robustness of traffic sign recognition model against FGSM attack."]
    ordered_list(objectives)
    st.subheader("User Guide")
    user_guide = ["**Overview Page:** Get introduced to the app and its purpose, understand the fundamentals of traffic sign recognition, and learn about the significance of addressing adversarial attacks in deep learning models.", "**EDA (Exploratory Data Analysis) Page:**    Explore data visualizations related to traffic signs, providing insights into the dataset used in our models.", "**Prediction Page:** This is where you can interact with the model. Upload traffic sign images, see real-time predictions, and test the model's performance against adversarial attacks."]
    ordered_list(user_guide)
    # st.write("**1. Overview Page:** Get introduced to the app and its purpose, understand the fundamentals of traffic sign recognition, and learn about the significance of addressing adversarial attacks in deep learning models.", "**EDA (Exploratory Data Analysis) Page:**    Explore data visualizations related to traffic signs, providing insights into the dataset used in our models.")
    #
    # st.write("**2. EDA (Exploratory Data Analysis) Page:**    Explore data visualizations related to traffic signs, providing insights into the dataset used in our models.")
    #
    # st.markdown("**3. Prediction Page:** This is where you can interact with the model. Upload traffic sign images, see real-time predictions, and test the model's performance against adversarial attacks.")
    # st.markdown("""
    # - Click on 'Browse files' to upload your traffic sign image.
    # - View the standard model's prediction and then generate an adversarial version of the image.
    # - Compare predictions between the standard and robust models to evaluate their performance.
    # """)

if __name__ == "__main__":
    main()