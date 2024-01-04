import streamlit as st
from widgets.general import title, normal_text, ordered_list
from util.custom_theme import load_css
from streamlit_lottie import st_lottie

# Main app logic
def main():
    st.set_page_config(
        page_title="Traffic Sign Recognition Against Adversarial Attack",
        page_icon="🚗",
    )
    load_css()
    title("Welcome to SignGuard", "SignGuard", color="yellow", additional_style="font-weight: bold;")
    st.subheader("Introduction")
    normal_text("Traffic signs are essential tools that provide crucial information and instructions to drivers. Despite advancements in navigation technology, such as Google Maps and Waze, the reliance on physical traffic signs persists due to their critical role in road safety. This project aims to enhance road safety by focusing on traffic sign recognition technology, a key component of Advanced Driver Assistance Systems (ADAS) employed in autonomous vehicles. Traffic sign recognition involves the ability to identify and interpret various traffic signs using camera-captured images. Deep learning models, renowned for their ability to learn complex patterns from massive datasets, are widely used in traffic sign recognition. However, they are vulnerable to adversarial attacks, where slight, often imperceptible perturbations in input images can lead to incorrect predictions. This vulnerability poses a significant challenge to the reliability of these systems.", style='text-align: justify')
    st.subheader("Problem Statement")
    normal_text("Deep learning models are vulnerable to adversarial attacks, which deceive models into making incorrect predictions through barely noticeable alterations to input data. This project specifically focuses on the Fast Gradient Sign Method (FGSM), a widely recognised and simple attack that significantly impacts model accuracy. FGSM creates adversarial examples by using the gradient of neural networks. It then multiplies a small factor, epsilon (ε), which controls the magnitude of perturbation (Goodfellow et al., 2015). At an epsilon of 8/255, this method can drastically reduce a model's accuracy, from 98.45% to 38.81%.", style='text-align: justify')
    st.subheader("Objectives")

    # Create columns for Lottie animations
    lottie_cols = st.columns([0.05, 0.25, 0.05, 0.25, 0.05, 0.25, 0.05])

    # Display Lottie animations in the specified columns
    with lottie_cols[1]:
        st_lottie("https://lottie.host/89057c28-44d5-44e7-b49d-47fcdf06c674/gX9LcWhTSe.json")

    with lottie_cols[3]:
        st_lottie("https://lottie.host/0d1d36ae-aaa6-496a-a52d-3f1670d622f7/07kmiadSXj.json")

    with lottie_cols[5]:
        st_lottie("https://lottie.host/0f8a558c-0fb5-489a-b0d4-ce8c348daf6c/cJybnVRjrM.json")
    objectives = ["To develop an adversarially robust traffic sign recognition model.", "To evaluate the effectiveness of the models in mitigating the impact of the FGSM attack.", "To implement a dashboard to demonstrate the robustness of traffic sign recognition models against FGSM attack."]
    ordered_list(objectives)

    # Create columns for objective descriptions
    # obj_cols = st.columns([0.05, 0.25, 0.05, 0.25, 0.05, 0.25, 0.05])

    # # Display descriptions in the specified columns
    # with obj_cols[1]:
    #     st.markdown("<p class='center-p'>To develop an adversarially robust traffic sign recognition model</p>", unsafe_allow_html=True)
    #
    # with obj_cols[3]:
    #     st.markdown("<p class='center-p'>To evaluate the effectiveness of the model in mitigating the impact of the FGSM attack</p>", unsafe_allow_html=True)
    #
    # with obj_cols[5]:
    #     st.markdown("<p class='center-p'>To implement a dashboard to demonstrate the robustness of traffic sign recognition model against FGSM attack</p>", unsafe_allow_html=True)
    st.subheader("User Guide")
    user_guide = ["**Overview Page:** Get introduced to the app and its purpose, understand the fundamentals of traffic sign recognition, and learn about the significance of addressing adversarial attacks in deep learning models.", "**EDA (Exploratory Data Analysis) Page:**    Explore data visualizations related to traffic signs, providing insights into the dataset used in our models.", "**Prediction Page:** This is where you can interact with the model. Upload traffic sign images, see real-time predictions, and test the model's performance against adversarial attacks."]
    ordered_list(user_guide)

if __name__ == "__main__":
    main()