# Import necessary libraries
import streamlit as st #UI 
from streamlit_option_menu import option_menu
import tensorflow as tf #framework #framework for training
import numpy as np #for image resizing 
import time

# Set the global page config
st.set_page_config(
    page_title="SeeZoom",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Tensorflow Model Prediction function
def model_prediction(test_image):
    # Load pre-trained model
    model = tf.keras.models.load_model('trained_model.h5')
    # Load and preprocess the test image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single Image to batch
    # Make predictions using the model
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]
    accuracy_percent = confidence * 100
    return predicted_class_index, accuracy_percent
    pass

# TaskBar setup
app_mode = option_menu(
    
    menu_title=None,
    options=["Home", "About", "Practice", "Prediction"],
    icons=["house", "book", "lightbulb" , "star"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#274134"},
        "body": {"background-color": "red"},
        "icons": {"color": "orange"},
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "green",
        },
        "nav-link-selected": {"background-color": "#1A2421"},
    },
)


# Home Page
if app_mode == 'Home':
    st.markdown(
        """
        <style>
            img {
                margin-top: -8px;  /* Adjust the top margin as needed */
                margin-bottom: -8px;  /* Adjust the bottom margin as needed */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image('1.jpg', use_column_width=True)
    st.image('2.jpg', use_column_width=True)
    st.image('3.jpg', use_column_width=True)

    st.image('footer.jpg', use_column_width=True)


# About Our Project Page
elif(app_mode=='About'):
    st.markdown(
        """
        <style>
            img {
                margin-top: -40px;  /* Adjust the top margin as needed */
                margin-bottom: -8px;  /* Adjust the bottom margin as needed */
            }
            .centered-text {
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.image('about1.jpg', use_column_width=True)

    # Center-align the image using streamlit columns
    col1, col2= st.columns(2)
    
    with col1:
        st.image('logo.png', use_column_width=True)



    with col2:
        about = '''<span style="font-size: 20px;">We predict the classification of vegetable and fruit seeds, focusing on five specific varieties for each category. 
        The system provides classifications for fruit seeds such as melon, singkamas, sweet corn, watermelon, and waxy corn.
        The vegetable seeds such as kalabasa, pechay, red pole sitao, snap beans, and snow peas. 
        </span>'''

        st.markdown(about, unsafe_allow_html=True)

        about2 = '''<span style="font-size: 20px;">The project is limited to ten varieties of seeds, and the model does not have the accuracy to classify other variations of seeds.'''
        st.markdown(about2, unsafe_allow_html=True)

        about3 = '''<span style="font-size: 20px;">To achieve successful predictions, we recommend uploading a close-up image of the seeds. Our aim is to contribute to the agriculture sector 
        by leveraging machine learning for precision agriculture.'''
        st.markdown(about3, unsafe_allow_html=True)
    
 
    st.image('footer.jpg', use_column_width=True)
 


    #image_path = 'footer.jpg'
    #st.image(image_path, width=575, use_column_width=True)  
   


# Practice Page
elif(app_mode=='Practice'):

    st.image('practice.jpg', use_column_width=True)

  
    # Initialize selected_predefined_image
    selected_predefined_image = "seed_ex\seed_ex_1.jpg"

    # Or choose from predefined images
    predefined_images = [
            "seed_ex\seed_ex_1.jpg",
            "seed_ex\seed_ex_2.jpg",
            "seed_ex\seed_ex_3.jpg",
            "seed_ex\seed_ex_4.jpg",
            "seed_ex\seed_ex_5.jpg",
            "seed_ex\seed_ex_6.jpg",
            "seed_ex\seed_ex_7.jpg",
            "seed_ex\seed_ex_8.jpg",
            "seed_ex\seed_ex_9.jpg",
            "seed_ex\seed_ex_10.jpg",

        ]

    col1, col2, col3, col4, col5 = st.columns([2, 1.6, 0.09, 1.5, 1])
        
    
    with col2:

        # Handle the case when selected_predefined_image is None
         selected_predefined_image = st.selectbox('Select an example image:', predefined_images, index=predefined_images.index(selected_predefined_image))

    with col4:

        # Show the selected predefined image
         st.write("")
         st.write("")

         if st.button('Show Example Image'):

            with col2:
                col1, col2= st.columns([1.5,3])
                with col2:
                 st.write("")
                 st.write("")

                 st.image(selected_predefined_image, width=350)
            

    # Predict Button
    col1, col2, col3, col4= st.columns([1.073, 0.2, 1.05, 1])

    with col2:
        if st.button('Predict'):
                
                with col3:
                    with st.spinner("Predicting..."):
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.01)  # Simulate processing time
                            progress_bar.progress(percent_complete + 1)

                    if predefined_images is not None:
                        result_index, accuracy_percent = model_prediction(selected_predefined_image)  # Use selected_predefined_image for prediction

                    # Reading Labels
                    with open("labels (1).txt") as f:
                        content = f.readlines()
                    label = [i[:-1] for i in content]
                    st.success("This is {} with {:.2f}% accuracy.".format(label[result_index], accuracy_percent))
        
# Prediction Page
elif app_mode == 'Prediction':
    
    st.image('predict.jpg', use_column_width=True)
    
    # Center-align the image using streamlit columns
    col1, col2, col3= st.columns(3)
    
    with col2:

        # Option to upload a custom image
        test_image = st.file_uploader('Zoom your seed. ', type=['jpg', 'jpeg', 'png'])


        if test_image is not None:
            # Check if the uploaded image is a seed image
            seed_image_types = ['jpg', 'jpeg', 'png']
            file_extension = test_image.name.split('.')[-1].lower()

            if file_extension in seed_image_types:
                    col1, col2, col3= st.columns([1,2,1])
                    with col2:
                        st.image(test_image, width=4, use_column_width=True)
                        # Update selected_predefined_image when a new image is uploaded
                        selected_predefined_image = None  # Set to None to prevent showing predefined image simultaneously


        # Predict Button
        col1, col2, col3= st.columns([1, 0.1, 5.4])

        with col1:
            if st.button('Predict'):

                with col3:
                    if test_image is None:
                        st.warning("Please add a seed image before clicking Predict.")
                    else:
                        with st.spinner("Predicting..."):
                            progress_bar = st.progress(0)
                            for percent_complete in range(100):
                                time.sleep(0.01)  # Simulate processing time
                                progress_bar.progress(percent_complete + 1)

                        if test_image is not None:
                            result_index, accuracy_percent = model_prediction(test_image)  # Use the uploaded image for prediction
                        
                        # Reading Labels
                        with open("labels (1).txt") as f:
                            content = f.readlines()
                        label = [i[:-1] for i in content]
                        st.success("This is {}, with {:.2f}% accuracy.".format(label[result_index], accuracy_percent))
      