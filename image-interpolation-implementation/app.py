import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from config import BilinearInterpolation

with st.sidebar:
    st.title('Task Summary on Computer Vision')
    operation = st.selectbox('Choose an operation',
                            ('Bilinear Interpolation', 'Image Color Segmentation'))

if operation == 'Bilinear Interpolation':
    st.title('Bilinear Interpolation')
    st.info('Implementation of custom Bilinear Interpolation algorithm on an image')

    st.write('''Bilinear interpolation is performed using linear interpolation first in one direction, 
                and then again in the other direction. Although each step is linear in the sampled values and in the position, 
                the interpolation as a whole is not linear but rather quadratic in the sample location. 
                \n Source: https://en.wikipedia.org/wiki/Bilinear_interpolation''')

    image = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'], help='Only .png and .jpg are accepted')

    col1, col2 = st.columns(2)

    if image:
        Image.load_img(image.read())
        with col1:
            st.subheader('Original Image')
            st.image(image)
        
        with col2:
            st.subheader('Generated Image')
            st.image(image)

# Image Color segmentation
elif operation == 'Image Color Segmentation':
    st.title('Image Color Segmentation')
    st.write('''
                This tiny project will help you segment images by their colors, the aim isn't to solve complex problems, it's an enabler
                for greater projects to come. I'm learning computer vision and it's only meant for practice 😊
            ''')
    
    image = st.file_uploader('Upload Image', type=['png','jpg','jpeg'], help='Only .png, jpeg or .jpg are acceptable')
    col1, col2 = st.columns(2)
    if image:
        with col1:
            st.subheader('Original Image')
            st.image(image)
        with col2:
            st.subheader('Segmented')
            st.image(image)

    with col1:
        st.write('')
        st.color_picker('color range')