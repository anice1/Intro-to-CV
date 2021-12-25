import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from config import BilinearInterpolation




with st.sidebar:
    st.title('Task Summary on Computer Vision')
    operation = st.selectbox('Choose an operation',('Bilinear Interpolation', 'Image Color Segmentation'))

if operation == 'Bilinear Interpolation':
    st.title('Bilinear Interpolation')
    st.info('A Practical implementation of Bilinear Interpolation algorithm on an image')

    st.write('''Bilinear interpolation is performed using linear interpolation first in one direction, 
                and then again in the other direction. Although each step is linear in the sampled values and in the position, 
                the interpolation as a whole is not linear but rather quadratic in the sample location. 
                \n Source: https://en.wikipedia.org/wiki/Bilinear_interpolation''')

    image = st.file_uploader('Upload Image', type=['png', 'jpg'], help='Only .png and .jpg are accepted')

    col1, col2 = st.columns(2)

    if image:
        with col1:
            st.subheader('Original Image')
            st.image(image)
        
        with col2:
            st.subheader('Generated Image')
            st.image(image)

elif operation == 'Image Color Segmentation':
    st.title('Image Color Segmentation')