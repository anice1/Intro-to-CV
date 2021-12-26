import streamlit as st
from PIL import Image
from config import BilinearInterpolation
import matplotlib.colors as color
import matplotlib.pyplot as plt
import cv2
import io

def bilinear_app():
    st.title('Bilinear Interpolation')
    st.info('Implementation of custom Bilinear Interpolation algorithm on an image')

    st.write('''Bilinear interpolation is performed using linear interpolation first in one direction, 
                and then again in the other direction. Although each step is linear in the sampled values and in the position, 
                the interpolation as a whole is not linear but rather quadratic in the sample location. 
                \n Source: https://en.wikipedia.org/wiki/Bilinear_interpolation''')

    image = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'], help='Only .png and .jpg are accepted')

    col1, col2 = st.columns(2)
    if image:
        Image.open(image.read())
        with col1:
            st.subheader('Original Image')
            st.image(image)
        
        with col2:
            st.subheader('Generated Image')
            st.image(image)


def image_segmentation_app():
    st.title('Image Color Segmentation')
    st.write('''
                This tiny project will help you segment images by their colors, the aim isn't to solve complex problems, it's an enabler
                for greater projects to come. I'm learning computer vision and it's only meant for practice 😊
            ''')
    
    image = st.file_uploader('Upload Image', type=['png','jpg','jpeg'], help='Only .png, jpeg or .jpg are acceptable')
    image = Image.open('../images/business.jpg')

    col1, col2 = st.columns(2)

    with col1:
        upper_bound = color.hex2color(st.color_picker('Pick an upper bound'))
        upper_bound = color.rgb_to_hsv(upper_bound)

    with col2:
        lower_bound = color.hex2color(st.color_picker('Pick a Lower bound'))    
        lower_bound = color.rgb_to_hsv(lower_bound)
    
    #Image handling
    if image:
        with col1:
            st.subheader('Original Image')
            st.image(image)
        with col2:
            st.subheader('Segmented')
            st.image(image)
    
    image = cv2.imread('../images/business.jpg')
    
    #convert image to HSV
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(new_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(new_image, new_image, mask=mask)
    st.image(result)