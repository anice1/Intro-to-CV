from matplotlib import cm
import streamlit as st
from PIL import Image
from config import BilinearInterpolation
import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
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
        # convert our image to RGB because it's coming as CMYK
        image = cmyk_2_rgb(image)

        bilinear = BilinearInterpolation(image, [2,2])
        
        with col1:
            st.subheader('Original Image')
            st.image(image, clamp=True)
        
        with col2:
            st.subheader('Generated Image')
            gen_image = bilinear.core_transform()
            st.image(gen_image, clamp=True)


def image_segmentation_app():
    
    st.title('Image Color Segmentation')
    st.write('''
                This tiny project will help you segment images by their colors, the aim isn't to solve complex problems, it's an enabler
                for greater projects to come. I'm learning computer vision and it's only meant for practice 😊
            ''')
    
    image = st.file_uploader('Upload Image', type=['png','jpg','jpeg'], help='Only .png, jpeg or .jpg are acceptable')

    col1, col2 = st.columns(2)
    with col1:
        upper_bound = color.hex2color(st.color_picker('Pick an upper bound'))
        upper_bound = color.rgb_to_hsv(upper_bound)
    

    with col2:
        lower_bound = color.hex2color(st.color_picker('Pick a Lower bound'))    
        lower_bound = color.rgb_to_hsv(lower_bound)
    
    #Image handling
    if image:
        
        image = cmyk_2_rgb(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        with col1:
            st.subheader('Original Image')
            st.image(image)

        with col2:
            st.subheader('Segmented')
            segmented = cv2.inRange(hsv_image, (upper_bound[0],upper_bound[1],upper_bound[2]), 
                                                (lower_bound[0],lower_bound[1],lower_bound[2]))
            st.image(segmented, clamp=True)
    

def cmyk_2_rgb(image):
    if np.asarray(Image.open(image)).shape[2] > 3:
        image = np.asarray(Image.open(image))
        c = image[:,:,0]
        m = image[:,:,1]
        y = image[:,:,2]
        k = image[:,:,3]

        red = 255 * (1-c/100) * (1-k/100)
        green = 255 * (1-m/100) * (1-k/100)
        blue = 255 * (1-y/100) * (1 - k/100)
        return np.asarray((red,green,blue))
    return np.asarray(Image.open(image))