from config import BilinearInterpolation
import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from scipy import signal
import cv2


def bilinear_app():
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

def blur_and_edge_detector_app():
    image = st.file_uploader('Upload an Image', accept_multiple_files=False, type=['png','jpg','jpeg'], help="Will help you blur and detect edge of an image",)
    col1, col2 = st.columns(2)
    with col1:
        filter = st.slider('Uniform Filter level',min_value=1, max_value=10, step=1)
    with col2:
        if image:
            st.image(image,use_column_width='auto')
            image_array = np.array(Image.open(image))
            
            # Convert to Gray
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            smoothing_threshold = np.ones((filter,filter))

            blurred_image = signal.convolve2d(gray_image, smoothing_threshold, boundary='symm', mode='same')
            blurred_image = cv2.resize(blurred_image, dsize=None, fx=0.5, fy=0.5)
            cv2.imwrite('blurred_image.jpg',blurred_image)
            st.image('blurred_image.jpg', clamp=False)

        xMask = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        yMask = xMask.T
        fx = signal.convolve2d(blurred_image,xMask, boundary='symm',mode='same')
        fy = signal.convolve2d(blurred_image,yMask, boundary='symm',mode='same')

        magnitude = (fx**2 + fy**2)**0.5
        threshold = magnitude.max() -50 * magnitude.std()
        st.image(magnitude>threshold, clamp=True) 
    


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