import streamlit as st
from operations import *


with st.sidebar:
    st.title('Task Summary on Computer Vision')
    operation = st.selectbox('Choose an operation',
                            ('Bilinear Interpolation', 'Image Color Segmentation'))

# Perform for Bilinear Interpolation
if operation == 'Bilinear Interpolation':
    bilinear_app()

# Image Color segmentation
elif operation == 'Image Color Segmentation':
    image_segmentation_app()