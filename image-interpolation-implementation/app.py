from operations import *


st.set_page_config(
    page_icon="🤖",
    page_title="Intro to Computer Vision",
    initial_sidebar_state="expanded",
    menu_items={
        'About':'###### Reachout to the developer here https://www.linkedin.com/in/linkedin.com/in/ac-nice-81a367147',
        'Report a bug':'https://www.github.com/anochima',

    }
)

with st.sidebar:
    st.title('Task Summary on Computer Vision')
    operation = st.selectbox('Choose an operation',
                            ('Image Blurring & Edge Detection',
                            'Bilinear Interpolation', 
                            'Image Color Segmentation',))

# Perform for Bilinear Interpolation
if operation == 'Bilinear Interpolation':
    st.title('Bilinear Interpolation')
    st.info('Implementation of custom Bilinear Interpolation algorithm on an image')

    st.write('''Bilinear interpolation is performed using linear interpolation first in one direction, 
                and then again in the other direction. Although each step is linear in the sampled values and in the position, 
                the interpolation as a whole is not linear but rather quadratic in the sample location. 
                \n Source: https://en.wikipedia.org/wiki/Bilinear_interpolation''')
    bilinear_app()

# Image Color segmentation
elif operation == 'Image Color Segmentation':
    st.title('Image Color Segmentation')
    st.write('''
                This tiny project will help you segment images by their colors, the aim isn't to solve complex problems, it's an enabler
                for greater projects to come. I'm learning computer vision and it's only meant for practice 😊
            ''')
    image_segmentation_app()

elif operation == 'Image Blurring & Edge Detection':
    st.title('Image Blurring/Sharpening and Edge Detection')
    st.write('''
                This tiny project will help you blur/Sharpen and Detect Edges of images, the aim isn't to solve complex problems, it's an enabler
                for greater projects to come. I'm learning computer vision and it's only meant for practice 😊
            ''')
    blur_and_edge_detector_app()