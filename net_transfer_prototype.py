import streamlit as st
import time
def customMsg(msg, wait=0, type_='warning'):
    placeholder = st.empty()
    styledMsg = f'\
        <div class="element-container" style="width: 693px;">\
            <div class="alert alert-{type_} stAlert" style="width: 693px;">\
                <div class="markdown-text-container">\
                    <p>{msg}</p></div></div></div>\
    '
    placeholder.markdown(styledMsg, unsafe_allow_html=True)
    time.sleep(wait)
    placeholder.empty()
#st.markdown("_The Libraries needed for the Application will take time to load. Please wait, your patience will be rewarded shortly..._")

import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image

import tensorflow_hub as hub
# Include PIL, load_image before main()
from PIL import Image

msg = "The Libraries needed for the Application will take time to load. Please wait, your patience will be rewarded shortly..."
customMsg(msg, 7, 'warning')

#vid_file = Image.open("Style_transfer_example.mov","rb").read() #play the video stored in specified location
#st.video(vid_file)

def load_image(content_image_file):
    content_img = Image.open(content_image_file)
    return content_img

st.title('New Image Generator') 
st.markdown("_~ C R Deepak Kumar, Guhanesvar M, Tilak Vijayaraghvan_")

st.subheader("Content Image")
content_image_file = st.file_uploader("Upload Content Image", type=["png","jpg","jpeg"])

if content_image_file is not None:
    # To See details
    file_details = {"filename":content_image_file.name, "filetype":content_image_file.type,
                              "filesize":content_image_file.size}
    st.write(file_details)
    # To View Uploaded Image
    st.image(load_image(content_image_file),width=350)
    
def load_image(style_image_file):
    style_img = Image.open(style_image_file)
    return style_img


st.subheader("Style Image")
style_image_file = st.file_uploader("Upload Style Image", type=["png","jpg","jpeg"])

if style_image_file is not None:
    # To See details
    file_details = {"filename":style_image_file.name, "filetype":style_image_file.type,
                              "filesize":style_image_file.size}
    st.write(file_details)
    # To View Uploaded Image
    st.image(load_image(style_image_file),width=350)

st.markdown("_Style image is superimposed on the Content image which generates a new image_")  




if st.button('Generate'):
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)
    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    def imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
            plt.imshow(image)
        if title:
            plt.title(title)

    sty_image = Image.open(style_image_file)
    sty_image = sty_image.resize((500,500))
    rgb_im = sty_image.convert('RGB')
    rgb_im.save(fp="newabs1image.jpeg")
    b=load_img("newabs1image.jpeg")

    content_image = Image.open(content_image_file)
    content_image = content_image.resize((500,500))
    cont_rgb_im = content_image.convert('RGB')
    cont_rgb_im.save(fp="newironmanimage.jpeg")
    a=load_img("newironmanimage.jpeg")

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(a), tf.constant(b))[0]
    stylized_image = tensor_to_image(stylized_image)
    st.subheader("New Image")
    st.image(stylized_image,width=500)
