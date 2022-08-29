# In this file we will use the trained model to segment, then perform a diagnosis
import os
import streamlit as st
from PIL import Image
import numpy as np
from skimage import io
from d_segmentation import segmentation

@st.cache
def load_image(uploaded_img):
    img = Image.open(uploaded_img)
    return img

def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

def st_segment(**kwargs):
    
    with st.spinner('Segmentation in progress...'):
        uploaded_img = kwargs.get('img')
        resize_factor = kwargs.get('resize_factor')
        if uploaded_img is None:
            return
        
        img = load_image(uploaded_img)
        save_uploadedfile(uploaded_img)
        st.image(img, 250, 250)

        # Start segmentation process
        img = io.imread(os.path.join("tempDir",uploaded_img.name))
        segmented_image = segmentation.segment(img = img, model = "model\\model_1.h5", size=resize_factor)
        st.image(segmented_image, 250, 250)

        io.imsave(os.path.join("tempDir",'result_seg.png'),segmented_image)
        
    st.success('Done!')

def st_accuracy(**kwargs):
    uploaded_gt = kwargs.get('gt')
    if uploaded_gt is None:
        return
    ground_truth = load_image(uploaded_gt)
    save_uploadedfile(uploaded_gt)
    st.image(ground_truth, 250, 250)
    ground_truth = io.imread(os.path.join("tempDir",uploaded_gt.name))
    segmentation_result = io.imread(os.path.join("tempDir", "result_seg.png"))
    ret = segmentation.compute_accuracy(segmentation_result=segmentation_result, ground_truth=ground_truth)
    st.write(ret)
    

if __name__ == '__main__':
    try:
        os.mkdir('tempDir')
    except:
        pass
    st.title('Segmentation of lesion dermoscopic images!')
    # tab1, tab2 = st.tabs(['Segmentation', 'Compute Accuracy'])

    # with tab1:
    uploaded_img = st.file_uploader("Upload mage", type=['png'])
    uploaded_gt = st.file_uploader("Upload ground truth", type=['png'])
    resize_factor = st.select_slider("Resize factor (1 means no resize)", options = np.arange(0.2,1.1,0.1))
    st.button("Run segmentation", key=None, help='Click to start segmentation', on_click=st_segment, args= None, 
        kwargs={'img' : uploaded_img,
                'resize_factor' : resize_factor,
                })
    
    # with tab2:
    st.button("Compute accuracy", key=None, help=None, on_click=st_accuracy, args= None, 
        kwargs={'gt' : uploaded_img,
                })






