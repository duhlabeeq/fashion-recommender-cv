import os
import pickle

import numpy as np
import streamlit as st
from obj_detection import ObjDetection
from PIL import Image

from src.utilities import (ExactIndex, extract_img, visualize_outfits,
                           remove_bg, build_category_indices, complementary_search)


# --- UI Configurations --- #
st.set_page_config(page_title="Smart Stylist powered by computer vision",
                   page_icon=":shopping_bags:"
                   )

st.markdown("# :female_fairy: :shopping_bags:")
st.markdown("# :rainbow[Your personal AI Stylist] :magic_wand:")

# --- Message --- #
st.write("Hello, welcome to my project page! :smiley:")
st.write("Smart Stylist is a computer vision powered web-app that lets you upload an image of an outfit and return recommendations on similar style. An image with white background works best. ")
st.write("For more information on how the system works, check out the project page [here](https://www.joankusuma.com/post/smart-stylist-a-fashion-recommender-system-powered-by-computer-vision) ")
st.divider()
st.info("Check out the gallery in sidebar to get some ideas", icon="👈🏼")

# --- Load Model and Data --- #
with st.spinner('Please wait while your model is loading'):
    yolo = ObjDetection(onnx_model='./models/best.onnx',
                        data_yaml='./models/data.yaml')
    
INDEX_READY = (
    os.path.exists("flatIndex.index") and
    os.path.exists("img_paths.pkl") and
    os.path.exists("embeddings.pkl")
)

if INDEX_READY:
    with open("img_paths.pkl", "rb") as im_file:
        image_paths = pickle.load(im_file)
    with open("embeddings.pkl", "rb") as file:
        embeddings = pickle.load(file)
    loaded_idx = ExactIndex.load(embeddings, image_paths, "flatIndex.index")
    with st.spinner('Building category indices...'):
        cat_indices = build_category_indices(embeddings, image_paths)
else:
    st.warning(
        "No index found. Run `python pipeline/build_index.py --dataset <your_dataset>` "
        "to build the index before using recommendations.",
        icon="⚠️"
    )
    cat_indices = {}

def upload_image():
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        if image_file.type in ('image/png', 'image/jpeg'):
            st.success('Valid Image File Type')
            return image_file
        else:
            st.error('Only the following image files are supported (png, jpg, jpeg)')

# --- Object Detection and Recommendations --- #
def main():
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object)
        st.image(image_obj)
        button = st.button('Show Recommendations', disabled=not INDEX_READY)
        if button:
            with st.spinner("Detecting fashion items from image..."):
                image_array = np.array(image_obj.convert("RGB"))
                cropped_objs = yolo.crop_objects(image_array)
                if cropped_objs is not None:
                    # Filter out empty crops
                    cropped_objs = [(obj, cls) for obj, cls in cropped_objs if obj.size > 0]
                    if cropped_objs:
                        prediction = True
                    else:
                        st.caption("No fashion objects detected.")
                else:
                    st.caption("No fashion objects detected.")

        if prediction:
            # Show detected items and their classes
            detected_labels = [cls for _, cls in cropped_objs]
            st.caption(f":rainbow[Detected:] {', '.join(detected_labels)}")

            with st.spinner("Removing backgrounds and finding complementary items..."):
                all_boards = []
                for obj, class_name in cropped_objs:
                    # Step 1: remove background
                    obj_clean = remove_bg(obj)

                    # Step 2: extract embedding
                    embedding = extract_img(obj_clean)

                    # Step 3: find complementary items (not similar ones)
                    comp_paths = complementary_search(embedding, cat_indices, class_name, k=3)
                    all_boards.extend(comp_paths)

                if all_boards:
                    rec_fig = visualize_outfits(all_boards)
                    st.pyplot(rec_fig)
                else:
                    st.warning("No complementary items found for the detected garment type.")

if __name__ == "__main__":
    main()