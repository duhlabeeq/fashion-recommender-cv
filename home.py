import os
import pickle
import uuid
from datetime import datetime

import numpy as np
import streamlit as st
from obj_detection import ObjDetection
from PIL import Image

from src.utilities import (ExactIndex, extract_img, visualize_outfits,
                           remove_bg, build_category_indices, complementary_search)

GALLERY_DATA_DIR = "gallery_data"
GALLERY_HISTORY_FILE = "gallery_history.pkl"

def load_gallery_history():
    if os.path.exists(GALLERY_HISTORY_FILE):
        with open(GALLERY_HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    return []

def save_gallery_history(history):
    with open(GALLERY_HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)

def save_session_to_gallery(input_array, detected_objs, rec_paths):
    os.makedirs(GALLERY_DATA_DIR, exist_ok=True)
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Save uploaded image (from numpy array stored in session state)
    input_path = os.path.join(GALLERY_DATA_DIR, f"{session_id}_input.jpg")
    Image.fromarray(input_array).save(input_path)

    # Save detected (bg-removed) items
    detected_paths = []
    for i, (arr, cls) in enumerate(detected_objs):
        det_path = os.path.join(GALLERY_DATA_DIR, f"{session_id}_detected_{i}_{cls}.jpg")
        Image.fromarray(arr).save(det_path)
        detected_paths.append((det_path, cls))

    # Copy recommendation images into gallery_data so they're self-contained
    rec_saved = []
    for j, src in enumerate(rec_paths):
        cat = os.path.basename(os.path.dirname(src))
        dst = os.path.join(GALLERY_DATA_DIR, f"{session_id}_rec_{j}_{cat}.jpg")
        try:
            Image.open(src).convert("RGB").save(dst)
            rec_saved.append((dst, cat))
        except Exception:
            pass

    entry = {
        "id": session_id,
        "timestamp": timestamp,
        "input_path": input_path,
        "detected": detected_paths,
        "rec_items": rec_saved,  # list of (path, category)
    }

    history = load_gallery_history()
    history.insert(0, entry)  # newest first
    save_gallery_history(history)


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
    # Session state for 2-step flow
    if 'detected_objs' not in st.session_state:
        st.session_state.detected_objs = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'input_array' not in st.session_state:
        st.session_state.input_array = None

    uploaded = upload_image()

    if uploaded:
        image_obj = Image.open(uploaded)
        # Store as numpy array so it survives button reruns
        st.session_state.input_array = np.array(image_obj.convert("RGB"))
        st.image(image_obj)

        # --- Step 1: Detect + Remove Background ---
        if st.button('Detect Items'):
            with st.spinner("Detecting items and removing backgrounds..."):
                image_array = np.array(image_obj.convert("RGB"))
                cropped_objs = yolo.crop_objects(image_array)
                if cropped_objs:
                    cropped_objs = [(obj, cls) for obj, cls in cropped_objs if obj.size > 0]
                if cropped_objs:
                    # Remove background and filter out near-empty results
                    cleaned = []
                    for obj, cls in cropped_objs:
                        clean = remove_bg(obj)
                        # Skip if less than 5% of pixels are non-white (empty crop)
                        non_white = np.mean(clean < 250)
                        if non_white > 0.05:
                            cleaned.append((clean, cls))
                    st.session_state.detected_objs = cleaned if cleaned else None
                    st.session_state.recommendations = None
                    if not cleaned:
                        st.warning("No valid fashion items detected.")
                else:
                    st.session_state.detected_objs = None
                    st.warning("No fashion items detected in this image.")

        # --- Show bg-removed items + Step 2 button ---
        if st.session_state.detected_objs:
            labels = [cls for _, cls in st.session_state.detected_objs]
            st.success(f"Detected: **{', '.join(labels)}**")

            cols = st.columns(len(st.session_state.detected_objs))
            for col, (obj_clean, cls) in zip(cols, st.session_state.detected_objs):
                col.image(obj_clean, caption=cls)

            st.divider()

            if st.button('Show Recommendations', disabled=not INDEX_READY):
                with st.spinner("Finding complementary items..."):
                    all_boards = []
                    for obj_clean, class_name in st.session_state.detected_objs:
                        embedding = extract_img(obj_clean)
                        comp_paths = complementary_search(embedding, cat_indices, class_name)
                        all_boards.extend(comp_paths)
                    st.session_state.recommendations = all_boards[:6]

            if st.session_state.get('recommendations'):
                col_title, col_btn = st.columns([3, 1])
                with col_title:
                    st.markdown("#### Recommended Items")
                with col_btn:
                    if st.button("💾 Save to Gallery", use_container_width=True):
                        save_session_to_gallery(
                            st.session_state.input_array,
                            st.session_state.detected_objs,
                            st.session_state.recommendations
                        )
                        st.success("Saved! Check the Gallery page.", icon="✅")

                rec_cols = st.columns(6)
                for col, path in zip(rec_cols, st.session_state.recommendations):
                    with col:
                        st.image(path, use_column_width=True)
                        cat = os.path.basename(os.path.dirname(path))
                        st.caption(cat)

if __name__ == "__main__":
    main()