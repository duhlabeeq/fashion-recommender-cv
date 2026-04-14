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
WARDROBE_DIR = "wardrobe_images"
WARDROBE_FILE = "wardrobe.pkl"

def load_wardrobe():
    if os.path.exists(WARDROBE_FILE):
        with open(WARDROBE_FILE, "rb") as f:
            return pickle.load(f)
    return {"Shirts": [], "Pants": [], "Shoes": []}

def save_wardrobe(wardrobe):
    with open(WARDROBE_FILE, "wb") as f:
        pickle.dump(wardrobe, f)

def add_to_wardrobe(detected_objs, corrected_classes):
    """Save bg-removed input images into wardrobe by category."""
    os.makedirs(WARDROBE_DIR, exist_ok=True)
    wardrobe = load_wardrobe()
    for i, (arr, _) in enumerate(detected_objs):
        category = corrected_classes[i] if i < len(corrected_classes) else None
        if category not in ("Shirts", "Pants", "Shoes"):
            continue
        filename = f"{category}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(WARDROBE_DIR, filename)
        Image.fromarray(arr).save(path)
        if path not in wardrobe[category]:
            wardrobe[category].append(path)
    save_wardrobe(wardrobe)

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

    # Save input image
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
        "rec_items": rec_saved,
    }

    history = load_gallery_history()
    history.insert(0, entry)
    save_gallery_history(history)


# --- UI Configurations --- #
st.set_page_config(page_title="StyleSenseAI - Your personal AI Stylist",
                   page_icon=":shopping_bags:"
                   )

st.markdown("# :rainbow[StyleSenseAI]")
st.markdown("### Your personal AI Stylist")

# --- Message --- #
st.write("StyleSenseAI is a computer vision powered web-app that lets you upload an image of an outfit and returns complementary style recommendations. An image with a white background works best.")

st.markdown("""
**Built with:**
- **YOLOv5** — object detection to identify fashion items in your image
- **FashionCLIP** — vision-language model for fashion-aware image embeddings
- **FAISS** — fast similarity search across 34,000+ fashion product images
- **rembg** — automatic background removal for clean item detection
- **Streamlit** — interactive web interface
""")
st.divider()
st.info("Check out the gallery in the sidebar to get some ideas", icon="👈")

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
    if 'corrected_classes' not in st.session_state:
        st.session_state.corrected_classes = []

    # --- Check if coming from My Wardrobe (file-based handoff) ---
    # Read from file once, then keep in session state for all subsequent reruns
    if os.path.exists("wardrobe_selection.pkl"):
        try:
            with open("wardrobe_selection.pkl", "rb") as _f:
                _sel = pickle.load(_f)
            st.session_state.wardrobe_image = np.array(Image.open(_sel["path"]).convert("RGB"))
            st.session_state.wardrobe_category = _sel["category"]
            st.session_state.detected_objs = None
            st.session_state.recommendations = None
            st.session_state.corrected_classes = []
        except Exception:
            pass
        os.remove("wardrobe_selection.pkl")

    wardrobe_image = st.session_state.get('wardrobe_image')
    wardrobe_category = st.session_state.get('wardrobe_category')

    uploaded = upload_image()

    # Determine the active image: wardrobe item takes priority
    if wardrobe_image is not None:
        image_array = wardrobe_image
        st.session_state.input_array = image_array
        st.info(f"Loaded from your wardrobe — **{wardrobe_category}**")
        st.image(image_array)

        if st.button('Detect Items'):
            with st.spinner("Detecting items and removing backgrounds..."):
                cropped_objs = yolo.crop_objects(image_array)
                if cropped_objs:
                    cropped_objs = [(obj, cls) for obj, cls in cropped_objs if obj.size > 0]
                if cropped_objs:
                    cleaned = []
                    for obj, cls in cropped_objs:
                        clean = remove_bg(obj)
                        non_white = np.mean(clean < 250)
                        if non_white > 0.05:
                            cleaned.append((clean, cls))
                    # Use the wardrobe category as the pre-selected class
                    if cleaned:
                        st.session_state.detected_objs = cleaned
                        st.session_state.corrected_classes = [wardrobe_category] * len(cleaned)
                        st.session_state.recommendations = None
                    else:
                        st.warning("No valid fashion items detected.")
                else:
                    st.warning("No fashion items detected in this image.")

    elif uploaded:
        # Clear any wardrobe selection when user uploads a fresh image
        st.session_state.wardrobe_image = None
        st.session_state.wardrobe_category = None
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
                    cleaned = []
                    for obj, cls in cropped_objs:
                        clean = remove_bg(obj)
                        non_white = np.mean(clean < 250)
                        if non_white > 0.05:
                            cleaned.append((clean, cls))
                    st.session_state.detected_objs = cleaned if cleaned else None
                    st.session_state.recommendations = None
                    # Initialise corrected classes from YOLO detection
                    st.session_state.corrected_classes = [cls for _, cls in cleaned] if cleaned else []
                    if not cleaned:
                        st.warning("No valid fashion items detected.")
                else:
                    st.session_state.detected_objs = None
                    st.warning("No fashion items detected in this image.")

    # --- Show bg-removed items + Step 2 button ---
    VALID_CLASSES = ['Shirts', 'Pants', 'Shoes']

    if st.session_state.detected_objs:
        # Pad corrected_classes if needed
        while len(st.session_state.corrected_classes) < len(st.session_state.detected_objs):
            st.session_state.corrected_classes.append(st.session_state.detected_objs[len(st.session_state.corrected_classes)][1])

        cols = st.columns(len(st.session_state.detected_objs))
        for i, (col, (obj_clean, _)) in enumerate(zip(cols, st.session_state.detected_objs)):
            with col:
                st.image(obj_clean, use_column_width=True)
                current = st.session_state.corrected_classes[i]
                # If detected class not in our 3 supported types, prompt user to pick
                if current in VALID_CLASSES:
                    chosen = st.selectbox("Category", VALID_CLASSES, index=VALID_CLASSES.index(current), key=f"cls_{i}")
                else:
                    chosen = st.selectbox("Choose a category", VALID_CLASSES, index=None, key=f"cls_{i}", placeholder="Choose a category...")
                st.session_state.corrected_classes[i] = chosen

        st.divider()

        all_selected = all(c is not None for c in st.session_state.corrected_classes)
        if st.button('Show Recommendations', disabled=not INDEX_READY or not all_selected):
            with st.spinner("Finding complementary items..."):
                all_boards = []
                for i, (obj_clean, _) in enumerate(st.session_state.detected_objs):
                    class_name = st.session_state.corrected_classes[i]
                    embedding = extract_img(obj_clean)
                    comp_paths = complementary_search(embedding, cat_indices, class_name, query_image_array=obj_clean)
                    all_boards.extend(comp_paths)
                st.session_state.recommendations = all_boards[:6]

    if st.session_state.get('recommendations'):
        col_title, col_btn = st.columns([3, 1])
        with col_title:
            st.markdown("#### Recommended Items")
        with col_btn:
            can_save = st.session_state.input_array is not None and st.session_state.detected_objs is not None
            if st.button("💾 Save to Gallery", use_container_width=True, disabled=not can_save):
                save_session_to_gallery(
                    st.session_state.input_array,
                    st.session_state.detected_objs,
                    st.session_state.recommendations
                )
                add_to_wardrobe(
                    st.session_state.detected_objs,
                    st.session_state.corrected_classes
                )
                st.success("Saved! Check the Gallery & My Wardrobe pages.", icon="✅")

        rec_cols = st.columns(6)
        for col, path in zip(rec_cols, st.session_state.recommendations):
            with col:
                st.image(path, use_column_width=True)
                cat = os.path.basename(os.path.dirname(path))
                label = 'Shirts' if cat == 'Shirts & Tops' else cat
                st.caption(label)

if __name__ == "__main__":
    main()
