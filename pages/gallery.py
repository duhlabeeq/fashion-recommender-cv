import os
import pickle

import streamlit as st

st.set_page_config(page_title="Gallery", page_icon=":shopping_bags:")

st.markdown("# :female_fairy: :frame_with_picture:")
st.markdown("# :rainbow[App Gallery] :magic_wand:")
st.divider()

# --- Sample outfits section ---
st.markdown("### Sample Outfits")
st.write("Try uploading one of these on the Home page.")

outfits = [
    ("Pink & White",    "gallery/sample_query/pink-white/pw_1.jpg"),
    ("Black Coat",      "gallery/sample_query/black-coat/bc_1.jpg"),
    ("Sweater & Skirt", "gallery/sample_query/sweater-skirt/ss_1.jpg"),
    ("Black Jacket",    "gallery/sample_query/black-jacket/bk_1.jpg"),
]

cols = st.columns(4)
for col, (label, path) in zip(cols, outfits):
    with col:
        st.image(path, use_column_width=True)
        st.caption(label)

st.divider()

# --- Saved sessions section ---
GALLERY_HISTORY_FILE = "gallery_history.pkl"

def load_gallery_history():
    if os.path.exists(GALLERY_HISTORY_FILE):
        with open(GALLERY_HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    return []

history = load_gallery_history()

if not history:
    st.info("No saved sessions yet. Use the **Home** page and click **Save to Gallery**.", icon="👈🏼")
else:
    st.markdown("### Saved Sessions")
    for entry in history:
        st.markdown(f"**{entry['timestamp']}**")

        # Row: input | detected items | 6 recommendations
        input_col, det_cols_area, rec_cols_area = st.columns([1, 1, 6])

        with input_col:
            if os.path.exists(entry["input_path"]):
                st.image(entry["input_path"], caption="Input", use_column_width=True)

        with det_cols_area:
            for det_path, cls in entry.get("detected", []):
                if os.path.exists(det_path):
                    st.image(det_path, caption=cls, use_column_width=True)

        with rec_cols_area:
            rec_items = entry.get("rec_items", [])
            if rec_items:
                rec_subcols = st.columns(len(rec_items))
                for col, (path, cat) in zip(rec_subcols, rec_items):
                    with col:
                        if os.path.exists(path):
                            st.image(path, caption=cat, use_column_width=True)

        st.divider()
