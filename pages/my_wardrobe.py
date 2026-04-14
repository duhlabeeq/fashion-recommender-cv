import os
import pickle

import streamlit as st

WARDROBE_FILE = "wardrobe.pkl"
CATEGORIES = ["Shirts", "Pants", "Shoes"]

st.set_page_config(page_title="My Wardrobe", page_icon=":closet:", layout="wide")

st.markdown("# :closet: My Wardrobe")
st.write("All items you have saved, organised by category.")
st.divider()

def load_wardrobe():
    if os.path.exists(WARDROBE_FILE):
        with open(WARDROBE_FILE, "rb") as f:
            return pickle.load(f)
    return {cat: [] for cat in CATEGORIES}

wardrobe = load_wardrobe()
total = sum(len(wardrobe.get(c, [])) for c in CATEGORIES)

if total == 0:
    st.info("Your wardrobe is empty! Upload an image on the Home page and click **Save to Gallery** to start building your collection.", icon="👗")
else:
    st.markdown(f"**{total} item{'s' if total != 1 else ''} saved**")
    st.divider()

    icons = {"Shirts": "👕", "Pants": "👖", "Shoes": "👟"}

    def save_wardrobe(data):
        with open(WARDROBE_FILE, "wb") as f:
            pickle.dump(data, f)

    for category in CATEGORIES:
        items = [p for p in wardrobe.get(category, []) if os.path.exists(p)]
        if not items:
            continue

        st.markdown(f"### {icons.get(category, '')} {category} &nbsp; <span style='font-size:0.85rem; color:grey;'>({len(items)} items)</span>", unsafe_allow_html=True)

        cols = st.columns(6)
        for i, path in enumerate(items):
            with cols[i % 6]:
                st.image(path, use_column_width=True)
                if st.button("Use", key=f"use_{category}_{i}", use_container_width=True):
                    import pickle as _pkl
                    _sel = {"path": path, "category": category}
                    with open("wardrobe_selection.pkl", "wb") as _f:
                        _pkl.dump(_sel, _f)
                    st.success("Item selected! Click **home** in the sidebar to continue.")

                if st.button("🗑️ Remove", key=f"remove_{category}_{i}", use_container_width=True):
                    wardrobe[category].remove(path)
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    save_wardrobe(wardrobe)
                    st.rerun()

        st.divider()
