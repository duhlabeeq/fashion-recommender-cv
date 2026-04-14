<h2 align="center">StyleSenseAI — Personalized Fashion Recommendation System</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Project-StyleSenseAI-informational" alt="Project Name">
  <img src="https://img.shields.io/badge/Tech-FashionCLIP%20%7C%20YOLOv5%20%7C%20FAISS-blue" alt="Core Tech">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red" alt="Streamlit">
</p>

---

## Overview

**StyleSenseAI** is a computer vision powered web application that takes an image of a fashion item and recommends a complete, color-matched outfit. Upload a shirt, pants, or shoes — the system detects the item, removes its background, and returns 6 complementary fashion pieces from a catalog of 34,000+ product images.

---

## How It Works

1. **Upload** an image containing a clothing item
2. **Detect** — YOLOv5 identifies and crops the fashion item; background is removed automatically
3. **Correct** — confirm or change the detected category (Shirts / Pants / Shoes)
4. **Recommend** — FashionCLIP embeds the item; FAISS searches for the best color-complementary matches across 6 outfit categories

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv5 (ONNX, 21 fashion classes) |
| Feature Extraction | FashionCLIP (`patrickjohncyh/fashion-clip`) |
| Similarity Search | FAISS IndexFlatL2 |
| Background Removal | rembg (U2Net) |
| Frontend | Streamlit |
| Dataset | Kaggle Fashion Product Images (~34K images, 14 categories) |

---

## Project Structure

```
fashion-recommender-cv/
│
├── home.py                  # Main Streamlit page — upload, detect, recommend
├── obj_detection.py         # YOLOv5 ONNX wrapper for object detection
│
├── pages/
│   ├── gallery.py           # Gallery page — sample outfits + saved sessions
│   └── my_wardrobe.py       # Wardrobe page — saved items by category
│
├── src/
│   └── utilities.py         # FashionCLIP embedding, FAISS search, color matching, complement rules
│
├── models/
│   ├── best.onnx            # YOLOv5 ONNX model (fine-tuned on fashion)
│   └── data.yaml            # YOLO class labels config
│
├── pipeline/
│   ├── organize_dataset.py  # Organises flat Kaggle dataset into category subfolders
│   └── build_index.py       # Builds FashionCLIP embeddings + FAISS index
│
├── fashion-dataset/         # Organised dataset (14 category subfolders)
├── gallery/                 # Sample query images for the gallery page
│
├── embeddings.pkl           # FashionCLIP embeddings for all dataset images
├── flatIndex.index          # FAISS index file
├── img_paths.pkl            # Image paths corresponding to embeddings
│
├── gallery_data/            # Saved gallery session images (auto-generated)
├── gallery_history.pkl      # Gallery session history (auto-generated)
├── wardrobe_images/         # Saved wardrobe item images (auto-generated)
├── wardrobe.pkl             # Wardrobe index by category (auto-generated)
│
├── requirements.txt         # Python dependencies
└── packages.txt             # System packages (for Streamlit Cloud deployment)
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/duhlabeeq/fashion-recommender-cv.git
cd fashion-recommender-cv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the FAISS index (first time only)
Download the [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset, then run:
```bash
python pipeline/organize_dataset.py --csv archive/styles.csv --images archive/images --output fashion-dataset
python pipeline/build_index.py --dataset fashion-dataset
```

### 4. Run the app
```bash
streamlit run home.py
```

---

## Features

- **2-step flow** — detect first, recommend second
- **Background removal** — clean product-style crops automatically
- **Color-matched recommendations** — fashion color theory (complementary hues, neutral pairings)
- **Category correction** — override wrong YOLO detections via dropdown
- **My Wardrobe** — save and browse your uploaded items by category (Shirts / Pants / Shoes)
- **Gallery** — save and revisit past outfit recommendation sessions

---

## Complement Rules

| Input | Recommended (6 items) |
|---|---|
| Shirts | 2 Pants + 1 Belt + 1 Shoes + 1 Watch + 1 Sunglasses |
| Pants | 2 Shirts + 1 Belt + 1 Shoes + 1 Watch + 1 Sunglasses |
| Shoes | 2 Shirts + 1 Pants + 1 Belt + 1 Watch + 1 Sunglasses |
