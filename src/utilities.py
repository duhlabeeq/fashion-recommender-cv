import colorsys
import faiss
import random
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image
from rembg import remove

# ---------------------------------------------------------------------------
# FashionCLIP — lazy-loaded via transformers (no fashion-clip package needed)
# ---------------------------------------------------------------------------
_model = None
_processor = None
MODEL_ID = 'patrickjohncyh/fashion-clip'

def _get_model():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _processor = CLIPProcessor.from_pretrained(MODEL_ID)
        _model = CLIPModel.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _processor


def extract_img(image_array):
    """Extract a 512-dim FashionCLIP embedding from a numpy RGB image."""
    import torch
    model, processor = _get_model()
    pil_img = Image.fromarray(image_array).convert("RGB")
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        embedding = model.vision_model(pixel_values=inputs['pixel_values']).pooler_output
    return embedding.cpu().numpy().astype('float32')


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------
def remove_bg(image_array):
    """Remove background, return RGB numpy array on white background."""
    pil_img = Image.fromarray(image_array)
    no_bg = remove(pil_img)                          # RGBA
    white_bg = Image.new("RGB", no_bg.size, (255, 255, 255))
    white_bg.paste(no_bg, mask=no_bg.split()[3])     # alpha as mask
    return np.array(white_bg)


# ---------------------------------------------------------------------------
# Complement rules: YOLO class → index categories to search
# ---------------------------------------------------------------------------
COMPLEMENTS = {
    'Shirts': [('Pants', 2), ('Belts', 1), ('Shoes', 1), ('Watches', 1), ('Sunglasses', 1)],
    'Pants':  [('Shirts & Tops', 2), ('Belts', 1), ('Shoes', 1), ('Watches', 1), ('Sunglasses', 1)],
    'Shoes':  [('Shirts & Tops', 2), ('Pants', 1), ('Belts', 1), ('Watches', 1), ('Sunglasses', 1)],
}

# YOLO label → dataset category name when they differ
YOLO_TO_INDEX_CAT = {
    'Shirts':  'Shirts & Tops',
    'Coats':   'Coats & Jackets',
    'Scarves': 'Scarves & Shawls',
}


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------
def get_category_from_path(path):
    """
    Extract category from path. Supports two conventions:
      1. dataset/CategoryName/image.jpg   → parent folder name
      2. {hash}_{Category}_{id}.jpg       → underscore-encoded name
    """
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        parent = parts[-2]
        # If parent is a meaningful category dir (not 'index_images'), use it
        if parent not in ('index_images', '.', ''):
            return parent
    # Fall back to filename encoding: {hash}_{Category}_{id}.jpg
    name = os.path.basename(path).rsplit('.', 1)[0]
    tokens = name.split('_')
    return '_'.join(tokens[1:-1])


def build_category_indices(embeddings, image_paths):
    """Build a per-category FAISS sub-index dict at startup (one-time cost)."""
    category_map = {}
    for i, path in enumerate(image_paths):
        cat = get_category_from_path(path)
        category_map.setdefault(cat, []).append(i)

    cat_indices = {}
    for cat, idxs in category_map.items():
        vecs = embeddings[idxs].astype('float32')
        paths = [image_paths[i] for i in idxs]
        idx = faiss.IndexFlatL2(vecs.shape[1])
        idx.add(vecs)
        cat_indices[cat] = (idx, paths)
    return cat_indices


# ---------------------------------------------------------------------------
# Color-based reranking helpers
# ---------------------------------------------------------------------------
def _dominant_color_array(image_array, size=60):
    """Dominant non-white color from a numpy RGB array."""
    img = Image.fromarray(image_array).convert("RGB").resize((size, size))
    pixels = np.array(img).reshape(-1, 3).astype(float)
    mask = ~np.all(pixels > 228, axis=1)   # ignore near-white bg
    pixels = pixels[mask]
    if len(pixels) < 5:
        return None
    return pixels.mean(axis=0)


def _dominant_color_path(path, size=60):
    """Dominant non-white color from an image file."""
    try:
        img = Image.open(path).convert("RGB").resize((size, size))
        pixels = np.array(img).reshape(-1, 3).astype(float)
        mask = ~np.all(pixels > 228, axis=1)
        pixels = pixels[mask]
        if len(pixels) < 5:
            return None
        return pixels.mean(axis=0)
    except Exception:
        return None


def _to_hsv(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360.0, s, v


def _hue_dist(h1, h2):
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def _fashion_score(query_rgb, candidate_rgb):
    """
    Returns a score where lower = better color pairing.
    Rules:
      - Neutral input (grey/white/black) → prefer saturated candidates
      - Colorful input → prefer neutrals or complementary hue (~180° apart)
    """
    qh, qs, _ = _to_hsv(query_rgb)
    ch, cs, _ = _to_hsv(candidate_rgb)

    if qs < 0.20:
        # Neutral input: reward candidate saturation
        return 1.0 - cs
    else:
        # Colorful input
        if cs < 0.25:
            # Neutral candidate always pairs well
            return 0.0
        # Reward complementary hue (180° apart = best, 0° = worst)
        hue_dist = _hue_dist(qh, ch)
        return 1.0 - (hue_dist / 180.0)


def complementary_search(query_vector, cat_indices, class_name, query_image_array=None):
    """Return complementary item paths reranked by fashion color theory."""
    rules = COMPLEMENTS.get(class_name, [])
    query = query_vector.reshape(1, -1).astype('float32')

    query_rgb = None
    if query_image_array is not None:
        query_rgb = _dominant_color_array(query_image_array)

    results = []
    for cat, count in rules:
        if cat not in cat_indices:
            continue
        idx, paths = cat_indices[cat]

        # Fetch more candidates so we can rerank by color
        n_candidates = min(count * 15, idx.ntotal)
        if n_candidates == 0:
            continue

        _, indices = idx.search(query, n_candidates)
        candidates = [paths[i] for i in indices[0] if i >= 0]

        if query_rgb is not None:
            scored = []
            for p in candidates:
                cand_rgb = _dominant_color_path(p)
                score = _fashion_score(query_rgb, cand_rgb) if cand_rgb is not None else 0.5
                scored.append((score, p))
            scored.sort(key=lambda x: x[0])
            selected = [p for _, p in scored[:count]]
        else:
            selected = candidates[:count]

        results.extend(selected)
    return results


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------
class ExactIndex():
    def __init__(self, vectors, img_paths):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.img_paths = img_paths

    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k)
        return [self.img_paths[i] for i in indices[0]]

    def save(self, filename):
        faiss.write_index(self.index, filename)

    @classmethod
    def load(cls, vectors, img_paths, filename):
        instance = cls(vectors, img_paths)
        instance.index = faiss.read_index(filename)
        return instance


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualize_outfits(boards):
    fig, axs = plt.subplots(4, 2, figsize=(10, 8))
    plt.suptitle("Recommended items based on detected fashion objects", fontsize=14, y=1)

    num_items = min(len(boards), 8)
    random_paths = random.sample(boards, num_items)

    for i, ax in enumerate(axs.flatten()):
        ax.axis('off')
        if i < len(random_paths):
            try:
                img = mpimg.imread(random_paths[i])
                ax.imshow(img)
            except Exception:
                pass
    return fig


def viz_thumbnail(im_path, tn_sz):
    a_img = mpimg.imread(im_path)
    img_height, img_width = a_img.shape[:2]
    max_dim = max(img_height, img_width)
    pad_vert = (max_dim - img_height) // 2
    pad_horiz = (max_dim - img_width) // 2
    padded_img = np.pad(a_img, ((pad_vert, pad_vert), (pad_horiz, pad_horiz), (0, 0)),
                        mode='constant', constant_values=255)
    fig, ax = plt.subplots(figsize=tn_sz)
    ax.imshow(padded_img)
    ax.axis('off')
    return fig
