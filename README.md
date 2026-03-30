# Amazon ML Challenge 2025

**Top 10% finish** — scored 50 against the top score of 38, out of 7,000+ teams.

The task: extract structured entity values (weight, dimensions, voltage, wattage, etc.) from Amazon product images and metadata. We had 3 days on paper, but power cuts and hardware issues knocked that down to roughly 30 hours of actual work.

---

## Collaborators

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/coolboy-dev">
        <img src="https://github.com/coolboy-dev.png" width="80px;" alt="Yug"/><br />
        <sub><b>Yug</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/PRATEEK-260">
        <img src="https://github.com/PRATEEK-260.png" width="80px;" alt="Prateek"/><br />
        <sub><b>Prateek</b></sub>
      </a>
    </td>
  </tr>
</table>

---

## Approaches

### Main approach — DINOv2 + MiniLM multimodal regression

- Downloaded and preprocessed ~75K product images
- Extracted image embeddings using **DINOv2** (ViT-S/14, Facebook) via batch processing on Colab GPU
- Extracted text embeddings from product titles and bullet points using **all-MiniLM-L6-v2** (sentence-transformers)
- Fused both embedding vectors into a single multimodal dataset
- Trained a regression head (small MLP) on top of the fused embeddings to predict entity values
- Ran inference on the test set and formatted output to match the required submission format

### Attempt 2 — CLIP + LightGBM

- Used **OpenAI CLIP (ViT-B/32)** to jointly encode image and text into a shared embedding space
- Applied log-transform on target values before training to handle skewed distributions
- Trained a **LightGBM** GBDT model on the CLIP embeddings with GPU acceleration
- Inverse-transformed predictions at inference time

---

## Stack

- Python, PyTorch, HuggingFace Transformers
- DINOv2, CLIP (ViT-B/32), sentence-transformers (MiniLM)
- LightGBM, scikit-learn
- Google Colab (T4 GPU), Google Drive for storage

---

## Results

| Metric | Value |
|---|---|
| Final score | 50 |
| Top team score | 38 |
| Percentile | Top 10% |
| Effective dev time | ~30 hours |

Lower score = better (F1-based evaluation).

---

## Files

| File | Description |
|---|---|
| `Main.ipynb` | DINOv2 + MiniLM embeddings → MLP regression head |
| `Attempt 2 (fine tuning direct transformers).ipynb` | CLIP embeddings → LightGBM |

Both notebooks run on Google Colab and expect data mounted from Google Drive at `/content/drive/MyDrive/amazon/`.
