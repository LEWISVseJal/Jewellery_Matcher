# Jewellery Matcher

AI-powered visual jewellery search and matching application.

## Overview

Jewellery Matcher allows a user to upload a jewellery image and find
visually similar designs from a catalogue. The project maintains
separate **Gold** and **Prototype** collections and is designed for
cross-collection visual matching.

Example:

- Gold query → Prototype results
- Prototype query → Gold results

The matching pipeline uses image preprocessing, DINOv2 visual
embeddings, and similarity comparison. `rembg` / U2-Net is used for
image segmentation/background removal.

## Features

- Jewellery image upload and preview
- Visual similarity search
- Gold and Prototype catalogues
- Cross-collection matching
- DINOv2 image embeddings
- Image segmentation/background removal
- JSON jewellery catalogue
- NumPy embedding indexes
- Segmented catalogue images
- Add Jewellery web interface
- Responsive web UI
- Flask backend
- HTML/CSS/JavaScript frontend

## Project Structure

```text
Jewellery_Matcher/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── catalogue/
│   │   ├── gold/
│   │   └── prototype/
│   ├── database/
│   │   ├── jewellery.json
│   │   ├── gold_embeddings.npy
│   │   ├── prototype_embeddings.npy
│   │   └── segmented_catalogue/
│   │       ├── gold/
│   │       └── prototype/
│   ├── scripts/
│   │   └── create_embeddings.py
│   ├── services/
│   │   ├── catalogue.py
│   │   ├── embedding.py
│   │   ├── index_manager.py
│   │   ├── matcher.py
│   │   └── segmentation.py
│   └── uploads/
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   └── add_jewellery.html
│   └── static/
│       ├── css/
│       ├── images/
│       └── js/
├── images_G_P/
│   ├── Gold_img/
│   └── Prototype_img/
├── .gitignore
└── README.md
```

## Technology Stack

- Python 3.11
- Flask
- PyTorch
- Transformers
- DINOv2
- rembg
- ONNX Runtime
- NumPy
- Pillow
- HTML5
- CSS3
- JavaScript

## Matching Pipeline

```text
Upload Image
     ↓
Image Validation
     ↓
Segmentation / Background Removal
     ↓
DINOv2 Feature Extraction
     ↓
Image Embedding
     ↓
Collection Handling
     ↓
Similarity Search
     ↓
Similar Jewellery Results
```

The system is intended to focus on visual design characteristics such as
shape, structure, pattern, stone arrangement, and overall jewellery
design rather than relying only on colour.

## Catalogue

Gold jewellery is stored in:

```text
backend/catalogue/gold/
```

Prototype jewellery is stored in:

```text
backend/catalogue/prototype/
```

Catalogue metadata is stored in:

```text
backend/database/jewellery.json
```

## Embeddings

Generated catalogue embeddings are stored in:

```text
backend/database/gold_embeddings.npy
backend/database/prototype_embeddings.npy
```

To regenerate embeddings:

```powershell
python -m backend.scripts.create_embeddings
```

## Segmented Catalogue

Processed catalogue images are stored in:

```text
backend/database/segmented_catalogue/
├── gold/
└── prototype/
```

## Installation

### 1. Clone the repository

```powershell
git clone <repository-url>
cd Jewellery_Matcher
```

### 2. Create the virtual environment

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.venv\Scriptsctivate
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

For the current CPU PyTorch setup:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

## Model Cache

AI model files are intentionally **not included in Git**.

The project uses a local `model_cache/` directory for downloaded models
such as:

- DINOv2
- rembg / U2-Net

The models can be downloaded/generated on the machine after cloning the
repository.

Typical cache structure:

```text
model_cache/
├── huggingface/
└── rembg/
```

`model_cache/` is excluded by `.gitignore`.

## Environment Variables

The project can use:

```text
HF_HOME=<project>\model_cache\huggingface
HF_HUB_CACHE=<project>\model_cache\huggingface\hub
U2NET_HOME=<project>\model_cache\rembg
```

Set paths according to the local machine.

## Running the Application

From the project root:

```powershell
.venv\Scripts\activate
python -m backend.app
```

Open the local Flask address displayed in the terminal.

## Web API

### Match Jewellery

```text
POST /api/match
```

The frontend sends the selected image using the `image` form field.

### Add Jewellery

```text
POST /api/jewellery/add
```

Used by the Add Jewellery page to add catalogue jewellery.

## Supported Image Formats

```text
JPG
JPEG
PNG
WEBP
BMP
```

The current matching interface accepts images up to 20 MB.

## Git

The repository includes the project source, catalogue data, embeddings,
source images, and segmented catalogue.

Large/generated local resources such as the following remain excluded:

```text
model_cache/
.venv/
__pycache__/
```

## Troubleshooting

### Missing embeddings

Regenerate them with:

```powershell
python -m backend.scripts.create_embeddings
```

### First model run

The first DINOv2 or rembg run may download the required model files into
`model_cache/`.

### Windows paging-file error

If Windows reports that the paging file is too small, increase available
virtual memory because CPU-based DINOv2 inference can require
significant committed memory.

## Current Development Scope

The current project focuses on:

1.  Jewellery catalogue management
2.  Image preprocessing
3.  Visual embedding generation
4.  Cross-collection similarity matching
5.  Web-based image search
6.  Add Jewellery workflow

Advanced production features can be added later, including stronger
jewellery-specific matching, improved non-jewellery rejection,
authentication, deployment, and mobile integration.

## Author

**Sejal Lewis**

Jewellery Visual Search / Matching System
