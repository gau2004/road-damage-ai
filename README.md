# Deep Learning-Based Road Damage Classification and Interactive Model Retraining System

An M.Tech AIML project: a Flask web app that classifies road images as
**Broken Road** or **Not Broken Road** using a MobileNetV2 transfer-learning
model, with a **Configure & Retrain** page that lets you change
hyperparameters and retrain the model live, with real accuracy/loss graphs
and a confusion matrix generated from the actual training run.

## 1. Requirements

- Python 3.10 or 3.11 (TensorFlow 2.16 does not support 3.12+ well)
- pip

## 2. Setup

```bash
# 1. Unzip and enter the project
cd road_damage_project

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 3. Add the dataset

Download the **Kaggle Road Images Dataset** (Broken Road / Not Broken Road)
and place the images like this:

```
road_damage_project/
└── dataset/
    ├── Broken Road/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── Not Broken Road/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Delete the two placeholder `.txt` files inside those folders once your
images are in place. Folder names must match exactly (including capitalization
and the space) — the folder names become the class labels.

## 4. Train the first model

You have two options:

**Option A — from the command line (recommended for the first run):**

```bash
cd model
python train.py --dataset_dir ../dataset --epochs 10
cd ..
```

This trains the model and saves it to `model/saved_model/road_model.keras`.

**Option B — from the web app:**

Start the app (step 5 below) and use the **Configure & Retrain** page to
train the first model directly from your browser.

## 5. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- **Home** — overview and status of whether a model is loaded.
- **Predict** — upload a road image and get a Broken / Not Broken prediction
  with a confidence score.
- **Configure & Retrain** — change optimizer, activation, learning rate,
  batch size, epochs, dropout, image size, and data augmentation, then
  retrain the model and see live accuracy/loss graphs, confusion matrix,
  precision, recall, F1-score and training time — all computed from the
  actual run, not hardcoded.

## 6. Project structure

```
road_damage_project/
├── app.py                     # Flask backend (predict + retrain routes)
├── requirements.txt
├── README.md
├── model/
│   ├── model_builder.py       # MobileNetV2 transfer-learning model
│   ├── train.py                # standalone CLI training script
│   └── saved_model/            # road_model.keras + class_indices.json land here
├── utils/
│   ├── preprocess.py           # ImageDataGenerators (resize, normalize, augment)
│   └── metrics.py              # accuracy/loss plots, confusion matrix, precision/recall/F1
├── dataset/
│   ├── Broken Road/             # <-- put your images here
│   └── Not Broken Road/         # <-- put your images here
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── retrain.html
└── static/
    ├── css/style.css
    ├── js/script.js
    ├── uploads/                 # user-uploaded prediction images (auto-created)
    └── results/                 # generated graphs (auto-created)
```

## 7. Notes for your report

- **Model**: MobileNetV2 (ImageNet weights, frozen backbone) +
  GlobalAveragePooling2D + Dense(128) + Dropout + Dense(1, sigmoid).
- **Loss**: binary cross-entropy (2-class problem).
- **Metrics produced per run**: training/validation accuracy, training/validation
  loss, precision, recall, F1-score, confusion matrix, training time.
- All metrics and graphs are generated fresh on every retrain — good for
  showing "before vs after" hyperparameter comparisons in your project report.
- If you don't have a GPU, keep epochs modest (5–15) and image size at 128–224;
  training runs on CPU but will be slower.

## 8. Troubleshooting

- **"No trained model found yet"** on the Predict page — train a model first
  (see step 4).
- **"Dataset not found"** on retrain — check that `dataset/Broken Road/` and
  `dataset/Not Broken Road/` both contain image files.
- **TensorFlow install issues** — make sure you're on Python 3.10/3.11; on
  Apple Silicon Macs you may need `tensorflow-macos` instead of `tensorflow`.
