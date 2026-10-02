# Plant Disease Detection Web App

This folder is a Flask website that uses the **actual trained MobileNetV2 model** from the Plant Disease Detection project. It does not invent class labels or fake prediction scores.

Project title: **Plant Disease Detection Using CNN and Transfer Learning**

## What was found

The Cursor workspace `ML PROJECT 22` was empty. The trained files used by this app were located elsewhere on the same computer and copied into `web_app/models/` (originals were not modified):

| Item | Location used |
| --- | --- |
| Trained model | `Downloads/plant_disease_mobilenetv2.keras` → `web_app/models/plant_disease_mobilenetv2.keras` |
| Class labels | Notebook `train_ds.class_names` order, also `Desktop/PlantDiseaseProject__/class_names.txt` |
| Training notebook | `Downloads/Plant_Disease_Detection_CNN_Transfer_Learning_UPDATED.ipynb` |

The notebook reported **test accuracy 92.29%** and **test loss 0.2192**. It does **not** contain a `model.save(...)` cell. The `.keras` file already existed in Downloads and is the file this app loads.

## Preprocessing (matched to training)

Training used `tf.keras.utils.image_dataset_from_directory` with `image_size=(224, 224)` and RGB images. Pixel values stayed in **0–255**. `mobilenet_v2.preprocess_input` is **already inside the saved model** (`TrueDivide` then `Subtract`). The web app therefore:

1. Opens the upload as RGB
2. Resizes to 224×224 with bilinear interpolation
3. Converts to `float32` in 0–255
4. Does **not** call `preprocess_input` again

Class order is the 38 PlantVillage folder names returned by Keras (alphabetical folder order), not a hand-made list.

## Folder layout

```
web_app/
  app.py
  requirements.txt
  README.md
  templates/index.html
  static/style.css
  static/script.js
  models/
    plant_disease_mobilenetv2.keras
    class_names.txt
```

## Run on macOS / Linux

```bash
cd web_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in a browser. Upload a leaf JPG/PNG and click **Predict Disease**.

## Run on Windows PowerShell

```powershell
cd web_app
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Open http://127.0.0.1:5000

## Optional environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `MODEL_PATH` | Path to the `.keras` file | `web_app/models/plant_disease_mobilenetv2.keras` |
| `CLASS_NAMES_PATH` | Path to class label text file | `web_app/models/class_names.txt` |
| `PORT` | Flask port | `5000` |

macOS/Linux example:

```bash
export MODEL_PATH="/full/path/to/plant_disease_mobilenetv2.keras"
python app.py
```

PowerShell example:

```powershell
$env:MODEL_PATH = "C:\path\to\plant_disease_mobilenetv2.keras"
python app.py
```

## If the model is missing

Do not create a dummy model. Copy your real file here:

`web_app/models/plant_disease_mobilenetv2.keras`

Because the Colab notebook never saved the model, run this in the **same Colab session** where `model` is already trained:

```python
model.save("/content/plant_disease_mobilenetv2.keras")

from google.colab import files
files.download("/content/plant_disease_mobilenetv2.keras")
```

Also save labels in the same order as training:

```python
from pathlib import Path
Path("/content/class_names.txt").write_text("\n".join(class_names))
files.download("/content/class_names.txt")
```

## API

- `GET /` — website
- `GET /health` — whether the model and labels loaded
- `POST /predict` — form field `image` (JPG/JPEG/PNG). JSON includes `predicted_class`, `confidence` (softmax score × 100), and `status` (`Healthy` / `Diseased`).

Confidence is the model's output probability for the top class. It is **not** a guarantee of correctness.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `Trained model file not found` | Put the `.keras` file in `web_app/models/` or set `MODEL_PATH` |
| `No module named tensorflow` / Flask / PIL | Activate the venv and run `pip install -r requirements.txt` |
| Frontend error about JSON / connection | The Flask process is not running. Start `python app.py` first |
| Wrong class names | Do not alphabetize the file yourself; keep the notebook `class_names` order |
| macOS TensorFlow install issues | Use Python 3.10–3.12. TensorFlow does not support every Python version |

The website will show an error if the backend or model is unavailable. It will not invent a successful prediction.
