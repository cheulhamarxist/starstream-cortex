# Sign-Language-to-Text-Conversion

## Run locally

Use Python 3.11. From this directory, create and activate a virtual environment,
then install the project dependencies and start Flask:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. On the first run, the app trains a model from the
images in `data/` and saves it as `modelnet_model.h5`; this can take a while.
Later runs load that saved model. In VS Code, select this virtual environment as
the Python interpreter. Data, model, and upload paths are resolved relative to
`app.py`, so you can launch it from another working directory.

The included model recognizes static hand signs for digits 0–9; it does not
translate continuous signing or sentences. Its training set contains 69 images
per digit, so accuracy can vary with camera angle, lighting, and background.
For best results, keep one hand centered and fully visible.

## Run with Docker Compose

```powershell
docker compose up --build
```

The app is available at `http://127.0.0.1:5000`. The first run trains the model,
so allow time for startup.