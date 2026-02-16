import io, os, csv, base64, hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
APP_TITLE = "🌿 Agri-LLaVA Pro — Leaf Disease Detector"
MODEL_PATH = os.environ.get("MODEL_PATH", "leaf_model.pth")
USERS_CSV = "users.csv"              # {username,password_hash}
DATA_DIR = "data"                    # per-user prediction history as CSV
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)  # for future assets if you want

# ──────────────────────────────────────────────────────────────────────────────
# Class list (15 classes)
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Tomato__Tomato_mosaic_virus', 'Tomato_Early_blight',
    'Pepper__bell___Bacterial_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Late_blight', 'Tomato_healthy', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Tomato_Septoria_leaf_spot', 'Potato___Late_blight',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Target_Spot',
    'Tomato_Leaf_Mold', 'Tomato_Bacterial_spot', 'Potato___healthy'
]

# Cure dictionary
CURES = {
    'Tomato__Tomato_mosaic_virus': 'Remove infected plants; control aphids and avoid tobacco use near crops.',
    'Tomato_Early_blight': 'Use Mancozeb or Chlorothalonil fungicide; remove affected leaves.',
    'Pepper__bell___Bacterial_spot': 'Apply copper fungicides; avoid wetting foliage.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Use neem oil or miticides; increase humidity.',
    'Tomato_Late_blight': 'Use Ridomil Gold or Metalaxyl; remove infected plants immediately.',
    'Tomato_healthy': 'No disease detected; maintain proper watering and fertilization.',
    'Pepper__bell___healthy': 'Healthy leaf; no treatment needed.',
    'Potato___Early_blight': 'Apply fungicides with chlorothalonil; remove infected foliage.',
    'Tomato_Septoria_leaf_spot': 'Use neem oil or chlorothalonil; remove infected leaves.',
    'Potato___Late_blight': 'Apply copper-based fungicide; destroy infected plants.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Remove infected plants; use resistant varieties.',
    'Tomato__Target_Spot': 'Use Daconil or copper-based sprays; avoid leaf wetness.',
    'Tomato_Leaf_Mold': 'Improve air circulation; apply sulfur-based fungicides.',
    'Tomato_Bacterial_spot': 'Apply copper-based fungicides and avoid overhead watering.',
    'Potato___healthy': 'Healthy potato leaf; no action needed.'
}

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (robust to both state_dict and full model)
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_leaf_model(model_path: str, num_classes: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Place your file in the repo root as '{model_path}' (use Git LFS if > 10MB)."
        )
    try:
        # Try assuming it's a state_dict
        model = build_model(num_classes)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    except Exception:
        # Fallback: maybe the file is a whole serialized model
        model = torch.load(model_path, map_location=device)
        # If it's a scripted model, it may not need eval() set later, but we still call it.
    model.to(device).eval()
    return model

model = load_leaf_model(MODEL_PATH, len(CLASS_NAMES))

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — use the same as training (most common: 224 + ImageNet norm)
# If your training did NOT use normalization, set NORMALIZE=False.
# ──────────────────────────────────────────────────────────────────────────────
NORMALIZE = True  # set False if your training skipped normalization

normalize_tfms = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
if NORMALIZE:
    transform_list.append(normalize_tfms)
transform = transforms.Compose(transform_list)

# ──────────────────────────────────────────────────────────────────────────────
# Auth helpers (CSV “DB”)
# ──────────────────────────────────────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def ensure_users_csv():
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["username", "password_hash"])

def user_exists(username: str) -> bool:
    ensure_users_csv()
    with open(USERS_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["username"] == username:
                return True
    return False

def verify_user(username: str, password: str) -> bool:
    ensure_users_csv()
    hpw = hash_pw(password)
    with open(USERS_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["username"] == username and row["password_hash"] == hpw:
                return True
    return False

def signup_user(username: str, password: str) -> Optional[str]:
    if not username or not password:
        return "Username and password are required."
    if user_exists(username):
        return "Username already exists."
    ensure_users_csv()
    with open(USERS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, hash_pw(password)])
    return None

# ──────────────────────────────────────────────────────────────────────────────
# HTML templates
# ──────────────────────────────────────────────────────────────────────────────
def page_html(body: str, title: str = APP_TITLE) -> str:
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <title>{title}</title>
        <style>
          body {{
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            background: #f1f8e9;
            color: #1b5e20; margin: 0; padding: 0;
          }}
          .container {{ max-width: 880px; margin: 0 auto; padding: 24px; }}
          h1 {{ color: #2e7d32; }}
          .card {{
            background: #fff; border-radius: 14px; padding: 20px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.06);
            margin: 16px 0;
          }}
          input, button {{
            font-size: 16px; padding: 10px 14px; border-radius: 10px; border: 1px solid #c8e6c9;
          }}
          button {{
            background: #43a047; color: #fff; border: none; cursor: pointer;
          }}
          button:hover {{ background: #2e7d32; }}
          .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items:center; }}
          .muted {{ color: #4e5d52; }}
          .pill {{ background:#e8f5e9; padding:6px 10px; border-radius:999px; display:inline-block; }}
          .footer {{ margin-top: 16px; color:#6b7a70; font-size: 13px; }}
          .link {{ color:#1b5e20; text-decoration:none; }}
        </style>
      </head>
      <body>
        <div class="container">
          {body}
          <div class="footer">© {datetime.now().year} Agri-LLaVA Pro</div>
        </div>
      </body>
    </html>
    """

def auth_page(message: str = "") -> str:
    return page_html(f"""
      <h1>🌿 Agri-LLaVA Pro</h1>
      <div class="card">
        <div class="row">
          <div style="flex:1; min-width:280px">
            <h2>🔐 Login</h2>
            <form action="/login" method="post">
              <input name="username" placeholder="Username" required style="width:100%; margin:6px 0" />
              <input name="password" type="password" placeholder="Password" required style="width:100%; margin:6px 0" />
              <button type="submit">Login</button>
            </form>
          </div>
          <div style="flex:1; min-width:280px">
            <h2>🆕 Sign up</h2>
            <form action="/signup" method="post">
              <input name="username" placeholder="Choose username" required style="width:100%; margin:6px 0" />
              <input name="password" type="password" placeholder="Choose password" required style="width:100%; margin:6px 0" />
              <button type="submit">Create account</button>
            </form>
          </div>
        </div>
        <p class="muted">{message}</p>
      </div>
    """)

def app_page(username: str, flash: str = "") -> str:
    return page_html(f"""
      <h1>📸 Smart Leaf Analyzer</h1>
      <p class="pill">Logged in as <b>{username}</b></p>
      {"<p class='muted'>" + flash + "</p>" if flash else ""}

      <div class="card">
        <h2>Upload or Capture a Leaf Image</h2>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input type="hidden" name="username" value="{username}">
          <input type="file" name="file" accept="image/*" capture="environment" required>
          <button type="submit">🔎 Analyze</button>
        </form>
      </div>

      <div class="card">
        <h2>📥 Download your history</h2>
        <p class="muted">Your predictions are saved to CSV automatically.</p>
        <a class="link" download href="/history?user={username}">⬇️ {username}_predictions.csv</a>
      </div>

      <div>
        <a class="link" href="/">↩️ Log out</a>
      </div>
    """)

# ──────────────────────────────────────────────────────────────────────────────
# Damage estimate (quick heuristic)
# ──────────────────────────────────────────────────────────────────────────────
def estimate_damage_percent(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"))
    damaged = np.sum(gray < 120)  # threshold tweakable
    return float(min(100.0, (damaged / gray.size) * 100.0))

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def predict_image(pil_img: Image.Image):
    img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    label = CLASS_NAMES[pred_idx.item()]
    confidence = float(conf.item() * 100.0)
    damage = estimate_damage_percent(pil_img)
    cure = CURES.get(label, "No cure info available.")
    return label, confidence, damage, cure

def save_history(username: str, filename: str, label: str, conf: float, damage: float, cure: str):
    user_csv = os.path.join(DATA_DIR, f"{username}_predictions.csv")
    file_exists = os.path.exists(user_csv)
    with open(user_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["Timestamp", "Username", "Image", "Disease", "Confidence(%)", "Damage(%)", "Cure"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), username, filename, label, f"{conf:.2f}", f"{damage:.2f}", cure])

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return auth_page("Login or sign up to begin.")

@app.post("/signup")
def route_signup(username: str = Form(...), password: str = Form(...)):
    err = signup_user(username.strip(), password.strip())
    if err:
        return HTMLResponse(auth_page(f"❌ {err}"), status_code=400)
    return RedirectResponse(url=f"/app?user={username}", status_code=302)

@app.post("/login")
def route_login(username: str = Form(...), password: str = Form(...)):
    if not verify_user(username.strip(), password.strip()):
        return HTMLResponse(auth_page("❌ Invalid username or password."), status_code=401)
    return RedirectResponse(url=f"/app?user={username}", status_code=302)

@app.get("/app", response_class=HTMLResponse)
def route_app(user: str):
    if not user_exists(user):
        return HTMLResponse(auth_page("❌ Session invalid. Please log in."), status_code=401)
    return app_page(user)

@app.post("/predict", response_class=HTMLResponse)
async def route_predict(file: UploadFile = File(...), username: str = Form(...)):
    if not user_exists(username):
        return HTMLResponse(auth_page("❌ Please log in."), status_code=401)

    # Read image
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    # Predict
    label, conf, damage, cure = predict_image(pil)

    # Save CSV history
    save_history(username, file.filename, label, conf, damage, cure)

    # Inline preview image
    b64 = base64.b64encode(raw).decode("utf-8")

    body = f"""
    <h1>✅ Prediction Result</h1>
    <div class="card">
      <div class="row">
        <div style="flex:1; min-width:260px">
          <img src="data:{file.content_type};base64,{b64}" style="max-width:100%; border-radius:12px; border:1px solid #c8e6c9" />
        </div>
        <div style="flex:1; min-width:260px">
          <p><b>Image:</b> {file.filename}</p>
          <p><b>Disease:</b> {label}</p>
          <p><b>Confidence:</b> {conf:.2f}%</p>
          <p><b>Estimated Damage:</b> {damage:.2f}%</p>
          <p><b>💊 Cure:</b> {cure}</p>
          <p><a class="link" href="/app?user={username}">🔙 Back</a></p>
          <p><a class="link" download href="/history?user={username}">⬇️ Download your CSV history</a></p>
        </div>
      </div>
    </div>
    """
    return HTMLResponse(page_html(body))

@app.get("/history")
def route_history(user: str):
    if not user_exists(user):
        return HTMLResponse(auth_page("❌ Please log in."), status_code=401)
    path = os.path.join(DATA_DIR, f"{user}_predictions.csv")
    if not os.path.exists(path):
        # return empty CSV if none yet
        header = "Timestamp,Username,Image,Disease,Confidence(%),Damage(%),Cure\n"
        return Response(content=header, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{user}_predictions.csv"'})
    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{user}_predictions.csv"'})

# ──────────────────────────────────────────────────────────────────────────────
# Uvicorn entry (Hugging Face Spaces will autodetect FastAPI app)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
