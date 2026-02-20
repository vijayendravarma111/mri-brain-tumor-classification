# Brain Tumor Classifier — same 2-column UI & API shape as Bone app

import os, io
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models
import uvicorn

# ----------------- Paths & device -----------------
ROOT = Path(__file__).resolve().parent
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoid OpenMP dll clash on Windows

APP_TITLE = "Brain Tumor Detection"  # change to "Brain Disease Detection" if you prefer
DEFAULT_CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

# ----------------- Load brain model (from your training) -----------------
def _load_ckpt():
    """
    Supports either:
      - models/best_model.pt  (from your train.py; contains state_dict + class_names + img_size)
      - your_brain_model.pth  (full model object)
    """
    p1 = ROOT / "models" / "best_model.pt"
    p2 = ROOT / "your_brain_model.pth"
    if p1.exists():
        return "best_model", torch.load(str(p1), map_location="cpu")
    if p2.exists():
        return "pth", torch.load(str(p2), map_location="cpu")
    raise FileNotFoundError("Place 'models/best_model.pt' (preferred) or 'your_brain_model.pth' next to this file.")

_kind, _raw = _load_ckpt()

if _kind == "best_model":
    class_names = list(_raw.get("class_names", DEFAULT_CLASSES))
    img_size = int(_raw.get("img_size", 224))
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(_raw["state_dict"])
else:
    model = _raw
    class_names = DEFAULT_CLASSES
    img_size = 224

model.to(DEVICE).eval()

PREP = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _tta_tensor(pil: Image.Image) -> torch.Tensor:
    imgs = [
        pil,
        ImageOps.mirror(pil),
        pil.rotate(+8, resample=Image.BILINEAR),
        pil.rotate(-8, resample=Image.BILINEAR),
    ]
    tensor_list = [PREP(im.convert("RGB")) for im in imgs]
    return torch.stack(tensor_list, dim=0).to(DEVICE)

@torch.no_grad()
def _predict_pil(pil: Image.Image):
    xs = _tta_tensor(pil)
    with torch.amp.autocast(device_type="cuda", enabled=USE_CUDA):
        logits = model(xs)
        prob = torch.softmax(logits, 1).mean(0)
    conf, idx = torch.max(prob, dim=0)
    top_label = class_names[idx.item()]
    top_conf = float(conf)
    return top_label, top_conf

def _summary(label: str, conf: float) -> str:
    pct = f"{conf*100:.0f}%"
    l = label.lower()
    if l == "glioma":
        msg = [
            f"Glioma ",
            "Findings suggest an intra-axial glial tumor.",
            "Recommendations:",
            "• Contrast-enhanced MRI brain (tumor protocol).",
            "• Neurosurgery / neuro-oncology referral.",
            "• Correlate with symptoms; watch for mass-effect signs.",
        ]
    elif l == "meningioma":
        msg = [
            f"Meningioma ",
            "Extra-axial dural-based lesion is suspected.",
            "Recommendations:",
            "• Contrast-enhanced MRI for characterization.",
            "• Neurosurgical consultation if symptomatic or large.",
            "• Imaging follow-up to assess growth.",
        ]
    elif l == "pituitary":
        msg = [
            f"Pituitary lesion ",
            "Sellar/suprasellar mass consistent with pituitary adenoma is likely.",
            "Recommendations:",
            "• Dedicated pituitary MRI with contrast.",
            "• Endocrinology hormonal profile.",
            "• Ophthalmologic visual-field assessment if indicated.",
        ]
    else:
        msg = [
            f"No tumor detected ",
            "No obvious neoplastic features on this slice.",
            "Recommendations:",
            "• Correlate clinically across the full MRI series.",
            "• Repeat or advanced sequences if symptoms persist.",
        ]
    return "\n".join(msg)

def score_image_bytes(raw: bytes) -> dict:
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    label, conf = _predict_pil(im)
    report = _summary(label, conf)
    return {
        "task": APP_TITLE,
        "label": label,
        "confidence": conf,
        "status": label.capitalize(),
        "report_text": report,   # same name used by Bone UI
    }

# ----------------- FastAPI -----------------
app = FastAPI(title=f"MedVision • {APP_TITLE}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------------- Embedded HTML (exact same 2-column layout as Bone) -----------------
COMMON_HEAD = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css">
<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
<title>MedVision</title>
<style>
 body{background:#0b1020;color:#e7ecf8}
 .card{backdrop-filter:blur(8px);background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1)}
 .btn{background:#312e81;border:1px solid rgba(255,255,255,.15)}
 .btn:hover{background:#4338ca}
 .drop{border:2px dashed rgba(255,255,255,.25)}
 pre{white-space:pre-wrap}
</style></head><body class="min-h-screen">
<div class="max-w-6xl mx-auto px-4 py-10">
"""

def MODEL_PAGE(title):
    return COMMON_HEAD + f"""
<h2 class="text-3xl font-extrabold mb-6">{title}</h2>

<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
  <!-- LEFT: Upload -->
  <div class="card rounded-2xl p-6">
    <div id="drop" class="drop rounded-xl p-8 text-center cursor-pointer">
      <input id="file" type="file" accept="image/*" class="hidden">
      <p id="hint" class="text-lg">Drag & drop a brain MRI image, or <span class="underline">browse</span></p>
      <img id="preview" class="mx-auto rounded-xl hidden mt-4 max-h-80"/>
    </div>
    <div class="mt-6 flex items-center gap-4">
      <button id="analyzeBtn" class="btn px-5 py-2 rounded-xl">Analyze</button>
      <span id="status" class="opacity-80"></span>
    </div>
  </div>

  <!-- RIGHT: Report -->
  <div class="card rounded-2xl p-6 hidden" id="reportCard">
    <h3 class="text-xl font-semibold mb-2">Description of Report</h3>
    <div id="report" class="space-y-1"></div>
  </div>
</div>

</div>
<script>
let selectedFile=null;
const drop=document.getElementById('drop'),file=document.getElementById('file'),
preview=document.getElementById('preview'),hint=document.getElementById('hint'),
btn=document.getElementById('analyzeBtn'),status=document.getElementById('status'),
card=document.getElementById('reportCard'),report=document.getElementById('report');

drop.onclick=()=>file.click();
drop.ondragover=e=>{{e.preventDefault();drop.classList.add('border-indigo-400')}}; 
drop.ondragleave=()=>drop.classList.remove('border-indigo-400');
drop.ondrop=e=>{{e.preventDefault();drop.classList.remove('border-indigo-400'); if(e.dataTransfer.files.length)setFile(e.dataTransfer.files[0]);}};
file.onchange=e=>{{if(e.target.files.length)setFile(e.target.files[0]);}};

function setFile(f){{
  selectedFile=f;
  const url=URL.createObjectURL(f);
  preview.src=url; preview.classList.remove('hidden'); hint.classList.add('hidden');
}}

btn.onclick=async()=>{{
  if(!selectedFile){{status.textContent='Select an image first.';return;}}
  status.textContent='Analyzing…'; btn.disabled=true; card.classList.add('hidden'); report.innerHTML='';
  const fd=new FormData(); fd.append('model','brain'); fd.append('file',selectedFile);
  try{{
      const r=await axios.post('/analyze',fd); const d=r.data;
      report.innerHTML=`<div><span class="opacity-70">Task:</span> <b>${{d.task}}</b></div>
      <div><span class="opacity-70">Prediction:</span> <b>${{d.label}}</b></div>
      
      <div class="mt-3"><span class="opacity-70">Report:</span>
           <pre class="mt-1 bg-black bg-opacity-20 rounded-lg p-3 border border-white border-opacity-10">${{d.report_text}}</pre></div>
      <div class="mt-3 opacity-70 text-xs">Timestamp: ${{new Date().toLocaleString()}}</div>`;
      card.classList.remove('hidden'); status.textContent='Done.';
  }}catch(e){{status.textContent='Error.'; console.error(e);}}
  finally{{btn.disabled=false;}}
}};
</script>
</body></html>
"""

# ----------------- Routes -----------------
app = FastAPI(title=f"MedVision • {APP_TITLE}")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def brain_page():
    return MODEL_PAGE(APP_TITLE)

@app.post("/analyze")
async def analyze(model: str = Form(...), file: UploadFile = File(...)):
    if model != "brain":
        return JSONResponse({"error": "Only 'brain' is supported."}, status_code=400)
    raw = await file.read()
    out = score_image_bytes(raw)
    out["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}

if __name__ == "__main__":
    uvicorn.run("brain_app:app", host="127.0.0.1", port=8000, reload=True)
