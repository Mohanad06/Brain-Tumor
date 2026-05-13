from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import base64
import uvicorn

app = FastAPI(title="Brain Tumor Classifier")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_INFO = {
    'glioma':      {'ar': 'ورم دبقي',        'risk': 'high',   'color': '#ef4444'},
    'meningioma':  {'ar': 'ورم سحائي',       'risk': 'medium', 'color': '#f97316'},
    'notumor':     {'ar': 'لا يوجد ورم',     'risk': 'none',   'color': '#22c55e'},
    'pituitary':   {'ar': 'ورم نخامي',       'risk': 'medium', 'color': '#f97316'},
}

model = load_model('brain_tumor_model.keras')

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NeuroScan AI — Brain Tumor Classifier</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:       #070b12;
  --bg2:      #0d1421;
  --bg3:      #111926;
  --border:   rgba(99,179,237,0.12);
  --border2:  rgba(99,179,237,0.25);
  --accent:   #63b3ed;
  --accent2:  #4299e1;
  --text:     #e2e8f0;
  --muted:    #718096;
  --mono:     'Space Mono', monospace;
  --sans:     'DM Sans', sans-serif;
}

body{
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Animated grid background */
body::before{
  content:'';
  position:fixed;inset:0;
  background-image:
    linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridmove 20s linear infinite;
  pointer-events:none;z-index:0;
}
@keyframes gridmove{to{background-position:40px 40px}}

/* Glowing orbs */
body::after{
  content:'';
  position:fixed;
  width:600px;height:600px;
  top:-200px;right:-200px;
  background: radial-gradient(circle, rgba(66,153,225,0.06) 0%, transparent 70%);
  pointer-events:none;z-index:0;
  animation: orb 8s ease-in-out infinite alternate;
}
@keyframes orb{to{transform:translate(-60px,60px)}}

.container{
  position:relative;z-index:1;
  max-width:900px;
  margin:0 auto;
  padding:3rem 2rem;
}

/* Header */
header{
  text-align:center;
  margin-bottom:3.5rem;
  animation: fadedown 0.7s ease both;
}
@keyframes fadedown{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:none}}

.badge{
  display:inline-flex;align-items:center;gap:6px;
  background: rgba(99,179,237,0.08);
  border: 1px solid var(--border2);
  border-radius:100px;
  padding:5px 14px;
  font-family:var(--mono);
  font-size:11px;
  color:var(--accent);
  letter-spacing:0.08em;
  margin-bottom:1.2rem;
}
.badge-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--accent);
  animation:pulse 2s infinite;
}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(0.8)}}

h1{
  font-family:var(--mono);
  font-size:clamp(1.8rem,4vw,3rem);
  font-weight:700;
  letter-spacing:-0.02em;
  background: linear-gradient(135deg, #e2e8f0 0%, #63b3ed 50%, #90cdf4 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  line-height:1.1;
  margin-bottom:0.75rem;
}

.subtitle{
  color:var(--muted);
  font-size:0.95rem;
  font-weight:300;
  letter-spacing:0.02em;
}

/* Stats bar */
.stats-bar{
  display:flex;gap:1px;
  background:var(--border);
  border:1px solid var(--border);
  border-radius:12px;
  overflow:hidden;
  margin-bottom:2rem;
  animation: fadein 0.8s 0.2s ease both;
}
@keyframes fadein{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

.stat{
  flex:1;
  padding:1rem;
  text-align:center;
  background:var(--bg2);
  transition:background 0.2s;
}
.stat:hover{background:var(--bg3)}
.stat-num{
  font-family:var(--mono);
  font-size:1.3rem;
  font-weight:700;
  color:var(--accent);
  display:block;
}
.stat-label{
  font-size:0.7rem;
  color:var(--muted);
  letter-spacing:0.06em;
  text-transform:uppercase;
}

/* Upload zone */
.upload-card{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius:16px;
  padding:2rem;
  margin-bottom:2rem;
  animation: fadein 0.8s 0.3s ease both;
  transition: border-color 0.3s;
}
.upload-card:hover{border-color:var(--border2)}

.drop-zone{
  border: 1.5px dashed var(--border2);
  border-radius:12px;
  padding:3rem 2rem;
  text-align:center;
  cursor:pointer;
  transition: all 0.25s ease;
  position:relative;
  overflow:hidden;
}
.drop-zone::before{
  content:'';
  position:absolute;inset:0;
  background: radial-gradient(ellipse at center, rgba(99,179,237,0.04) 0%, transparent 70%);
  opacity:0;
  transition:opacity 0.3s;
}
.drop-zone:hover::before,.drop-zone.dragging::before{opacity:1}
.drop-zone:hover,.drop-zone.dragging{
  border-color:var(--accent);
  background:rgba(99,179,237,0.03);
  transform:scale(1.005);
}

.upload-icon{
  width:56px;height:56px;
  margin:0 auto 1rem;
  background:rgba(99,179,237,0.1);
  border-radius:14px;
  display:flex;align-items:center;justify-content:center;
  font-size:1.5rem;
  transition: transform 0.3s;
}
.drop-zone:hover .upload-icon{transform:translateY(-4px)}

.drop-text{font-size:0.95rem;color:var(--text);margin-bottom:0.3rem}
.drop-sub{font-size:0.78rem;color:var(--muted)}

#fileInput{display:none}

/* Preview */
#preview-section{display:none;margin-top:1.5rem}
.preview-wrap{
  display:flex;gap:1.5rem;align-items:flex-start;
  flex-wrap:wrap;
}
#preview-img{
  width:160px;height:160px;
  object-fit:cover;
  border-radius:10px;
  border:1px solid var(--border2);
}
.preview-info{flex:1;min-width:160px}
.file-name{
  font-family:var(--mono);
  font-size:0.78rem;
  color:var(--accent);
  margin-bottom:0.5rem;
  word-break:break-all;
}
.file-size{font-size:0.75rem;color:var(--muted)}

/* Analyze button */
.btn{
  display:inline-flex;align-items:center;gap:8px;
  background: linear-gradient(135deg, var(--accent2), var(--accent));
  color: #070b12;
  border:none;
  border-radius:10px;
  padding:0.75rem 1.75rem;
  font-family:var(--mono);
  font-size:0.82rem;
  font-weight:700;
  cursor:pointer;
  letter-spacing:0.04em;
  transition: all 0.2s ease;
  margin-top:1rem;
}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(99,179,237,0.25)}
.btn:active{transform:translateY(0)}
.btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}

/* Spinner */
.spinner{
  width:16px;height:16px;
  border:2px solid rgba(7,11,18,0.3);
  border-top-color:#070b12;
  border-radius:50%;
  animation:spin 0.7s linear infinite;
  display:none;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* Results */
#result-card{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:16px;
  overflow:hidden;
  display:none;
  animation: fadein 0.5s ease both;
}

.result-header{
  padding:1rem 1.5rem;
  background:var(--bg3);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;
}
.result-header-dot{
  width:8px;height:8px;border-radius:50%;
  background:var(--accent);
}
.result-header-label{
  font-family:var(--mono);
  font-size:0.72rem;
  color:var(--muted);
  letter-spacing:0.1em;
  text-transform:uppercase;
}

.result-body{padding:1.5rem}

.diagnosis-row{
  display:flex;align-items:center;gap:1rem;
  margin-bottom:1.5rem;
  flex-wrap:wrap;
}
.diagnosis-name{
  font-family:var(--mono);
  font-size:1.6rem;
  font-weight:700;
  letter-spacing:-0.01em;
}
.risk-badge{
  padding:4px 12px;
  border-radius:100px;
  font-size:0.7rem;
  font-family:var(--mono);
  letter-spacing:0.08em;
  text-transform:uppercase;
  font-weight:700;
}
.risk-none{background:rgba(34,197,94,0.12);color:#22c55e;border:1px solid rgba(34,197,94,0.25)}
.risk-medium{background:rgba(249,115,22,0.12);color:#f97316;border:1px solid rgba(249,115,22,0.25)}
.risk-high{background:rgba(239,68,68,0.12);color:#ef4444;border:1px solid rgba(239,68,68,0.25)}

/* Confidence bars */
.bars-title{
  font-size:0.72rem;
  color:var(--muted);
  letter-spacing:0.08em;
  text-transform:uppercase;
  margin-bottom:0.75rem;
  font-family:var(--mono);
}
.bar-row{
  display:flex;align-items:center;gap:10px;
  margin-bottom:0.6rem;
}
.bar-label{
  font-family:var(--mono);
  font-size:0.72rem;
  color:var(--muted);
  width:90px;flex-shrink:0;
}
.bar-track{
  flex:1;height:6px;
  background:rgba(255,255,255,0.06);
  border-radius:100px;overflow:hidden;
}
.bar-fill{
  height:100%;border-radius:100px;
  transition:width 0.8s cubic-bezier(0.16,1,0.3,1);
  width:0%;
}
.bar-pct{
  font-family:var(--mono);
  font-size:0.72rem;
  color:var(--muted);
  width:38px;text-align:right;flex-shrink:0;
}

/* Scan lines effect on result */
.scan-lines{
  position:relative;overflow:hidden;
}
.scan-lines::after{
  content:'';
  position:absolute;inset:0;
  background:repeating-linear-gradient(
    0deg,
    transparent,transparent 2px,
    rgba(99,179,237,0.015) 2px,rgba(99,179,237,0.015) 4px
  );
  pointer-events:none;
}

/* Arabic result */
.ar-label{
  font-size:0.85rem;
  color:var(--muted);
  margin-top:0.3rem;
}

/* Footer */
footer{
  text-align:center;
  margin-top:3rem;
  padding-top:1.5rem;
  border-top:1px solid var(--border);
  font-size:0.72rem;
  color:var(--muted);
  font-family:var(--mono);
  animation: fadein 1s 0.5s ease both;
}
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="badge">
      <span class="badge-dot"></span>
      MODEL ACTIVE — MobileNetV2 Transfer Learning
    </div>
    <h1>NeuroScan<br/>AI Classifier</h1>
    <p class="subtitle">Brain Tumor MRI Analysis · 4-Class Detection · Deep Learning</p>
  </header>

  <div class="stats-bar">
    <div class="stat"><span class="stat-num">4</span><span class="stat-label">Tumor Classes</span></div>
    <div class="stat"><span class="stat-num">224²</span><span class="stat-label">Input Resolution</span></div>
    <div class="stat"><span class="stat-num">MNv2</span><span class="stat-label">Backbone</span></div>
    <div class="stat"><span class="stat-num">Softmax</span><span class="stat-label">Classifier</span></div>
  </div>

  <div class="upload-card">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="upload-icon">🧠</div>
      <p class="drop-text">Drop MRI scan here or click to upload</p>
      <p class="drop-sub">Supports JPG, PNG, JPEG · Any resolution</p>
      <input type="file" id="fileInput" accept="image/*"/>
    </div>

    <div id="preview-section">
      <div class="preview-wrap">
        <img id="preview-img" src="" alt="MRI Preview"/>
        <div class="preview-info">
          <p class="bars-title">Selected File</p>
          <p class="file-name" id="file-name-display">—</p>
          <p class="file-size" id="file-size-display">—</p>
          <button class="btn" id="analyzeBtn" onclick="analyze()">
            <div class="spinner" id="spinner"></div>
            <span id="btn-text">⬡ RUN ANALYSIS</span>
          </button>
        </div>
      </div>
    </div>
  </div>

  <div id="result-card" class="scan-lines">
    <div class="result-header">
      <div class="result-header-dot"></div>
      <span class="result-header-label">Diagnostic Output</span>
    </div>
    <div class="result-body">
      <div class="diagnosis-row">
        <div>
          <div class="diagnosis-name" id="res-class">—</div>
          <div class="ar-label" id="res-ar">—</div>
        </div>
        <span class="risk-badge" id="res-risk">—</span>
      </div>
      <p class="bars-title">Confidence Distribution</p>
      <div id="bars-container"></div>
    </div>
  </div>

  <footer>
    NEUROSCAN AI · BRAIN TUMOR CLASSIFICATION · COLLEGE PROJECT<br/>
    MobileNetV2 Transfer Learning · Trained on Kaggle MRI Dataset
  </footer>

</div>

<script>
const classInfo = {
  glioma:     {ar:'ورم دبقي',    risk:'high',   color:'#ef4444'},
  meningioma: {ar:'ورم سحائي',  risk:'medium', color:'#f97316'},
  notumor:    {ar:'لا يوجد ورم',risk:'none',   color:'#22c55e'},
  pituitary:  {ar:'ورم نخامي',  risk:'medium', color:'#f97316'},
};

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
let selectedFile = null;

dropZone.addEventListener('dragover', e => {e.preventDefault(); dropZone.classList.add('dragging')});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragging');
  const f = e.dataTransfer.files[0];
  if(f && f.type.startsWith('image/')) setFile(f);
});

fileInput.addEventListener('change', e => { if(e.target.files[0]) setFile(e.target.files[0]); });

function setFile(f){
  selectedFile = f;
  document.getElementById('preview-section').style.display = 'block';
  document.getElementById('file-name-display').textContent = f.name;
  document.getElementById('file-size-display').textContent = (f.size/1024).toFixed(1)+' KB';
  const reader = new FileReader();
  reader.onload = e => document.getElementById('preview-img').src = e.target.result;
  reader.readAsDataURL(f);
  document.getElementById('result-card').style.display = 'none';
}

async function analyze(){
  if(!selectedFile) return;
  const btn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btn-text');
  btn.disabled = true;
  spinner.style.display = 'block';
  btnText.textContent = 'ANALYZING...';

  const form = new FormData();
  form.append('image', selectedFile);

  try{
    const res = await fetch('/predict', {method:'POST', body:form});
    const data = await res.json();
    showResult(data);
  } catch(e){
    alert('Error connecting to model. Make sure the server is running.');
  }

  btn.disabled = false;
  spinner.style.display = 'none';
  btnText.textContent = '⬡ RUN ANALYSIS';
}

function showResult(data){
  const card = document.getElementById('result-card');
  card.style.display = 'block';

  const info = classInfo[data.class] || {ar:'—', risk:'none', color:'#63b3ed'};
  document.getElementById('res-class').textContent = data.class.toUpperCase();
  document.getElementById('res-class').style.color = info.color;
  document.getElementById('res-ar').textContent = info.ar;

  const riskEl = document.getElementById('res-risk');
  riskEl.textContent = info.risk === 'none' ? 'CLEAR' : info.risk.toUpperCase()+' RISK';
  riskEl.className = 'risk-badge risk-'+info.risk;

  const container = document.getElementById('bars-container');
  container.innerHTML = '';
  const probs = data.all_probabilities || {};
  Object.entries(probs).sort((a,b)=>b[1]-a[1]).forEach(([cls, prob]) => {
    const pct = (prob*100).toFixed(1);
    const color = classInfo[cls]?.color || '#63b3ed';
    container.innerHTML += `
      <div class="bar-row">
        <span class="bar-label">${cls}</span>
        <div class="bar-track">
          <div class="bar-fill" id="bar-${cls}" style="background:${color}"></div>
        </div>
        <span class="bar-pct">${pct}%</span>
      </div>`;
  });
  setTimeout(()=>{
    Object.entries(probs).forEach(([cls,prob])=>{
      const el = document.getElementById('bar-'+cls);
      if(el) el.style.width = (prob*100)+'%';
    });
  }, 50);

  card.scrollIntoView({behavior:'smooth', block:'nearest'});
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr)[0]
    all_probs = {CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(len(CLASS_NAMES))}
    predicted_class = CLASS_NAMES[int(np.argmax(preds))]
    confidence = round(float(np.max(preds)), 4)

    return JSONResponse({
        "class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "class_info": CLASS_INFO[predicted_class]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)