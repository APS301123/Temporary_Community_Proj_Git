from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src import database as db
from src.detector import LostFoundDetector, count_by

load_dotenv()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.2"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

detector = LostFoundDetector(confidence_threshold=MODEL_CONFIDENCE)


@asynccontextmanager
async def lifespan(application: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Lost & Found", lifespan=lifespan)


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------

BROWSE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Lost &amp; Found</title>
  <style>
    :root{--primary:#2563eb;--primary-dark:#1d4ed8;--bg:#f1f5f9;--card:#fff;
          --text:#0f172a;--muted:#64748b;--border:#e2e8f0;--success:#16a34a;}
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:var(--bg);color:var(--text);}
    header{background:#1e3a5f;color:#fff;padding:1.25rem 2rem;
           display:flex;justify-content:space-between;align-items:center;}
    header .brand h1{font-size:1.4rem;font-weight:800;}
    header .brand p{font-size:.85rem;opacity:.75;margin-top:.2rem;}
    header a{color:rgba(255,255,255,.7);text-decoration:none;font-size:.85rem;}
    header a:hover{color:#fff;}
    main{max-width:1200px;margin:0 auto;padding:1.5rem 2rem;}
    .controls{background:#fff;border-radius:12px;padding:1.25rem 1.5rem;
              margin-bottom:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.08);}
    .search-input{width:100%;padding:.7rem 1rem;border:1.5px solid var(--border);
                  border-radius:8px;font-size:1rem;outline:none;margin-bottom:1rem;}
    .search-input:focus{border-color:var(--primary);}
    .filter-row{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin-bottom:.6rem;}
    .filter-label{font-size:.75rem;font-weight:700;color:var(--muted);
                  text-transform:uppercase;letter-spacing:.05em;min-width:4.5rem;}
    .chip{padding:.35rem .85rem;border-radius:999px;border:1.5px solid var(--border);
          background:#fff;font-size:.82rem;cursor:pointer;transition:all .15s;white-space:nowrap;}
    .chip:hover{border-color:var(--primary);color:var(--primary);}
    .chip.active{background:var(--primary);color:#fff;border-color:var(--primary);}
    .color-filters{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;}
    .cdot{width:26px;height:26px;border-radius:50%;border:2.5px solid #fff;
          box-shadow:0 0 0 1.5px var(--border);cursor:pointer;transition:transform .15s;}
    .cdot:hover{transform:scale(1.15);}
    .cdot.active{box-shadow:0 0 0 2.5px var(--primary);}
    .search-hint{display:none;margin-top:.5rem;padding:.3rem .7rem;background:#eff6ff;
                 color:#2563eb;border-radius:6px;font-size:.8rem;}
    .stats-bar{text-align:center;color:var(--muted);font-size:.88rem;margin-bottom:.75rem;}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1.1rem;}
    .card{background:#fff;border-radius:12px;overflow:hidden;
          box-shadow:0 1px 3px rgba(0,0,0,.08);transition:transform .15s,box-shadow .15s;cursor:pointer;}
    .card:hover{transform:translateY(-2px);box-shadow:0 4px 14px rgba(0,0,0,.13);}
    .img-wrap{position:relative;}
    .card-img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#e2e8f0;}
    .card-placeholder{width:100%;aspect-ratio:1;background:#e2e8f0;
                      display:none;align-items:center;justify-content:center;font-size:2.5rem;}
    .cat-badge{position:absolute;top:8px;left:8px;padding:.2rem .55rem;border-radius:6px;
               font-size:.7rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:#fff;}
    .card-body{padding:.85rem;}
    .card-title{font-weight:700;font-size:.95rem;margin-bottom:.3rem;}
    .card-meta{font-size:.8rem;color:var(--muted);display:flex;flex-direction:column;
               gap:.2rem;margin-bottom:.65rem;}
    .card-meta span{display:flex;align-items:center;gap:.3rem;}
    .cbadge{display:inline-block;width:9px;height:9px;border-radius:50%;border:1px solid rgba(0,0,0,.12);}
    .claim-btn{width:100%;padding:.55rem;background:var(--primary);color:#fff;border:none;
               border-radius:8px;font-size:.88rem;font-weight:600;cursor:pointer;}
    .claim-btn:hover{background:var(--primary-dark);}
    .no-items{text-align:center;padding:3.5rem 2rem;color:var(--muted);grid-column:1/-1;}
    .no-items h3{font-size:1.15rem;margin-bottom:.4rem;}
    .loading{text-align:center;padding:3rem;color:var(--muted);grid-column:1/-1;}
    .modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);
                   z-index:200;align-items:center;justify-content:center;padding:1rem;}
    .modal-overlay.open{display:flex;}
    .modal{background:#fff;border-radius:16px;max-width:560px;width:100%;overflow:hidden;
           max-height:90vh;overflow-y:auto;}
    .modal-img{width:100%;max-height:360px;object-fit:contain;background:#1e293b;display:block;}
    .modal-body{padding:1.4rem;}
    .modal-title{font-size:1.2rem;font-weight:800;margin-bottom:.4rem;}
    .modal-meta{color:var(--muted);font-size:.88rem;margin-bottom:.9rem;line-height:1.6;}
    .modal-info{background:#eff6ff;border-radius:8px;padding:.9rem 1rem;
                margin-bottom:1rem;font-size:.88rem;color:#1d4ed8;line-height:1.5;}
    .btn-close{width:100%;padding:.65rem;background:var(--bg);color:var(--text);
               border:1.5px solid var(--border);border-radius:8px;font-size:.9rem;cursor:pointer;}
    footer{text-align:center;padding:2rem;color:var(--muted);font-size:.82rem;}
  </style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Lost &amp; Found</h1>
    <p>Browse items found at school &mdash; click any item to see how to claim it.</p>
  </div>
  <a href="/staff">Staff Portal &rarr;</a>
</header>

<main>
  <div class="controls">
    <input class="search-input" id="searchInput" type="text"
           placeholder="Search anything &mdash; &ldquo;blue Nike bag&rdquo;, &ldquo;water bottle&rdquo;, &ldquo;AirPods&rdquo;&hellip;" />
    <div class="search-hint" id="searchHint"></div>

    <div class="filter-row">
      <span class="filter-label">Category</span>
      <div id="catFilters" style="display:flex;gap:.6rem;flex-wrap:wrap;"></div>
    </div>

    <div class="filter-row">
      <span class="filter-label">Color</span>
      <div class="color-filters" id="colorFilters"></div>
    </div>
  </div>

  <div class="stats-bar" id="statsBar"></div>
  <div class="grid" id="itemsGrid"><div class="loading">Loading items&hellip;</div></div>
</main>

<div class="modal-overlay" id="modalOverlay">
  <div class="modal">
    <img class="modal-img" id="modalImg" src="" alt="Found item" />
    <div class="modal-body">
      <div class="modal-title" id="modalTitle"></div>
      <div class="modal-meta" id="modalMeta"></div>
      <div class="modal-info">
        To claim this item, please bring your student ID to the extended day office
        and describe the item. Staff will match it to this record.
      </div>
      <button class="btn-close" onclick="closeModal()">Close</button>
    </div>
  </div>
</div>

<footer>Lost &amp; Found System &mdash; Extended Day Office</footer>

<script>
const CATEGORIES = ["All","Apparel","Drinkware","Food Containers","Bags",
  "School Supplies","Electronics","Accessories","Personal Essentials","Footwear","Other"];

const CAT_COLORS = {
  Apparel:"#7c3aed",Drinkware:"#2563eb","Food Containers":"#ca8a04",
  Bags:"#16a34a","School Supplies":"#ea580c",Electronics:"#dc2626",
  Accessories:"#0891b2","Personal Essentials":"#db2777",Footwear:"#78350f",Other:"#64748b"
};

const COLOR_HEX = {
  black:"#1a1a1a",white:"#e8e8e8",gray:"#808080",red:"#dc2626",orange:"#ea580c",
  yellow:"#ca8a04",green:"#16a34a",blue:"#2563eb",purple:"#7c3aed",
  pink:"#db2777",brown:"#78350f",unknown:"#94a3b8"
};

const COLORS_LIST = [
  {name:"All",hex:null},
  {name:"black",hex:"#1a1a1a"},{name:"white",hex:"#e8e8e8"},{name:"gray",hex:"#808080"},
  {name:"red",hex:"#dc2626"},{name:"orange",hex:"#ea580c"},{name:"yellow",hex:"#ca8a04"},
  {name:"green",hex:"#16a34a"},{name:"blue",hex:"#2563eb"},{name:"purple",hex:"#7c3aed"},
  {name:"pink",hex:"#db2777"},{name:"brown",hex:"#78350f"}
];

let activeCategory = "All", activeColor = "All", searchTerm = "";
let debounceTimer;

// Build category chips
const catDiv = document.getElementById("catFilters");
CATEGORIES.forEach(cat => {
  const btn = document.createElement("button");
  btn.className = "chip" + (cat === "All" ? " active" : "");
  btn.textContent = cat;
  btn.onclick = () => {
    activeCategory = cat;
    catDiv.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    loadItems();
  };
  catDiv.appendChild(btn);
});

// Build color dots
const colorDiv = document.getElementById("colorFilters");
COLORS_LIST.forEach(c => {
  let el;
  if (c.name === "All") {
    el = document.createElement("button");
    el.className = "chip active";
    el.textContent = "All";
    el.style.padding = ".3rem .75rem";
  } else {
    el = document.createElement("div");
    el.className = "cdot";
    el.style.background = c.hex;
    el.title = c.name;
  }
  el.onclick = () => {
    activeColor = c.name;
    colorDiv.querySelectorAll(".cdot, .chip").forEach(d => d.classList.remove("active"));
    el.classList.add("active");
    loadItems();
  };
  colorDiv.appendChild(el);
});

// Search
document.getElementById("searchInput").addEventListener("input", e => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => { searchTerm = e.target.value; loadItems(); }, 300);
});

async function loadItems() {
  const hint = document.getElementById("searchHint");
  if (searchTerm.trim()) {
    const params = new URLSearchParams({ q: searchTerm });
    if (activeCategory !== "All") params.set("category", activeCategory);
    if (activeColor !== "All") params.set("color", activeColor);
    try {
      const res = await fetch("/search/smart?" + params);
      const data = await res.json();
      if (data.llm_used && data.interpreted) {
        hint.textContent = "✨ Interpreted as: “" + data.interpreted.label + "”";
        hint.style.display = "block";
      } else {
        hint.style.display = "none";
      }
      renderItems(data.items);
    } catch {
      hint.style.display = "none";
      document.getElementById("itemsGrid").innerHTML =
        '<div class="no-items"><h3>Could not load items.</h3></div>';
    }
  } else {
    hint.style.display = "none";
    const params = new URLSearchParams({ status: "unclaimed" });
    if (activeCategory !== "All") params.set("category", activeCategory);
    if (activeColor !== "All") params.set("color", activeColor);
    try {
      const res = await fetch("/items?" + params);
      renderItems(await res.json());
    } catch {
      document.getElementById("itemsGrid").innerHTML =
        '<div class="no-items"><h3>Could not load items.</h3></div>';
    }
  }
}

function renderItems(items) {
  const grid = document.getElementById("itemsGrid");
  document.getElementById("statsBar").textContent =
    items.length ? `${items.length} item${items.length !== 1 ? "s" : ""} currently in lost & found` : "";

  if (!items.length) {
    grid.innerHTML = `<div class="no-items">
      <h3>No items found</h3>
      <p>Try adjusting your search or filters, or check back later.</p>
    </div>`;
    return;
  }

  grid.innerHTML = items.map(item => {
    const catColor = CAT_COLORS[item.primary_category] || "#64748b";
    const itemColors = (item.color || "unknown").split(",").map(c => c.trim());
    const colorBadges = itemColors.map(c =>
      `<span><span class="cbadge" style="background:${COLOR_HEX[c]||'#94a3b8'}"></span>${c}</span>`
    ).join(" ");
    return `
    <div class="card" onclick="openModal(${item.id})">
      <div class="img-wrap">
        <img class="card-img" src="/image/${item.image_id}" alt="${item.primary_subcat}" loading="lazy"
             onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
        <div class="card-placeholder">&#128230;</div>
        <span class="cat-badge" style="background:${catColor}">${item.primary_category}</span>
      </div>
      <div class="card-body">
        <div class="card-title">${item.primary_subcat}</div>
        <div class="card-meta">
          ${colorBadges}
          ${item.location ? `<span>&#128205; ${item.location}</span>` : ""}
          <span>&#128197; ${item.date_found}</span>
        </div>
        <button class="claim-btn" onclick="event.stopPropagation();openModal(${item.id})">
          This is mine &rarr;
        </button>
      </div>
    </div>`;
  }).join("");
}

async function openModal(itemId) {
  const item = await (await fetch("/items/" + itemId)).json();
  document.getElementById("modalImg").src = "/image/" + item.image_id;
  document.getElementById("modalTitle").textContent =
    item.primary_subcat + " (" + item.primary_category + ")";
  const modalColors = (item.color || "unknown").split(",").map(c => c.trim());
  const modalColorStr = modalColors.map(c =>
    `<span class="cbadge" style="background:${COLOR_HEX[c]||'#94a3b8'};vertical-align:middle"></span> ${c}`
  ).join(", ");
  document.getElementById("modalMeta").innerHTML =
    `<strong>Color:</strong> ${modalColorStr}` +
    (item.location ? ` &nbsp;&bull;&nbsp; <strong>Found at:</strong> ${item.location}` : "") +
    ` &nbsp;&bull;&nbsp; <strong>Date:</strong> ${item.date_found}` +
    (item.description ? `<br><strong>Description:</strong> ${item.description}` : "") +
    (item.staff_notes ? `<br><strong>Notes:</strong> ${item.staff_notes}` : "");
  document.getElementById("modalOverlay").classList.add("open");
}

function closeModal() {
  document.getElementById("modalOverlay").classList.remove("open");
}
document.getElementById("modalOverlay").addEventListener("click", e => {
  if (e.target === e.currentTarget) closeModal();
});

loadItems();
</script>
</body>
</html>"""

STAFF_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Staff Portal &mdash; Lost &amp; Found</title>
  <style>
    :root{--primary:#2563eb;--primary-dark:#1d4ed8;--bg:#f1f5f9;--text:#0f172a;
          --muted:#64748b;--border:#e2e8f0;--success:#16a34a;--danger:#dc2626;}
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:var(--bg);color:var(--text);}
    header{background:#1e3a5f;color:#fff;padding:1rem 2rem;
           display:flex;justify-content:space-between;align-items:center;}
    header h1{font-size:1.25rem;font-weight:800;}
    header a{color:rgba(255,255,255,.7);text-decoration:none;font-size:.85rem;}
    header a:hover{color:#fff;}
    main{max-width:1100px;margin:0 auto;padding:1.75rem 2rem;}
    .section-title{font-size:1rem;font-weight:700;margin-bottom:.9rem;}
    .stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;}
    .stat-card{background:#fff;border-radius:12px;padding:1.1rem 1.4rem;
               box-shadow:0 1px 3px rgba(0,0,0,.08);}
    .stat-num{font-size:2rem;font-weight:900;}
    .stat-label{font-size:.82rem;color:var(--muted);margin-top:.15rem;}
    .upload-card{background:#fff;border-radius:12px;padding:1.5rem;
                 margin-bottom:2rem;box-shadow:0 1px 3px rgba(0,0,0,.08);}
    .drop-zone{border:2px dashed var(--border);border-radius:10px;padding:2.25rem;
               text-align:center;cursor:pointer;color:var(--muted);
               transition:all .2s;margin-bottom:1rem;user-select:none;}
    .drop-zone:hover,.drop-zone.over{border-color:var(--primary);
                                      color:var(--primary);background:#eff6ff;}
    .drop-zone .icon{font-size:2.25rem;margin-bottom:.4rem;}
    .drop-zone p{font-size:.88rem;}
    .drop-zone .fname{font-weight:700;color:var(--text);margin-top:.4rem;font-size:.9rem;}
    .form-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;}
    .form-group{display:flex;flex-direction:column;gap:.35rem;}
    label{font-size:.82rem;font-weight:700;color:var(--muted);}
    .form-input{padding:.6rem .85rem;border:1.5px solid var(--border);border-radius:8px;
                font-size:.93rem;outline:none;width:100%;}
    .form-input:focus{border-color:var(--primary);}
    select.form-input{cursor:pointer;}
    .btn{padding:.65rem 1.4rem;border:none;border-radius:8px;font-size:.92rem;
         font-weight:700;cursor:pointer;transition:opacity .15s;}
    .btn:disabled{opacity:.45;cursor:not-allowed;}
    .btn-primary{background:var(--primary);color:#fff;}
    .btn-primary:hover:not(:disabled){background:var(--primary-dark);}
    .btn-success{background:var(--success);color:#fff;}
    .btn-success:hover:not(:disabled){background:#15803d;}
    .detect-panel{background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;
                  padding:1.2rem;margin-top:1rem;display:none;}
    .detect-panel h3{color:#15803d;margin-bottom:.85rem;font-size:.95rem;}
    .detect-item{background:#fff;border-radius:8px;padding:.55rem .85rem;
                 margin-bottom:.45rem;font-size:.88rem;display:flex;align-items:center;gap:.4rem;}
    .detect-item .conf{font-size:.76rem;color:var(--muted);margin-left:auto;}
    .edit-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.75rem;margin:.9rem 0;}
    .table-card{background:#fff;border-radius:12px;
                box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden;}
    .table-header{padding:.9rem 1.4rem;border-bottom:1px solid var(--border);
                  display:flex;justify-content:space-between;align-items:center;}
    .filter-btns{display:flex;gap:.5rem;}
    .fbtn{padding:.3rem .75rem;border-radius:999px;border:1.5px solid var(--border);
          background:#fff;font-size:.8rem;cursor:pointer;}
    .fbtn.active{background:var(--primary);color:#fff;border-color:var(--primary);}
    table{width:100%;border-collapse:collapse;}
    th{text-align:left;padding:.65rem 1rem;font-size:.76rem;color:var(--muted);
       text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--border);}
    td{padding:.7rem 1rem;border-bottom:1px solid var(--border);
       font-size:.88rem;vertical-align:middle;}
    tr:last-child td{border-bottom:none;}
    tr:hover td{background:#f8fafc;}
    .thumb{width:50px;height:50px;border-radius:8px;object-fit:cover;background:#e2e8f0;}
    .sbadge{display:inline-block;padding:.18rem .6rem;border-radius:999px;
            font-size:.75rem;font-weight:700;}
    .s-unclaimed{background:#dcfce7;color:#15803d;}
    .s-claimed{background:#f1f5f9;color:#475569;}
    .actions{display:flex;gap:.4rem;}
    .btn-sm{padding:.35rem .7rem;border:none;border-radius:6px;
            font-size:.78rem;font-weight:600;cursor:pointer;}
    .btn-outline{background:#fff;color:var(--text);border:1.5px solid var(--border);}
    .btn-outline:hover{border-color:var(--primary);color:var(--primary);}
    .btn-del{background:#fef2f2;color:var(--danger);border:1.5px solid #fecaca;}
    .btn-del:hover{background:#fee2e2;}
    .empty-table{text-align:center;padding:2.5rem;color:var(--muted);}
    .color-chip-row{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.35rem;}
    .color-chip{display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .65rem;
                border-radius:20px;border:2px solid var(--border);cursor:pointer;
                font-size:.8rem;user-select:none;background:#f8fafc;transition:border-color .15s,background .15s;}
    .color-chip.selected{border-color:var(--primary);background:#eff6ff;}
    .color-chip .cc-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0;}
  </style>
</head>
<body>
<header>
  <h1>Staff Portal &mdash; Lost &amp; Found</h1>
  <a href="/">&larr; Student View</a>
</header>

<main>
  <!-- Stats -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-num" id="stat-total">&mdash;</div>
      <div class="stat-label">Total Items Logged</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" id="stat-unclaimed" style="color:#16a34a">&mdash;</div>
      <div class="stat-label">Unclaimed</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" id="stat-claimed" style="color:#64748b">&mdash;</div>
      <div class="stat-label">Claimed / Returned</div>
    </div>
  </div>

  <!-- Upload -->
  <div class="upload-card">
    <div class="section-title">Log New Found Item</div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/*" style="display:none">
      <div class="icon">&#128247;</div>
      <p>Drag &amp; drop a photo here, or <strong>click to select</strong></p>
      <div class="fname" id="fileName"></div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label for="location">Where was it found?</label>
        <input class="form-input" type="text" id="location"
               placeholder="e.g. Room 204, Gym, Cafeteria">
      </div>
      <div class="form-group">
        <label for="staffNotes">Notes (optional)</label>
        <input class="form-input" type="text" id="staffNotes"
               placeholder="Any extra details&hellip;">
      </div>
    </div>
    <div style="display:flex;gap:.6rem;align-items:center;">
      <button class="btn btn-primary" id="analyzeBtn" disabled>Analyze Image</button>
      <button class="btn btn-outline" id="cancelBtn" style="display:none">Cancel</button>
    </div>

    <div class="detect-panel" id="detectPanel">
      <h3>&#10003; Image analyzed &mdash; review and confirm</h3>
      <div id="detectList"></div>
      <div class="edit-row">
        <div class="form-group">
          <label>Category</label>
          <select class="form-input" id="editCategory"></select>
        </div>
        <div class="form-group">
          <label>Subcategory</label>
          <select class="form-input" id="editSubcat"></select>
        </div>
        <div class="form-group" style="grid-column:1/-1">
          <label>Colors <span style="color:var(--muted);font-weight:400">(select all that apply)</span></label>
          <div class="color-chip-row" id="colorChipRow"></div>
        </div>
      </div>
      <button class="btn btn-success" id="saveBtn">Save Item to Database</button>
    </div>
  </div>

  <!-- Items table -->
  <div class="table-card">
    <div class="table-header">
      <div class="section-title" style="margin:0">All Items</div>
      <div class="filter-btns">
        <button class="fbtn active" data-status="">All</button>
        <button class="fbtn" data-status="unclaimed">Unclaimed</button>
        <button class="fbtn" data-status="claimed">Claimed</button>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Photo</th><th>Item</th><th>Color</th>
          <th>Location</th><th>Date Found</th><th>Status</th><th>Actions</th>
        </tr>
      </thead>
      <tbody id="itemsTbody"></tbody>
    </table>
  </div>
</main>

<script>
const TAXONOMY = {
  Apparel:["Hoodie","Jacket","Sweater","Shirt","Pants","Shorts","School Uniform","Clothing"],
  Drinkware:["Water Bottle","Insulated Bottle","Coffee Tumbler","Mug","Cup"],
  "Food Containers":["Lunch Box","Lunch Bag","Food Container"],
  Bags:["Backpack","Bag"],
  "School Supplies":["Pencil Pouch","Notebook","Textbook","Folder","Calculator"],
  Electronics:["Smartphone","Headphones","Earbuds","Charger","Tablet","Laptop"],
  Accessories:["Umbrella","Hat","Cap"],
  "Personal Essentials":["Keys","Wallet","Eyeglasses"],
  Footwear:["Shoes","Sneakers"],
  Other:["Uncategorized"]
};
const COLORS = ["black","white","gray","red","orange","yellow","green",
                "blue","purple","pink","brown","unknown"];
const STAFF_COLOR_HEX = {
  black:"#1a1a1a",white:"#e8e8e8",gray:"#808080",red:"#dc2626",orange:"#ea580c",
  yellow:"#ca8a04",green:"#16a34a",blue:"#2563eb",purple:"#7c3aed",
  pink:"#db2777",brown:"#78350f",unknown:"#94a3b8"
};

let currentImageId = null;
let tableStatusFilter = "";

// Populate category select
const catSel = document.getElementById("editCategory");
Object.keys(TAXONOMY).forEach(cat => {
  const o = document.createElement("option");
  o.value = o.textContent = cat;
  catSel.appendChild(o);
});
catSel.addEventListener("change", () => updateSubcats());

// Populate color chips
const chipRow = document.getElementById("colorChipRow");
COLORS.forEach(c => {
  const chip = document.createElement("div");
  chip.className = "color-chip";
  chip.dataset.color = c;
  chip.innerHTML = `<span class="cc-dot" style="background:${STAFF_COLOR_HEX[c]||'#94a3b8'}"></span>${c}`;
  chip.onclick = () => chip.classList.toggle("selected");
  chipRow.appendChild(chip);
});

function getSelectedColors() {
  const sel = Array.from(document.querySelectorAll("#colorChipRow .color-chip.selected"))
    .map(ch => ch.dataset.color);
  return sel.length ? sel : ["unknown"];
}

function setSelectedColors(colors) {
  const arr = Array.isArray(colors) ? colors : [colors];
  document.querySelectorAll("#colorChipRow .color-chip").forEach(chip => {
    chip.classList.toggle("selected", arr.includes(chip.dataset.color));
  });
}

function updateSubcats(selected) {
  const sub = document.getElementById("editSubcat");
  sub.innerHTML = (TAXONOMY[catSel.value] || [])
    .map(s => `<option value="${s}" ${s===selected?"selected":""}>${s}</option>`)
    .join("");
}
updateSubcats();

// Drag-and-drop upload
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
  document.getElementById("fileName").textContent = file.name;
  fileInput._file = file;
  analyzeBtn.disabled = false;
  document.getElementById("detectPanel").style.display = "none";
  document.getElementById("cancelBtn").style.display = "none";
  currentImageId = null;
}

function resetForm() {
  document.getElementById("fileName").textContent = "";
  document.getElementById("location").value = "";
  document.getElementById("staffNotes").value = "";
  document.getElementById("detectPanel").style.display = "none";
  document.getElementById("cancelBtn").style.display = "none";
  fileInput.value = ""; fileInput._file = null;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyze Image";
  currentImageId = null;
}

document.getElementById("cancelBtn").addEventListener("click", resetForm);

analyzeBtn.addEventListener("click", async () => {
  const file = fileInput._file;
  if (!file) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  document.getElementById("cancelBtn").style.display = "inline-block";
  const fd = new FormData();
  fd.append("image", file);
  try {
    const res = await fetch("/staff/detect", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed");
    currentImageId = data.image_id;
    showResults(data);
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
  } catch(e) {
    alert("Error: " + e.message);
    resetForm();
  }
});

function showResults(data) {
  const list = document.getElementById("detectList");
  if (!data.detections.length) {
    list.innerHTML = '<div class="detect-item">No items auto-detected &mdash; set category manually below.</div>';
  } else {
    list.innerHTML = data.detections.map(d =>
      `<div class="detect-item">
        <strong>${d.subgroup}</strong>
        <span style="color:#64748b">(${d.group})</span>
        &nbsp;&middot;&nbsp; ${d.color}
        <span class="conf">${(d.confidence*100).toFixed(0)}% confidence</span>
      </div>`
    ).join("");
  }
  if (data.suggested) {
    catSel.value = data.suggested.category;
    updateSubcats(data.suggested.subcat);
    setSelectedColors(data.suggested.color);
  }
  const panel = document.getElementById("detectPanel");
  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("saveBtn").addEventListener("click", async () => {
  if (!currentImageId) return;
  const saveBtn = document.getElementById("saveBtn");
  saveBtn.disabled = true;
  saveBtn.textContent = "Saving…";
  try {
    const res = await fetch("/staff/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: currentImageId,
        location: document.getElementById("location").value,
        category: catSel.value,
        subcat: document.getElementById("editSubcat").value,
        color: getSelectedColors().join(","),
        notes: document.getElementById("staffNotes").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to save");
    // Reset form
    document.getElementById("fileName").textContent = "";
    document.getElementById("location").value = "";
    document.getElementById("staffNotes").value = "";
    document.getElementById("detectPanel").style.display = "none";
    fileInput.value = ""; fileInput._file = null;
    analyzeBtn.disabled = true;
    currentImageId = null;
    loadStats(); loadTable();
    saveBtn.textContent = "✓ Saved!";
    setTimeout(() => { saveBtn.textContent = "Save Item to Database"; saveBtn.disabled = false; }, 2000);
  } catch(e) {
    alert("Error: " + e.message);
    saveBtn.disabled = false;
    saveBtn.textContent = "Save Item to Database";
  }
});

// Table
document.querySelectorAll(".fbtn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".fbtn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    tableStatusFilter = btn.dataset.status;
    loadTable();
  });
});

async function loadStats() {
  const data = await (await fetch("/stats")).json();
  document.getElementById("stat-total").textContent = data.total;
  document.getElementById("stat-unclaimed").textContent = data.unclaimed;
  document.getElementById("stat-claimed").textContent = data.claimed;
}

async function loadTable() {
  const params = new URLSearchParams();
  if (tableStatusFilter) params.set("status", tableStatusFilter);
  const items = await (await fetch("/items?" + params)).json();
  const tbody = document.getElementById("itemsTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="7"><div class="empty-table">No items found.</div></td></tr>`;
    return;
  }
  tbody.innerHTML = items.map(item => `
    <tr>
      <td><img class="thumb" src="/image/${item.image_id}" alt="" loading="lazy"
               onerror="this.style.visibility='hidden'"></td>
      <td><strong>${item.primary_subcat}</strong><br>
          <span style="font-size:.78rem;color:#64748b">${item.primary_category}</span></td>
      <td>${(item.color||'unknown').split(',').map(c=>c.trim()).join(' / ')}</td>
      <td>${item.location || "&mdash;"}</td>
      <td>${item.date_found}</td>
      <td><span class="sbadge s-${item.status}">
        ${item.status === "unclaimed" ? "Unclaimed" : "Claimed"}
      </span></td>
      <td><div class="actions">
        ${item.status === "unclaimed"
          ? `<button class="btn-sm btn-outline" onclick="setStatus(${item.id},'claimed')">Mark Claimed</button>`
          : `<button class="btn-sm btn-outline" onclick="setStatus(${item.id},'unclaimed')">Unmark</button>`}
        <button class="btn-sm btn-del" onclick="deleteItem(${item.id})">Delete</button>
      </div></td>
    </tr>`).join("");
}

async function setStatus(id, status) {
  await fetch("/items/" + id + "/status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  loadStats(); loadTable();
}

async function deleteItem(id) {
  if (!confirm("Delete this item? This cannot be undone.")) return;
  await fetch("/items/" + id, { method: "DELETE" });
  loadStats(); loadTable();
}

loadStats();
loadTable();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# LLM-powered search helpers
# ---------------------------------------------------------------------------

_anthropic_client: anthropic.Anthropic | None = None

_SEARCH_SYSTEM = (
    "You are a lost & found search assistant for a school. "
    "Extract intent from the user's query and return ONLY valid JSON with these fields:\n"
    '- "category": one of [Apparel, Drinkware, "Food Containers", Bags, '
    '"School Supplies", Electronics, Accessories, "Personal Essentials", Footwear, Other] or null\n'
    '- "color": one of [black, white, gray, red, orange, yellow, green, blue, purple, pink, brown] or null\n'
    '- "keywords": 1-4 word object-type phrase (omit color/category already captured) or null\n'
    '- "label": brief human-readable summary, e.g. "blue water bottle"\n'
    "Return only the JSON object, no markdown fences."
)


def _anthropic() -> anthropic.Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _llm_parse_query(query: str) -> dict:
    client = _anthropic()
    if not client:
        return {"category": None, "color": None, "keywords": query, "label": query}
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=_SEARCH_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = response.content[0].text.strip()
        return json.loads(text)
    except Exception:
        return {"category": None, "color": None, "keywords": query, "label": query}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def browse_page() -> str:
    return BROWSE_PAGE


@app.get("/staff", response_class=HTMLResponse)
def staff_page() -> str:
    return STAFF_PAGE


@app.get("/search/smart")
def smart_search(q: str = "", category: str = "", color: str = "") -> dict:
    if not q.strip():
        items = db.get_items(
            status="unclaimed",
            category=category if category and category != "All" else None,
            color=color if color and color != "All" else None,
        )
        return {"items": items, "interpreted": None, "llm_used": False}

    parsed = _llm_parse_query(q)
    llm_used = _anthropic() is not None

    # Explicit chip selections override LLM-inferred ones
    effective_category = (
        category if category and category != "All" else parsed.get("category")
    ) or None
    effective_color = (
        color if color and color != "All" else parsed.get("color")
    ) or None
    effective_keywords = parsed.get("keywords") or q

    items = db.get_items(
        status="unclaimed",
        search=effective_keywords,
        category=effective_category,
        color=effective_color,
    )
    return {
        "items": items,
        "interpreted": {"label": parsed.get("label", q)},
        "llm_used": llm_used,
    }


@app.get("/image/{image_id}")
def serve_image(image_id: str) -> FileResponse:
    matches = list(UPLOAD_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(matches[0])


class SaveRequest(BaseModel):
    image_id: str
    location: str = ""
    category: str = "Other"
    subcat: str = "Uncategorized"
    color: str = "unknown"
    notes: str = ""


@app.post("/staff/detect")
async def staff_detect(image: UploadFile = File(...)) -> dict:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    image_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{image_id}{suffix}"

    try:
        with saved_path.open("wb") as out:
            shutil.copyfileobj(image.file, out)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    try:
        detections = detector.detect(saved_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}") from exc

    detections.sort(key=lambda d: d.foreground_score, reverse=True)
    det_list = [
        {
            "label": d.label,
            "group": d.group,
            "subgroup": d.subgroup,
            "color": ", ".join(d.color),
            "confidence": round(d.confidence, 4),
        }
        for d in detections
    ]

    suggested = None
    if detections:
        best = max(detections, key=lambda d: d.foreground_score)
        suggested = {
            "category": best.group,
            "subcat": best.subgroup,
            "color": best.color,  # list[str]
        }

    return {
        "image_id": image_id,
        "filename": f"{image_id}{suffix}",
        "detections": det_list,
        "suggested": suggested,
    }


@app.post("/staff/save")
def staff_save(body: SaveRequest) -> dict:
    primary_color = body.color.split(",")[0].strip()
    description = f"{primary_color.capitalize()} {body.subcat}"
    all_dets: list = []

    item_id = db.insert_item(
        image_id=body.image_id,
        filename="",
        location=body.location,
        primary_category=body.category,
        primary_subcat=body.subcat,
        color=body.color,
        description=description,
        staff_notes=body.notes,
        all_detections=all_dets,
    )
    item = db.get_item(item_id)
    return item or {"id": item_id}


@app.get("/items")
def list_items(
    status: str = "",
    search: str = "",
    category: str = "",
    color: str = "",
) -> list[dict]:
    return db.get_items(
        status=status or None,
        search=search or None,
        category=category or None,
        color=color or None,
    )


@app.get("/items/{item_id}")
def get_item(item_id: int) -> dict:
    item = db.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


class StatusRequest(BaseModel):
    status: str


@app.post("/items/{item_id}/status")
def update_status(item_id: int, body: StatusRequest) -> dict:
    if body.status not in ("unclaimed", "claimed"):
        raise HTTPException(status_code=400, detail="status must be 'unclaimed' or 'claimed'")
    item = db.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.update_item(item_id, status=body.status)
    return {"ok": True}


@app.delete("/items/{item_id}")
def delete_item(item_id: int) -> dict:
    item = db.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete_item(item_id)
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    return db.get_stats()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
