# Lost & Found CV to Google Sheets

This project is a starter pipeline for lost-and-found operations:

- Accept an image submission through an API endpoint.
- Detect likely items (bags, bottles, apparel, school supplies, electronics, etc.).
- Auto-map each detection to a **group** and a **subgroup**.
- Build a highly descriptive subgroup label using item type + dominant color + size.
- Write data to Google Sheets:
  - `Detections` tab for per-item logs.
  - `Summary` tab for running totals by group + descriptive subgroup.
  - `Group Totals` tab for overall totals per top-level group.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create env file:

```bash
cp .env.example .env
```

Set:

- `GOOGLE_SERVICE_ACCOUNT_JSON`: absolute path to service account key.
- `GOOGLE_SHEET_ID`: ID from your sheet URL.

Share your Google Sheet with the service account email (Editor access).

If you only want local item detection UI first, you can skip Google Sheet vars.

## 2) Run API

```bash
uvicorn src.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Open the web UI in your browser:

- **Student browse** (gallery + search): [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Staff portal** (upload, run detection, save to database): [http://127.0.0.1:8000/staff](http://127.0.0.1:8000/staff)

```bash
open http://127.0.0.1:8000/
# or
open http://127.0.0.1:8000/staff
```

On **Staff**, upload an image and click **Analyze Image** to see detections, then save if you want.

## 3) Submit an Image

```bash
curl -X POST "http://127.0.0.1:8000/submit" \
  -F "image=@/absolute/path/to/image.jpg" \
  -F "submitter=Front Office" \
  -F "location=Cafeteria"
```

On every submission:

- Rows are appended to `Detections` (one row per detected item).
- Counts are incremented in `Summary` for each `group + descriptive_subgroup`.
- Counts are incremented in `Group Totals` for each top-level group.

If Sheets is not configured, `/submit` still works and only returns detection JSON.

## Notes on Accuracy

- Default model uses YOLO-World prompts for common lost items.
- If YOLO-World is unavailable, it falls back to a general YOLO model.
- For production-level accuracy, fine-tune on your own campus/school dataset.

## Bootstrap training labels (no labeled data yet)

Put raw photos in `training_data/incoming_photos`, then run:

```bash
python3 -m src.auto_label --incoming training_data/incoming_photos --output training_data/labeled --conf 0.20
```

This generates:
- YOLO train/val images and labels under `training_data/labeled`
- `training_data/labeled/data.yaml` for YOLO training

Then quickly review/fix labels in CVAT, Label Studio, or Roboflow before training.

