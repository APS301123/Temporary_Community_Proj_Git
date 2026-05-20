# Training Data Folder

Use this folder to collect photos for training your lost-and-found model.

## Where to put new photos

- Put all new, unlabeled photos in `incoming_photos/`.
- Keep original image names if possible.

## Bootstrap labels when you have zero labels

From project root, run:

```bash
python3 -m src.auto_label --incoming training_data/incoming_photos --output training_data/labeled --conf 0.20
```

This will:
- detect objects in each image
- create YOLO label files automatically
- split data into train/val
- generate `training_data/labeled/data.yaml`

Then open and correct those labels in CVAT/Label Studio/Roboflow.

## Suggested next pipeline

1. Add photos to `incoming_photos/`.
2. Label them with a tool like CVAT or Label Studio.
3. Export labels in YOLO format.
4. Place files into:
   - `labeled/images/train/`
   - `labeled/images/val/`
   - `labeled/labels/train/`
   - `labeled/labels/val/`

## Tip

Collect variety:
- multiple angles
- different lighting
- crowded scenes
- partial occlusions
- different colors/materials of the same item type
