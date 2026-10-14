import shutil
from pathlib import Path
from PIL import Image

src_files = {
    "normal.jpg": Path("dataset/oral-lesion-dataset-org/SMART-OM/01. Normal/01. Unannotated/01. Dorsal tongue/6 - DT.jpg"),
    "variation.jpg": Path("dataset/oral-lesion-dataset-org/SMART-OM/02. Variation from normal/01. Unannotated/01. Dorsal tongue/SMITA000249_R_DT.jpeg"),
    "opmd.jpg": Path("dataset/oral-lesion-dataset-org/SMART-OM/03. OPMD/01. Unannotated/03. Left buccal mucosa/SMITA00007_W_LB.jpg"),
    "cancer.jpg": Path("dataset/oral-lesion-dataset-org/SMART-OM/04. Oral Cancer/01. Unannotated/07. Upper arch/Ca 1.JPG")
}

out_dir = Path("assets/presets")
out_dir.mkdir(parents=True, exist_ok=True)

for name, p in src_files.items():
    img = Image.open(p).convert("RGB")
    target_path = out_dir / name
    img.save(target_path, "JPEG", quality=92)
    print(f"Saved {name} ({target_path.stat().st_size} bytes)")
