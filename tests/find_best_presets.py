import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.inference_engine import OralXAIInferenceEngine

engine = OralXAIInferenceEngine.get_instance()
base = Path("dataset/oral-lesion-dataset-org/SMART-OM")

targets = {
    "01. Normal": "Normal",
    "02. Variation from normal": "Variation from normal",
    "03. OPMD": "OPMD",
    "04. Oral Cancer": "Oral Cancer"
}

best_images = {}

for folder, target_class in targets.items():
    print(f"\nSearching high-confidence samples for {target_class} in {folder}...")
    folder_path = base / folder / "01. Unannotated"
    all_imgs = list(folder_path.glob("**/*.*"))
    print(f"Found {len(all_imgs)} images.")
    
    scored = []
    for p in all_imgs[:30]: # check first 30
        if p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            img = Image.open(p)
            res = engine.predict(img)
            pred = res['predicted_class']
            conf = res['confidence']
            if pred == target_class:
                scored.append((conf, p, res))
        except Exception:
            continue
            
    scored.sort(reverse=True, key=lambda x: x[0])
    if scored:
        best_conf, best_p, best_res = scored[0]
        print(f"  -> BEST: {best_p.name} ({round(best_conf*100, 1)}% {best_res['predicted_class']})")
        best_images[target_class] = (best_p, best_conf, best_res)
    else:
        print(f"  -> No exact match in sample for {target_class}")

print("\nSummary of best representative clinical cases:")
for k, v in best_images.items():
    print(f"{k}: {v[0]} ({round(v[1]*100, 1)}%)")
