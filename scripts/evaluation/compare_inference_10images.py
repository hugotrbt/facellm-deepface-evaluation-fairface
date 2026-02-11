import os
import json
import csv
import re

def extract_age(s: str) -> int:

    # range like "5-7 years" or 30-40
    nombres = re.findall(r"\d+", s)
    a, b = map(int, nombres[:2])

    return (a + b) // 2

FACELLM_DIR = "results/facellm_parsed"
DEEPFACE_DIR = "results/deepface"
OUT_DIR = "results/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []

for fname in sorted(os.listdir(FACELLM_DIR)):
    if not fname.endswith(".json"):
        continue

    img_id = fname.replace(".json", "")

    with open(os.path.join(FACELLM_DIR, fname)) as f:
        facellm = json.load(f)

    with open(os.path.join(DEEPFACE_DIR, fname)) as f:
        deepface = json.load(f)[0]  # DeepFace returns a list

    age_f = facellm.get("age")
    age_d = deepface.get("age")

    gender_f = facellm.get("gender")
    gender_d = deepface.get("dominant_gender")

    race_f = facellm.get("race")
    race_d = deepface.get("dominant_race")

    expression_f = facellm.get("expression")
    expression_d = deepface.get("dominant_emotion")

    rows.append({
        "image_id": img_id,
        "age_facellm": age_f,
        "age_deepface": age_d,
        "age_abs_diff": abs(extract_age(age_f) - age_d) if age_f is not None and age_d is not None else None,
        "gender_facellm": gender_f,
        "gender_deepface": gender_d,
        "gender_match": int(gender_f == gender_d) if gender_f and gender_d else None,
        "race_facellm": race_f,
        "race_deepface": race_d,
        "race_match": int(race_f == race_d) if race_f and race_d else None,
	"expression_facellm": expression_f,
	"expression_deepface": expression_d,
	"expression_match": int(expression_f == expression_d) if expression_f and expression_d else None
    })

out_path = os.path.join(OUT_DIR, "summary.csv")

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] Comparison saved to {out_path}")