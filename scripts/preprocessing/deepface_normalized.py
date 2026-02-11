import json
import os
import pandas as pd

INPUT_FILE = "results/deepface_full.jsonl"
OUTPUT_DIR = "results/"

model = "DeepFace"
image_id = []
age = []
face_confidence = []
gender = []
gender_confidence = []
race = []
race_confidence = []


def normalize_race_deepface(race_raw: str) -> str:
    r = race_raw.lower().strip()

    race_map = {
        "asian": "Asian",
        "black": "Black",
        "white": "White",
        "indian": "Indian",
        "middle eastern": "Middle Eastern",
        "latino hispanic": "Latino_Hispanic"
    }

    if r not in race_map:
        raise ValueError(f"Unknown race value from DeepFace: {race_raw}")

    return race_map[r]

def age_to_fairface_bin(age: int) -> str:
    if age <= 2:
        return "0-2"
    elif age <= 9:
        return "3-9"
    elif age <= 19:
        return "10-19"
    elif age <= 29:
        return "20-29"
    elif age <= 39:
        return "30-39"
    elif age <= 49:
        return "40-49"
    elif age <= 59:
        return "50-59"
    elif age <= 69:
        return "60-69"
    else:
        return "more than 70"


with open(INPUT_FILE, "r", encoding="utf-8") as f :
  for line in f:
    data = json.loads(line)
    image_path = data["image_path"]
    raw = data["raw_output"]
    raw = raw[0]
    file_name = os.path.basename(image_path)
    image_id.append(int(os.path.splitext(file_name)[0]))

    estimated_age = raw["age"]
    age_bin = age_to_fairface_bin(estimated_age)
    age.append(age_bin)

    face_confidence.append(raw["face_confidence"])

    if raw["dominant_gender"] == "Man":
      gender.append("Male")
      gender_confidence.append(float(raw["gender"]["Man"]))
    else :
      gender.append("Female")
      gender_confidence.append(float(raw["gender"]["Woman"]))

    dominant_race = raw["dominant_race"]
    race_norm = normalize_race_deepface(dominant_race)
    race.append(race_norm)
    race_confidence.append(float(raw["race"][dominant_race]))
    
df_deepface = pd.DataFrame({
  "image_id": image_id,
  "model": model,
  "age": age,
  "face_confidence": face_confidence,
  "gender": gender,
  "gender_confidence": gender_confidence,
  "race": race,
  "race_confidence": race_confidence
})

df_deepface.to_csv("deep_face_3k.csv", index=False)