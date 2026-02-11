import pandas as pd
import json
import os

#Replace by outputs/raw/facellm_full.jsonl
INPUT_FILE = "results/facellm_full.jsonl"


model = "Facellm"
image_id = []
age = []
age_confidence = []
gender = []
gender_confidence = []
ethnicity = []
ethnicity_confidence = []

with open(INPUT_FILE, "r", encoding="utf-8") as f :
  for line in f:
    data = json.loads(line)
    image_path = data["image_path"]
    raw = data.get("raw_output", "")

    if data.get("status") != "ok":
      continue

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    raw_dict = json.loads(raw)

    confidence = raw_dict["confidence"]

    file_name = os.path.basename(image_path)
    image_id.append(int(os.path.splitext(file_name)[0]))

    age.append(raw_dict["age_range"])
    age_confidence.append(confidence["age"])

    gender.append(raw_dict["gender"])
    gender_confidence.append(confidence["gender"])

    ethnicity.append(raw_dict["ethnicity"])
    ethnicity_confidence.append(confidence["ethnicity"])


df_facellm = pd.DataFrame ({
  "image_id": image_id,
  "model": model,
  "age": age,
  "age_confidence": age_confidence,
  "gender": gender,
  "gender_confidence": gender_confidence,
  "race": ethnicity,
  "race_confidence": ethnicity_confidence
})

df_facellm.to_csv("face_llm_3k.csv", index = False)
