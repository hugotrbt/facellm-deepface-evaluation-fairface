import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
gt = pd.read_csv("data/fairface_3k/train_labels_inferred_only.csv")
df_deepface = pd.read_csv("data/fairface_3k/deepface_3k.csv")
df_facellm = pd.read_csv("data/fairface_3k/facellm_3k.csv")

# Rename columns (GT vs Pred)
gt = gt.rename(columns={"gender": "gender_gt", "race": "race_gt", "age": "age_gt"})
df_deepface = df_deepface.rename(columns={"gender": "gender_pred", "race": "race_pred", "age": "age_pred"})
df_facellm = df_facellm.rename(columns={"gender": "gender_pred", "race": "race_pred", "age": "age_pred"})

# Label alignment
gt["race_gt"] = gt["race_gt"].replace(("Southeast Asian", "East Asian"), "Asian")
gt["age_gt"] = gt["age_gt"].replace("more than 70", "70+")

# Merge GT + predictions
eval_deepface = gt.merge(df_deepface, on="image_id", how="inner")
eval_facellm = gt.merge(df_facellm, on="image_id", how="inner")

# =========================
# TASK (change here)
# =========================
TASK = "age"  # "race" or "gender" or "age"

if TASK == "race":
    gt_col = "race_gt"
    pred_col = "race_pred"
    conf_deepface = "race_confidence"
    conf_facellm = "race_confidence"
elif TASK == "gender":
    gt_col = "gender_gt"
    pred_col = "gender_pred"
    conf_deepface = "gender_confidence"
    conf_facellm = "gender_confidence"
elif TASK == "age":
    gt_col = "age_gt"
    pred_col = "age_pred"
    conf_deepface = "face_confidence"   # IMPORTANT: face_confidence = age_confidence for DeepFace
    conf_facellm = "age_confidence"
else:
    raise ValueError("TASK must be 'race', 'gender', or 'age'")

# Correctness
eval_deepface["correct"] = (eval_deepface[pred_col] == eval_deepface[gt_col]).astype(int)
eval_facellm["correct"] = (eval_facellm[pred_col] == eval_facellm[gt_col]).astype(int)

# =========================
# DeepFace
# =========================
if conf_deepface in ["race_confidence", "gender_confidence"]:
    eval_deepface[conf_deepface] = eval_deepface[conf_deepface] / 100

bins = np.linspace(0, 1, 11)
eval_deepface["conf_bin"] = pd.cut(eval_deepface[conf_deepface], bins=bins, include_lowest=True)

calibration_deepface = (
    eval_deepface
    .groupby("conf_bin", observed=True)
    .agg(
        mean_confidence=(conf_deepface, "mean"),
        accuracy=("correct", "mean"),
        count=("correct", "size")
    )
    .reset_index()
)

ece_deepface = np.sum(
    np.abs(calibration_deepface["accuracy"] - calibration_deepface["mean_confidence"])
    * (calibration_deepface["count"] / len(eval_deepface))
)

print("\n=== DeepFace ===")
print("TASK:", TASK)
print(calibration_deepface)
print("ECE:", ece_deepface)

plt.figure()
plt.plot([0, 1], [0, 1])
plt.plot(calibration_deepface["mean_confidence"], calibration_deepface["accuracy"], marker="o")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title(f"DeepFace - {TASK} (ECE={ece_deepface:.4f})")
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

# =========================
# FaceLLM
# =========================

print(eval_facellm[conf_facellm].value_counts().head(20))
eval_facellm[conf_facellm] = eval_facellm[conf_facellm].astype(float)

# Since FaceLLM confidence is basically shared across age/gender/race,
# we just use the chosen task confidence as "shared_confidence".
eval_facellm["shared_confidence"] = eval_facellm[conf_facellm]

calibration_facellm = (
    eval_facellm
    .groupby("shared_confidence")
    .agg(
        mean_confidence=("shared_confidence", "mean"),
        accuracy=("correct", "mean"),
        count=("correct", "size")
    )
    .reset_index()
    .sort_values("shared_confidence")
)

ece_facellm = np.sum(
    np.abs(calibration_facellm["accuracy"] - calibration_facellm["mean_confidence"])
    * (calibration_facellm["count"] / len(eval_facellm))
)

print("\n=== FaceLLM ===")
print("TASK:", TASK)
print(calibration_facellm)
print("ECE:", ece_facellm)

plt.figure()
plt.plot([0, 1], [0, 1])
plt.plot(calibration_facellm["mean_confidence"], calibration_facellm["accuracy"], marker="o")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title(f"FaceLLM - {TASK} (ECE={ece_facellm:.4f})")
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()