import pandas as pd

gt = pd.read_csv("train_labels.csv")
df_deepface = pd.read_csv("deepface_3k.csv")
df_facellm = pd.read_csv("facellm_3k.csv")

gt["image_id"] = (
    gt["image_id"]
    .str.replace("train/", "", regex=False)
    .str.replace(".jpg", "", regex=False)
    .astype(int)
)

# Union is required because some images (2 images) were not processed during FaceLLM inference
ids_inferred = set(df_facellm["image_id"].astype(int))

gt_filtered = gt[gt["image_id"].isin(ids_inferred)].copy()
gt_filtered.to_csv("train_labels_inferred_only.csv", index=False)

print("GT original :", len(gt))
print("DeepFace :", df_deepface["image_id"].nunique())
print("FaceLLM :", df_facellm["image_id"].nunique())
print("GT filtré :", len(gt_filtered))
