import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np



# Load data
gt = pd.read_csv("train_labels_inferred_only.csv")
df_deepface = pd.read_csv("deepface_3k.csv")
df_facellm = pd.read_csv("facellm_3k.csv")


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


# GENDER
crosstab_gender_deepface = pd.crosstab(
    eval_deepface["gender_gt"],
    eval_deepface["gender_pred"],
    normalize="index"
)

crosstab_gender_facellm = pd.crosstab(
    eval_facellm["gender_gt"],
    eval_facellm["gender_pred"],
    normalize="index"
)

acc_deepface = (eval_deepface["gender_gt"] == eval_deepface["gender_pred"]).mean()
acc_facellm = (eval_facellm["gender_gt"] == eval_facellm["gender_pred"]).mean()

print("\n--- GENDER ---")
print("DeepFace (normalized by GT):\n", crosstab_gender_deepface)
print("Gender accuracy DeepFace:", acc_deepface * 100)

print("\nFaceLLM (normalized by GT):\n", crosstab_gender_facellm)
print("Gender accuracy FaceLLM:", acc_facellm * 100)


# RACE
race_labels = ["Asian", "Black", "Indian", "Latino_Hispanic", "Middle Eastern", "White"]


confusionmatrix_deepface = confusion_matrix(
    eval_deepface["race_gt"],
    eval_deepface["race_pred"],
    labels=race_labels,
    normalize="true"
)

confusionmatrix_facellm = confusion_matrix(
    eval_facellm["race_gt"],
    eval_facellm["race_pred"],
    labels=race_labels,
    normalize="true"
)

f1score_deepface = f1_score(eval_deepface["race_gt"], eval_deepface["race_pred"], average="macro")
f1score_facellm = f1_score(eval_facellm["race_gt"], eval_facellm["race_pred"], average="macro")

print("\n--- RACE ---")
print("Macro-F1 DeepFace:", f1score_deepface)
print("Macro-F1 FaceLLM:", f1score_facellm)

fig_race, axes_race = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(confusionmatrix_deepface, display_labels=race_labels).plot(
    ax=axes_race[0], xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=False
)
axes_race[0].set_title("DeepFace — Race (normalized)")
axes_race[0].set_xlabel("Pred")
axes_race[0].set_ylabel("GT")

ConfusionMatrixDisplay(confusionmatrix_facellm, display_labels=race_labels).plot(
    ax=axes_race[1], xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=False
)
axes_race[1].set_title("FaceLLM — Race (normalized)")
axes_race[1].set_xlabel("Pred")
axes_race[1].set_ylabel("GT")

plt.tight_layout()
plt.show()


# AGE
age_acc_deepface = accuracy_score(eval_deepface["age_gt"], eval_deepface["age_pred"])
age_f1m_deepface = f1_score(eval_deepface["age_gt"], eval_deepface["age_pred"], average="macro")

age_acc_facellm = accuracy_score(eval_facellm["age_gt"], eval_facellm["age_pred"])
age_f1m_facellm = f1_score(eval_facellm["age_gt"], eval_facellm["age_pred"], average="macro")

print("\n--- AGE ---")
print("Accuracy age DeepFace:", age_acc_deepface)
print("Accuracy age FaceLLM:", age_acc_facellm)
print("Macro-F1 age DeepFace:", age_f1m_deepface)
print("Macro-F1 age FaceLLM:", age_f1m_facellm)


age_bins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

cm_age_deepface = confusion_matrix(
    eval_deepface["age_gt"],
    eval_deepface["age_pred"],
    labels=age_bins,
    normalize="true"
)

cm_age_facellm = confusion_matrix(
    eval_facellm["age_gt"],
    eval_facellm["age_pred"],
    labels=age_bins,
    normalize="true"
)

bin_to_idx = {b: i for i, b in enumerate(age_bins)}

y_true_df = eval_deepface["age_gt"].map(bin_to_idx).to_numpy()
y_pred_df = eval_deepface["age_pred"].map(bin_to_idx).to_numpy()

mae_bins_df = np.mean(np.abs(y_true_df - y_pred_df))
within_1_df = np.mean(np.abs(y_true_df - y_pred_df) <= 1)
within_2_df = np.mean(np.abs(y_true_df - y_pred_df) <= 2)

y_true_fl = eval_facellm["age_gt"].map(bin_to_idx).to_numpy()
y_pred_fl = eval_facellm["age_pred"].map(bin_to_idx).to_numpy()

mae_bins_fl = np.mean(np.abs(y_true_fl - y_pred_fl))
within_1_fl = np.mean(np.abs(y_true_fl - y_pred_fl) <= 1)
within_2_fl = np.mean(np.abs(y_true_fl - y_pred_fl) <= 2)

print("\nDistance en nombre de bins DeepFace:", mae_bins_df)
print("À ±1 bin DeepFace:", within_1_df)
print("À ±2 bins DeepFace:", within_2_df)

print("\nDistance en nombre de bins FaceLLM:", mae_bins_fl)
print("À ±1 bin FaceLLM:", within_1_fl)
print("À ±2 bins FaceLLM:", within_2_fl)

fig_age, axes_age = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(cm_age_deepface, display_labels=age_bins).plot(
    ax=axes_age[0], xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=False
)
axes_age[0].set_title("DeepFace — Age (normalized)")
axes_age[0].set_xlabel("Pred")
axes_age[0].set_ylabel("GT")

ConfusionMatrixDisplay(cm_age_facellm, display_labels=age_bins).plot(
    ax=axes_age[1], xticks_rotation=45, values_format=".2f", cmap="Blues", colorbar=False
)
axes_age[1].set_title("FaceLLM — Age (normalized)")
axes_age[1].set_xlabel("Pred")
axes_age[1].set_ylabel("GT")

plt.tight_layout()
plt.show()
