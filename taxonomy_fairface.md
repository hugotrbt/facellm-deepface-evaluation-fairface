# FairFace – Taxonomy and Evaluation Protocol

## 1. Dataset

**Dataset**: FairFace  
**Split**: validation / test (as defined by FairFace)

FairFace provides human-annotated labels for **gender**, **race/ethnicity**, and **age**. No ground-truth annotations are available for facial expression.

---

## 2. Evaluated Attributes

The following attributes are evaluated in this work:

- **Gender**
- **Race / Ethnicity (coarse-grained)**
- **Age (binned)**

Facial expression is explicitly excluded from quantitative evaluation due to the absence of ground-truth annotations in FairFace.

---

## 3. Gender Taxonomy

### 3.1 Ground Truth (FairFace)

FairFace provides binary gender annotations:

```
Gender_GT ∈ {Male, Female}
```

### 3.2 Evaluation Taxonomy

The evaluation taxonomy follows the ground truth exactly:

```
Gender ∈ {Male, Female}
```

### 3.3 Model Outputs Handling

- Vision-language models (e.g., FaceLLM) may output additional values such as `unknown`.
- During evaluation, predictions mapped to `unknown` are counted as **incorrect**.
- The `unknown` category may be analyzed separately for qualitative discussion.

---

## 4. Race / Ethnicity Taxonomy

### 4.1 Ground Truth (FairFace – Fine-Grained)

FairFace provides the following race / ethnicity categories:

```
Race_GT ∈ {
  Black,
  East Asian,
  Southeast Asian,
  Middle Eastern,
  White,
  Indian,
  Latino_Hispanic
}
```

### 4.2 Evaluation Taxonomy (Coarse-Grained)

Due to differences in output granularity between models and common practice in the literature, a coarse-grained taxonomy is used for evaluation:

```
Race_Coarse ∈ {
  Asian,
  Black,
  White,
  Latino_Hispanic,
  Middle_Eastern,
  Indian
}
```

### 4.3 Ground Truth Mapping

The following mapping is applied to FairFace ground-truth labels:

```
East Asian       → Asian
Southeast Asian  → Asian
Black            → Black
White            → White
Middle Eastern   → Middle_Eastern
Indian           → Indian
Latino_Hispanic  → Latino_Hispanic
```

### 4.4 Model Output Mapping

Model predictions are normalized and mapped to the same coarse-grained taxonomy using lexical normalization and keyword matching (e.g., `asian`, `african`, `latino`, `hispanic`, etc.). Predictions that cannot be mapped reliably are assigned to `unknown` and counted as incorrect during evaluation.

---

## 5. Age Taxonomy

### 5.1 Ground Truth Age Bins (FairFace)

FairFace provides age annotations in the following bins:

```
Age_GT ∈ {
  0–2,
  3–9,
  10–19,
  20–29,
  30–39,
  40–49,
  50–59,
  60–69,
  70+
}
```

### 5.2 Evaluation Strategy

Models produce age estimates in different formats:

- **FaceLLM**: age ranges (e.g., `20–30`, `30–40`)
- **DeepFace**: single numeric age value

To ensure fair comparison, age is evaluated using a **bin-overlap strategy**.

### 5.3 Age Evaluation Rules

- **FaceLLM** prediction is considered **correct** if the predicted age range overlaps with the ground-truth age bin.
- **DeepFace** prediction is considered **correct** if the predicted numeric age falls within the ground-truth age bin.

Formally:

```
Age_Correct = 1 if overlap(predicted_range, GT_bin)
Age_Correct = 0 otherwise
```

No continuous age error (e.g., MAE) is reported in the primary evaluation.

---

## 6. Evaluation Metrics

For each evaluated attribute, the following metrics are reported:

- **Gender**: Accuracy, confusion matrix
- **Race (coarse)**: Accuracy, macro-F1 score, confusion matrix
- **Age**: Bin overlap accuracy

All metrics are computed on the same image splits for all models.

---

## 7. Methodological Notes

- Ground-truth annotations are never modified; only evaluation mappings are applied.
- All models are evaluated in a **zero-shot** setting (no fine-tuning).
- Prompt design and output normalization are treated as part of the evaluation pipeline.

This protocol ensures a fair and reproducible comparison between vision-language models and dedicated face attribute classifiers.

