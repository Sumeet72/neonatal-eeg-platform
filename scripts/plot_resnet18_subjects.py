import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path(
    "results/ResNet18_evaluation/ResNet18_subject_wise_results.csv"
)

OUTPUT_DIR = Path("results/ResNet18_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Only subjects that actually contain seizure epochs
positive = df[df["seizure_epochs"] > 0].copy()

# Sort by sensitivity
positive = positive.sort_values(
    "sensitivity",
    ascending=True
)

plt.figure(figsize=(12, 10))

plt.barh(
    positive["subject_id"],
    positive["sensitivity"]
)

plt.xlabel("Sensitivity")
plt.ylabel("Subject")
plt.title(
    "ResNet-18 Subject-wise Seizure Sensitivity"
)

plt.xlim(0, 1)

plt.tight_layout()

output_file = (
    OUTPUT_DIR /
    "ResNet18_subject_wise_sensitivity.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("SUBJECT-WISE SENSITIVITY GRAPH")
print("=" * 70)
print(f"Subjects with seizures: {len(positive)}")
print(f"Saved to: {output_file}")