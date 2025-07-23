import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load the merged features dataset ===
INPUT_FILE = "datasource/processed/features_final_all_layers.csv"
df = pd.read_csv(INPUT_FILE)

# === Identify relevant flag columns ===
flag_cols = [col for col in df.columns if col.endswith('_flag') or 'xai_flag' in col]
label_col = 'label'

# === Create output folder ===
os.makedirs("output/phase2/flag_analysis", exist_ok=True)

# === Summary Table: % flagged per class ===
summary = {}
for col in flag_cols:
    class_counts = df.groupby(label_col)[col].mean()
    summary[col] = class_counts.to_dict()

summary_df = pd.DataFrame(summary).T.fillna(0)
summary_df.to_csv("output/phase2/flag_analysis/flag_summary_by_class.csv")
print("Saved summary stats to flag_summary_by_class.csv")

# === Plotting: heatmap of flag frequency by class ===
plt.figure(figsize=(10, 6))
sns.heatmap(summary_df, annot=True, fmt=".2f", cmap="YlOrBr")
plt.title("Average Flag Value per Class (1 = Flagged, 0 = Not Flagged)")
plt.xlabel("Class Label")
plt.ylabel("Flag Feature")
plt.tight_layout()
plt.savefig("output/phase2/flag_analysis/flag_heatmap.png")
print("Saved heatmap to flag_heatmap.png")

# Optional: Bar plots per flag
for col in flag_cols:
    plt.figure()
    sns.barplot(x=label_col, y=col, data=df)
    plt.title(f"Flag Frequency by Class: {col}")
    plt.savefig(f"output/phase2/flag_analysis/{col}_barplot.png")
    plt.close()