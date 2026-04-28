import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/eda", exist_ok=True)

# 1. Load data
customer_data = pd.read_csv("data/raw/telco_churn.csv")

# 2. Print shape
print("=" * 60)
print("SHAPE OF DATASET")
print("=" * 60)
print(f"Rows: {customer_data.shape[0]}, Columns: {customer_data.shape[1]}")
print()

# 3. Print column names and data types
print("=" * 60)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 60)
print(customer_data.dtypes)
print()

# 4. Print first 5 rows
print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(customer_data.head())
print()

# 5. Print count of missing values per column
print("=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
print(customer_data.isnull().sum())
print()

# 6. Print Churn value counts
print("=" * 60)
print("CHURN VALUE COUNTS")
print("=" * 60)
print(customer_data["Churn"].value_counts())
print()

# --- Convert TotalCharges to numeric (needed for plots) ---
customer_data["TotalCharges"] = pd.to_numeric(customer_data["TotalCharges"], errors="coerce")

# ===== PLOT 1: Churn Distribution Bar Chart =====
plt.figure(figsize=(8, 5))
sns.countplot(x="Churn", data=customer_data)
plt.title("Churn Distribution")
plt.tight_layout()
plt.savefig("outputs/eda/churn_distribution.png", dpi=150)
plt.close()
print("Saved: outputs/eda/churn_distribution.png")

# ===== PLOT 2: Correlation Heatmap =====
numeric_cols = customer_data.select_dtypes(include="number")
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: outputs/eda/correlation_heatmap.png")

# ===== PLOT 3: Box Plots (MonthlyCharges and TotalCharges vs Churn) =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(x="Churn", y="MonthlyCharges", data=customer_data, ax=axes[0])
axes[0].set_title("MonthlyCharges vs Churn")
sns.boxplot(x="Churn", y="TotalCharges", data=customer_data, ax=axes[1])
axes[1].set_title("TotalCharges vs Churn")
fig.suptitle("MonthlyCharges and TotalCharges vs Churn", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/eda/boxplots_charges_vs_churn.png", dpi=150)
plt.close()
print("Saved: outputs/eda/boxplots_charges_vs_churn.png")

# ===== PLOT 4: Tenure Histogram by Churn =====
plt.figure(figsize=(10, 6))
sns.histplot(data=customer_data, x="tenure", hue="Churn", bins=30, kde=True)
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig("outputs/eda/tenure_vs_churn.png", dpi=150)
plt.close()
print("Saved: outputs/eda/tenure_vs_churn.png")

print()
print("=" * 60)
print("ALL 4 PLOTS SAVED SUCCESSFULLY TO outputs/eda/")
print("=" * 60)
