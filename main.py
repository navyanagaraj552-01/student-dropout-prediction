import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
os.makedirs('results', exist_ok=True)

df = pd.read_csv("data/data.csv", sep=";")
print("Shape:", df.shape)
print(df["Target"].value_counts())

plt.figure()
df["Target"].value_counts().plot(kind='bar', color=['red','orange','green'])
plt.title('Target: Dropout / Enrolled / Graduate')
plt.tight_layout()
plt.savefig('results/01_class_balance.png')

leak_cols = [c for c in df.columns if "Curricular units" in c]
df_clean = df.drop(columns=leak_cols)
print(f"Dropped {len(leak_cols)} leakage columns")

X = df_clean.drop(columns=["Target"])
y = df_clean["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Baseline_Logistic": LogisticRegression(max_iter=1000),
    "Advanced_RandomForest": RandomForestClassifier(n_estimators=150, random_state=42)
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title(f'{name}')
    plt.savefig(f'results/confusion_{name}.png')

print("\nDONE!")