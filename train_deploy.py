"""
Train the DEPLOYMENT ensemble (sklearn-only, no XGBoost) for the Vercel web app.
Keeps the app light + fast cold-start while staying accurate.
"""
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RS = 42
COLS = ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
        "Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens",
        "Albumin","Albumin_and_Globulin_Ratio","Dataset"]

df = pd.read_csv("data/ilpd.csv", names=COLS)
df = df.dropna(subset=["Albumin"])
df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
    df["Albumin"] / (df["Total_Protiens"] - df["Albumin"]).replace(0, np.nan))
df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
    df["Albumin_and_Globulin_Ratio"].median()).clip(lower=0)
df["Gender_Male"] = (df["Gender"].str.strip().str.lower()=="male").astype(int)
df["Risk"] = (df["Dataset"]==1).astype(int)
df["Bilirubin_Ratio"] = (df["Direct_Bilirubin"]/df["Total_Bilirubin"].replace(0,np.nan)).fillna(0)
df["AST_ALT_Ratio"] = (df["Aspartate_Aminotransferase"]/df["Alamine_Aminotransferase"].replace(0,np.nan)).replace([np.inf,-np.inf],0).fillna(1)
df["Protein_Albumin_Diff"] = df["Total_Protiens"]-df["Albumin"]

feature_cols = ["Age","Gender_Male","Total_Bilirubin","Direct_Bilirubin",
    "Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase",
    "Total_Protiens","Albumin","Albumin_and_Globulin_Ratio","Bilirubin_Ratio",
    "AST_ALT_Ratio","Protein_Albumin_Diff"]

X = df[feature_cols].values; y = df["Risk"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

lr  = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                ("clf",LogisticRegression(class_weight="balanced",max_iter=2000))])
rf  = Pipeline([("imp",SimpleImputer(strategy="median")),
                ("clf",RandomForestClassifier(n_estimators=400,class_weight="balanced_subsample",random_state=RS))])
gb  = Pipeline([("imp",SimpleImputer(strategy="median")),
                ("clf",GradientBoostingClassifier(random_state=RS))])
svm = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                ("clf",SVC(kernel="rbf",class_weight="balanced",probability=True,random_state=RS))])

ensemble = VotingClassifier(estimators=[("lr",lr),("rf",rf),("gb",gb),("svm",svm)],
                            voting="soft", weights=[1,1,2,1]).fit(Xtr,ytr)

pred = ensemble.predict(Xte); proba = ensemble.predict_proba(Xte)[:,1]
m = {
 "accuracy": round(float(accuracy_score(yte,pred))*100,1),
 "precision": round(float(precision_score(yte,pred))*100,1),
 "recall": round(float(recall_score(yte,pred))*100,1),
 "f1": round(float(f1_score(yte,pred))*100,1),
 "auc": round(float(roc_auc_score(yte,proba)),3),
 "cm": confusion_matrix(yte,pred).tolist(),
 "n": int(len(df)),
 "features": feature_cols,
}
# quick CV on training for reference
cvacc = float(cross_val_score(ensemble, Xtr, ytr, cv=cv, scoring="accuracy").mean())
m["cv_accuracy"] = round(cvacc*100,1)

joblib.dump({"model":ensemble,"feature_columns":feature_cols}, "api/liver_model.pkl")
with open("api/model_metrics.json","w") as f: json.dump(m,f,indent=2)
print(json.dumps(m, indent=2))
print("saved api/liver_model.pkl + api/model_metrics.json")
