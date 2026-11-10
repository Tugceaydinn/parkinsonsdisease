import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv(sys.argv[1])
df["subject"] = df["name"].str.extract(r"(S\d+)")
print("recordings:", len(df), "subjects:", df["subject"].nunique())
print("subjects by status:", df.groupby("status")["subject"].nunique().to_dict())
print("recordings per subject min/median/max:", df.groupby("subject").size().agg(["min","median","max"]).tolist())
X = df.drop(columns=["name","status","subject"]); y = df["status"].values; g = df["subject"].values

def mk(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", model)])
models = {
  "Logistic Regression": lambda: mk(LogisticRegression(max_iter=5000, random_state=42)),
  "Random Forest": lambda: mk(RandomForestClassifier(n_estimators=300, random_state=42)),
}
scoring = {"acc":"accuracy","prec":"precision","rec":"recall","f1":"f1","auc":"roc_auc"}

print("\n== A) Notebook split: single 80/20 random split, random_state=42 (n_test=%d)" % int(round(0.2*len(df))))
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
for n,f in models.items():
    p=f().fit(Xtr,ytr); pr=p.predict(Xte); pp=p.predict_proba(Xte)[:,1]
    print(f"{n:20s} acc={accuracy_score(yte,pr):.3f} prec={precision_score(yte,pr):.3f} rec={recall_score(yte,pr):.3f} f1={f1_score(yte,pr):.3f} auc={roc_auc_score(yte,pp):.3f}")

print("\n== B) 5-fold stratified CV, recordings shuffled (subjects can appear in train AND test)")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for n,f in models.items():
    r=cross_validate(f(),X,y,cv=cv,scoring=scoring)
    print(f"{n:20s} " + " ".join(f"{k}={r['test_'+k].mean():.3f}±{r['test_'+k].std():.3f}" for k in scoring))

print("\n== C) 5-fold stratified GROUP CV (all recordings of a person stay in one fold) <- no subject leakage")
gcv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
for n,f in models.items():
    r=cross_validate(f(),X,y,cv=gcv,groups=g,scoring=scoring)
    print(f"{n:20s} " + " ".join(f"{k}={r['test_'+k].mean():.3f}±{r['test_'+k].std():.3f}" for k in scoring))
print("\nMajority-class baseline accuracy:", round(max(np.mean(y), 1-np.mean(y)),3))
