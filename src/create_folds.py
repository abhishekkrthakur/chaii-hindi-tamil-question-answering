import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../input/train.csv")
df["kfold"] = -1

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for f, (t_, v_) in enumerate(kf.split(X=df, y=df.language.values)):
    df.loc[v_, "kfold"] = f

df.to_csv("../input/train_folds.csv", index=False)
