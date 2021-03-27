import optuna
import pandas as pd
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

df = pd.read_csv("smart_road_measurements_new_d_weather.csv", header=0)

bins = [0, 0.35, 0.7, 1]
labels = [0, 1, 2]
df["Friction"] = pd.cut(df["Friction"], bins, labels=labels)


Y = df.loc[:, "Friction"].to_numpy()
df = df.drop("Friction", axis=1)
df = df.drop("Distance", axis=1)
df = df.drop("Date", axis=1)
df = df.drop("Time(+01:00)", axis=1)
X = df.values


def objective(trial: Trial, x, y) -> float:
    imputer = KNNImputer(n_neighbors=2)
    imputer.fit_transform(X)

    params = {
        "tree_method": "gpu_hist",
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.5),
        "colsample_bytree": trial.suggest_discrete_uniform(
            "colsample_bytree", 0.1, 1, 0.01
        ),
        "nthread": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=5, random_state=1337, shuffle=True)
    return cross_val_score(model, x, y, scoring="accuracy", cv=cv).mean()


study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=1337))
study.optimize(
    lambda trial: objective(trial, X, Y), n_trials=50, show_progress_bar=True
)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.to_csv("output/res.csv", sep="\t")
print(study.best_trial)
