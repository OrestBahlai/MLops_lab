import os
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def suggest_params(trial, cfg):
    space = cfg.random_forest
    return {
        "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
        "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
        "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low, space.min_samples_leaf.high),
    }

def objective_factory(cfg, X_train, y_train, X_test, y_test):
    def objective(trial):
        params = suggest_params(trial, cfg)
        
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)
            
            model = RandomForestClassifier(
                **params, 
                random_state=cfg.seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred)
            
            mlflow.log_metric(cfg.hpo.metric, score)
            
            return score
    return objective

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    train_df = pd.read_csv(cfg.data.train_path)
    test_df = pd.read_csv(cfg.data.test_path)
    
    target = cfg.data.target_col
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.json")
        
        if cfg.hpo.sampler.lower() == "random":
            sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, y_train, X_test, y_test)
        
        print(f"Starting Optuna study with {cfg.hpo.n_trials} trials...")
        study.optimize(objective, n_trials=cfg.hpo.n_trials)
        
        best_trial = study.best_trial
        print(f"\nBest Trial: {best_trial.number} | Best Score ({cfg.hpo.metric}): {best_trial.value:.4f}")
        
        mlflow.log_metric(f"best_{cfg.hpo.metric}", best_trial.value)
        mlflow.log_dict(best_trial.params, "best_params.json")
        
        best_model = RandomForestClassifier(**best_trial.params, random_state=cfg.seed)
        best_model.fit(X_train, y_train)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            
        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, cfg.mlflow.model_name)
            
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=cfg.mlflow.model_name,
                version=mv.version,
                stage=cfg.mlflow.stage
            )
            print(f"\nModel registered as '{cfg.mlflow.model_name}', version {mv.version} in {cfg.mlflow.stage} stage.")
            
        print("Optimization finished. Best model saved and logged to MLflow.")

if __name__ == "__main__":
    main()