
from sklearn.model_selection import cross_val_score
import os
import joblib

class Objective(object):
    def __init__(self, model,  X_train, y_train, cv=5, scoring='f1', mode='transform'):
        if mode == 'transform':
            self.X_train = model[:-1].fit_transform(X_train, y_train)
            self.y_train = y_train
        elif mode == 'resample':
            self.X_train, self.y_train = model[:-1].fit_resample(X_train, y_train)
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.mode = mode

class ObjectiveLGBM(Objective):
    def __init__(self, model, X_train, y_train, cv=5, scoring='f1', mode='transform'):
        super().__init__(model, X_train, y_train, cv, scoring, mode)

    def __call__(self, trial):
        x, y = self.X_train, self.y_train
        model = self.model
        cv = self.cv
        scoring = self.scoring

        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        classifier_obj = model['classifier'].set_params(**param)
        classifier_obj.fit(x, y)
        
        pipe = model[:-1]
        pipe.steps.append(['classifier',classifier_obj])

        fname = "{}_{:04d}.pkl".format(classifier_obj.__class__.__name__, trial.number)
        with open(os.path.join(os.pardir, 'models', fname), "wb") as fout:
            joblib.dump(pipe, fout)
        
        score = cross_val_score(classifier_obj, x, y, cv=cv, scoring=scoring)
        avg_score = score.mean()
        return avg_score
