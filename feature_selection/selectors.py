import logging
import logzero
import numpy as np
import pandas as pd
import sklearn
from logzero import logger
from sklearn.metrics import check_scoring
from sklearn.model_selection import ShuffleSplit, cross_validate, BaseCrossValidator
from sklearn.pipeline import Pipeline

__version__ = "2.3.2"


class FeatureSelector:
    """
    This class is a generic class, it should be inhereted and the methods load_data, train_model & evaluate_model should
    be implemented by the end user.
    Please follow the following guidances in order to make the pipeline work:
    - load_data: this function reads the data and returns in pandas data-frame two variables X,y
    - featureSelectionCV: is the func to use for feature selection
    - for custom action at the end of each iteration, action_on_update() should be used.
    """

    def __init__(self, df, target_col, log_level=logging.DEBUG, custom_cv=None):
        """
        Initiate the dataframe and the target column
        :param df: dataframe used for training the model
        :param target_col: target column to be predicted
        :param log_level: INFO, DEBUG, ERROR or CRITICAL
        :param custom_cv: custom cross validation func, it recommended to use subclass of BaseCrossValidator
        """
        self.df = df.copy(deep=True)
        self.target_col = target_col
        self.model_is_pipeline = False
        self.list_of_deleted_col = []
        logzero.loglevel(level=log_level)

        # Using custom cv if provided by user
        if custom_cv is None:
            self.cv = ShuffleSplit(test_size=0.3, random_state=0)
        else:
            if not (issubclass(custom_cv.__class__, BaseCrossValidator)):
                logger.warn(f"Be aware that {custom_cv.__class__} is not a subclass of {BaseCrossValidator}, this "
                            f"could cause problems during the cross-validation process. Make sure it is compatible "
                            f"with sklearn CrossValidate framework.")
            self.cv = custom_cv

    def load_data(self):
        """
        Return X, y. All of them in pandas dataframe format.
        """
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return X, y

    def build_model(self):
        """
        Here define your model with hyperparameters and return model that implements 'fit' method
        """
        raise NotImplementedError

    def __evaluate_iteration(self, small_is_better, new_score, baseline_score, tolerance=0):
        """
        This method is returning a boolean saying if new score is better baseline score
        :param small_is_better: boolean for smaller value is better
        :param new_score: score to compared with
        :param baseline_score: score to compare
        :param tolerance: TO BE IMPLEMENTED
        :return:
        """

        new_score_higher = (new_score - baseline_score >= tolerance)
        if small_is_better:
            if new_score_higher:
                return False
            else:
                return True
        else:
            if new_score_higher:
                return True
            else:
                return False

    def cv_evaluate_model(self, model, X, y, n_splits=5, scoring_metric="f1_macro", nb_ft=None, n_jobs=None):
        """
        This method uses a model definition to train and cross validate, it returns
        the average score obtained after the cv process.
        :param model: model definition (implements fit)
        :param X: Dataset with training features
        :param y: target
        :param n_splits: number of folds for cross validation
        :param scoring_metric: scoring metric to compute
        :return: average of scores
        """
        
        self.cv.__setattr__("n_splits", n_splits)
        scorer = check_scoring(model, scoring=scoring_metric)

        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            scoring={"score": scorer},
            cv=self.cv,
            pre_dispatch="2*n_jobs",
            error_score=np.nan,
            return_estimator=True,
            return_train_score=True,
            n_jobs=n_jobs
        )

        scores = cv_results["test_score"]
        list_features = []

        if nb_ft is not None:
            for item in cv_results["estimator"]:
                if self.model_is_pipeline:
                    clf = item.steps[-1][1]
                else:
                    clf = item
                df = pd.DataFrame(
                    [(k, v) for k, v in clf.get_booster().get_fscore().items()],
                    columns=["feature", "importance"],
                ).sort_values("importance", ascending=False)
                list_features.extend(df.feature[:nb_ft])

        return {"cv_score": scores.mean(),
                "cv_score_std": scores.std(),
                "train_score": cv_results["train_score"].mean(),
                "train_score_std": cv_results["train_score"].std(),
                f"top_{nb_ft}_ft": list(set(list_features))}

    def action_on_update(self, cols_to_drop):
        pass

    def __action_on_update(self, X, cols_to_drop, persist=True):
        """
        Private func, SHOULD not be modified.
        For custom actions on update (columns drop), please implement the method action_on_update()

        :param X: is the dataframe being used in the loop
        :param cols_to_drop: columns to be dropped from the training dataframe.
        :param persist: Choose to persist the modifications in the dataframe.
        """
        self.action_on_update(cols_to_drop)
        if persist:
            X.drop(cols_to_drop, axis=1, inplace=True)
            self.list_of_deleted_col.extend(cols_to_drop)
            logger.debug(f"list of deleted features : {self.list_of_deleted_col}")

    def feature_selection_cv(
        self,
        small_is_better=False,
        scoring_metric="f1_macro",
        cv=5,
        fast_version=False,
        nb_ft=None,
        n_jobs=None,
        tolerance=0
    ):
        """
        Returns the list of selected features after comparing the result of shuffling
        the column and retraining using CV.
        :param small_is_better: True if we are looking into reducing scoring_metric
        :param scoring_metric: what is the metric that is used to evaluate the model
        :param cv: number of folds
        :param fast_version: is we want to use the first cv to remove bottom features using model importance (Only for
                             xgboost model)
        :param nb_ft: number of features to keep in the first round of cv
        :param n_jobs: number of jobs for cross validation
        :param tolerance: tolerance when comparing performances
        :return: Array list with column names
        """
        X, y = self.load_data()
        clf = self.build_model()

        # Check if it's sklearn pipeline
        if clf.__class__ == Pipeline:
            model_type = str(clf.steps[-1][1].__class__)
            self.model_is_pipeline = True
            logger.debug(f"Working with sklearn pipeline, model type: {model_type}")
        else:
            model_type = str(clf.__class__)

        # Check if the model is compatible with fast_version
        if "xgboost" not in model_type and fast_version:
            logger.error(
                f"The type of model used {clf.__class__} is not supported for fast_version, the classic"
                f" version will be re-activated."
            )
            fast_version = False
        # If fast version, then the first iteration we only keep top parameters given by xgboost to speed up
        # the process of feature selection
        if fast_version:
            if nb_ft is None:
                raise AttributeError("nb_ft should not be null if fast_version parameter is used. Expected type: int.")

            base_line_cv_result = self.cv_evaluate_model(
                model=clf,
                X=X,
                y=y,
                n_splits=cv,
                scoring_metric=scoring_metric,
                nb_ft=nb_ft,
                n_jobs=n_jobs
            )
            base_line_score = base_line_cv_result["cv_score"]
            cols_to_ignore = [col for col in X.columns if col not in base_line_cv_result[f"top_{nb_ft}_ft"]]
            self.__action_on_update(X, cols_to_ignore, persist=True)
        else:
            base_line_cv_result = self.cv_evaluate_model(
                model=clf,
                X=X,
                y=y,
                n_splits=cv,
                scoring_metric=scoring_metric,
                nb_ft=None,
                n_jobs=n_jobs
                )
            base_line_score = base_line_cv_result["cv_score"]

        logger.info(f"Starting score {base_line_score}")

        features_impact = []
        for col in X.columns:
            # 1. delete col in the train_df
            origin_train_col = X[col].copy(deep=True)
            self.__action_on_update(X, [col], persist=False)
            # 2. Train the model with the deleted value and compare to baseline score
            new_clf = self.build_model()
            new_cv_result = self.cv_evaluate_model(
                model=new_clf,
                X=X.drop([col], axis=1),
                y=y,
                n_splits=cv,
                scoring_metric=scoring_metric,
                n_jobs=n_jobs
                )
            new_score = new_cv_result["cv_score"]
            logger.debug(f"Old score {base_line_score}, new score {new_score}")

            # adding iteration scores
            features_impact.append({
                'feature': col,
                f"{scoring_metric} old CV score": base_line_score,
                f"{scoring_metric} new CV score": new_score,
                f"{scoring_metric} new CV score std": new_cv_result["cv_score_std"],
                f"{scoring_metric} CV gain": base_line_score - new_score,
                f"{scoring_metric} train-cv gap": base_line_cv_result["train_score"] - new_cv_result["train_score"]
                })

            if self.__evaluate_iteration(small_is_better, new_score=new_score,
                                         baseline_score=base_line_score, tolerance=tolerance):
                logger.debug(f"Improvement or nothing changed ==> delete {col}")
                new_clf = self.build_model()

                base_line_cv_result = self.cv_evaluate_model(
                    model=new_clf,
                    X=X.drop([col], axis=1),
                    y=y,
                    n_splits=cv,
                    scoring_metric=scoring_metric,
                    n_jobs=n_jobs
                )
                base_line_score = base_line_cv_result["cv_score"]

                logger.info(f"New base line score: {base_line_score}")
                self.__action_on_update(X, [col], persist=True)
            else:
                logger.debug(f"The model is worse ==> keep {col}")
                X[col] = origin_train_col

        return X.columns, features_impact
