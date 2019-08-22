import pandas as pd
import numpy as np
import os
import json
import datetime
import itertools
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
#from sklearn.externals import joblib


# noinspection SpellCheckingInspection
class ModelBase(ABC):

    """
    This is the base class for every prediction model we are going to use.
    It helps the prediction model to be trained, tested, and saved.

    Each prediction model should inherit this base class, while specifying
    "y_variable", "handle_NA", and "scale".

    Following is the step leveraging this class.

    Training with CV>
    1. prepare CV dataset
    2. prepare grid by passing parameters to "construct_grid"
    2. pass inputs to "tune_parameters"
    3. pass outputs to "best_grid"

    Training>
    1. prepare parameters, trining and test dataset
    2. pass dataset to "get_data_for_model" function to change data to model specific format
    3. call "fit_model"
    4. call "test_model"

    Save model>
    1. take model from "fit_model"
    2. take scaler from "get_scaler" if neccessary. It takes training from DataHelper
    3. take training from "get_data_for_model"
    3. call "save_model" and pass all inputs

    Predict>
    1. instantiate object passing the filename (saved model).
    2. call "set_model_from_file"
    3. take dataset to predict from DataHelper
    4. call "predict_model"
    """

    filename_var = "variable_selected.txt"

    def __init__(self, filename=None):
        self.filename = filename
        self.y_variable = "bad_loan"
        self.handle_NA = False
        self.scale = False
        self.type = "classification"

        # this needs to be defined separately in child
        self.x_raw_variables = self.set_x_raw_variables()

        # we set these variables from the saved file
        # no need during the model fitting process
        self.model = None
        self.scaler = None
        self.x_variables = None
        self.model_description = None

    def __str__(self):
        return "ModelBase Object"

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):

        if filename is None:
            self._filename = filename
        elif type(filename) == str and filename[-3:] == "sav":
            self._filename = filename
        else:
            self._filename = None

    def save_model(self, model, scaler, training, name):

        # we need to save x_variables used and description as well
        df = training.drop(columns=[self.y_variable])
        xvar = df.columns.values
        desc = self.get_model_description(model)

        info = {"model": model, "scaler": scaler, "x_variables": xvar, "description": desc}

        print("Model is saved on {}".format(name))
        joblib.dump(info, self.get_modelname(name))  # name should be .sav file

    def get_model_description(self, model):
        d = datetime.datetime.strftime(datetime.datetime.today(), "%m/%d/%Y")
        return "{}, time: {}".format(self.__str__(), d)

    def load_model(self, name):
        print("Model is loaded from {}".format(name))
        model = self.get_modelname(name)
        return joblib.load(model)

    def set_model_from_file(self):

        if self.filename is None:
            print("Set filename first!")
            pass
        else:
            info = self.load_model(self.filename)
            self.model = info["model"]
            self.scaler = info["scaler"]
            self.x_variables = info["x_variables"]
            self.model_description = info["description"]

    @abstractmethod
    def set_x_raw_variables(self):
        pass

    @abstractmethod
    def fit_model(self, df, *args, **kwargs):
        pass

    @abstractmethod
    def test_model(self, df, model=None):
        pass

    def predict_model(self, df, verbose=False):

        """
        This function predicts the data using the saved model only. Note that we need to save model
        in the member first, before doing the prediction. It first transform the data to an appropriate
        form using the x_variables from the saved model. Please note it returns probabilities,
        not actual classifications.
        """

        # check model is ready
        if self.model is None or self.x_variables is None:
            print("Please set the model and x variables first !!!")
            pass

        # clean data
        df = self.clean_data(df, prediction=True, index=False)
        n = df.shape[0]

        # standardize data
        if self.scale and self.scaler is not None:
            types = df.dtypes
            cvars = [x for x in self.x_raw_variables if types[x] == "float" or types[x] == "int"]
            df.loc[:, cvars] = self.scaler.transform(df[cvars])

        # get dummy variables
        df = pd.get_dummies(df)
        cols = df.columns.values

        # check missing variables in our model
        missing_vars = [x for x in cols if x not in self.x_variables]
        if len(missing_vars) and verbose:
            print("Following x_variables are missing in training data: ")
            print(missing_vars)
            pass

        # check missing variables in our data. Then add columns with 0 values
        # This case is possible since prediction dataset is much smaller than our training dataset.
        # In this case, the dummy columns (one-hot-encoding) will not expanded fully to our training.
        # This is why we add missing columns with 0 values.
        missing_cols = [x for x in self.x_variables if x not in cols]
        for col in missing_cols:
            df[col] = np.zeros(n)

        # reorder columns
        df = df[self.x_variables]

        # finally, let's predict !!
        if self.type == "classification":
            pred = self.model.predict_proba(df)[:, 1]
        elif self.type == "regression":
            pred = self.model.predict(df)
        else:
            pred = None

        return pred

    def get_data_for_model(self, training, test):

        """
        This function does following:
            1. takes training and test,
            2. clean data as defined in clean_data method, such as
                - filter out x, y raw variables,
                - remove NA (if self.handle_NA == True)
            3. standardize continuous variables (if self.scale == True)
            4. one-hot-encoding,
            5. split back to the training and test.
        """

        # clean data
        training = self.clean_data(training)
        test = self.clean_data(test)

        # memorize the size to split
        train_size = training.shape[0]
        test_size = test.shape[0]

        # standardize continuous variables
        if self.scale:
            types = training.dtypes
            cvars = [x for x in self.x_raw_variables if types[x] == "float" or types[x] == "int"]
            scaler = StandardScaler()
            training.loc[:, cvars] = scaler.fit_transform(training[cvars])
            test.loc[:, cvars] = scaler.transform(test[cvars])

        # combine and make dummy columns
        combined = training
        combined = combined.append(test)
        combined = pd.get_dummies(combined)

        # get new training and test
        training, test = combined.iloc[0:train_size, :], combined.iloc[train_size:(train_size + test_size), :]
        if training.shape[0] != train_size or test.shape[0] != test_size:
            print("Something wrong here !!!")
            pass

        return training, test

    def clean_data(self, df, prediction=False, index=False):

        if prediction:
            variables = self.x_raw_variables
        else:
            variables = self.x_raw_variables + [self.y_variable]

        df = df[variables]
        if not self.handle_NA:
            df = df.dropna()

        if index:
            return df.index
        else:
            return df

    def clean_data_ind(self, df):

        variables = self.x_raw_variables + [self.y_variable]
        df = df[variables]
        if not self.handle_NA:
            df = df.dropna()
        return df.index

    def tune_parameters(self, df, CVs, grids, verbose=True):

        """
        This function conduct Cross-Validation for grid-search of hyper-parameters.
        The CV data is provided by DataHelper.
        """

        results = []
        for grid in grids:
            scores = []
            if verbose:
                print("fitting for {}".format(grid))
            for i, CV in enumerate(CVs):
                training, test = self.get_data_for_model(df.loc[CV[0], :], df.loc[CV[1], :])
                model = self.fit_model(training, **grid)
                score = self.test_model(test, model)
                scores.append(score)
                if verbose:
                    print(score)

            result = {'scores': scores, 'avg_score': self.avg_cv_results(scores), 'grid': grid}
            results.append(result)

        return results

    @staticmethod
    def avg_cv_results(measures):
        keys = measures[0].keys()
        values = np.array([list(x.values()) for x in measures]).sum(axis=0)
        values /= len(measures)
        return {x: y for x, y in zip(keys, values)}

    @staticmethod
    def construct_grids(**kwargs):

        keys = kwargs.keys()
        vals = kwargs.values()
        grids = list(itertools.product(*vals))
        grids = [{k: x[i] for i, k in enumerate(keys)} for x in grids]

        return grids

    @staticmethod
    def best_grid(scores, measure):

        raw_scores = [x["scores"] for x in scores]
        avg_scores = [x["avg_score"][measure] for x in scores]
        grids = [x["grid"] for x in scores]

        ind = int(np.argmax(avg_scores))
        return raw_scores[ind], avg_scores[ind], grids[ind]

    @staticmethod
    def get_modelname(name):
        path = os.getcwd() + "/Model/"
        return path + name

    @staticmethod
    def get_filename(name):
        path = os.getcwd() + "/Data/"
        return path + name

    def get_scaler(self, training):

        """
        This function returns StandardScaler after fitting it to continuous variables among
        x_raw_variables. When using this scaler, we always get continuous variables from x_raw_variables.
        The fitted scaler from this fucntion always has to be passed when saving the final model, if
        the model needs scaling.

        The input training is the training dataset taken directly from DataHelper.
        """

        types = training.dtypes
        cvars = [x for x in self.x_raw_variables if types[x] == "float" or types[x] == "int"]
        scaler = StandardScaler()
        scaler.fit(training[cvars])

        return scaler


# noinspection SpellCheckingInspection
class ModelClassification(ModelBase):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # object type
        self.type = "classification"

        # define classification specific variables
        self.testMeasure = ["score", "AUC"]
        self.y_variable = "bad_loan"

    def __str__(self):
        return "Classification Model Base"

    @abstractmethod
    def set_x_raw_variables(self):
        pass

    @abstractmethod
    def fit_model(self, df, *args, **kwargs):
        pass

    def test_model(self, df, model=None, plot=False):

        """
        This function test the fitted model. If a model object is not passed, then it looks
        for a stored model object (self.model). It returns the score as dictionary we chose.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable]

        if model is None:
            model = self.model

        results = {}
        if "score" in self.testMeasure:
            results["score"] = model.score(x_data, y_data)
        if "OOB_score" in self.testMeasure:
            results["OOB_score"] = model.oob_score_
        if "AUC" in self.testMeasure:
            results["AUC"] = self.get_auc(model, x_data, y_data, plot=plot)

        return results

    @staticmethod
    def get_auc(model, df_x, df_y, plot=False):

        pred = model.predict_proba(df_x)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(df_y, pred)
        auc = metrics.auc(fpr, tpr)

        if plot:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (area = %0.2f)' % auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

        return auc


# noinspection SpellCheckingInspection
class ModelLogistic(ModelClassification):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # setting
        self.handle_NA = False
        self.scale = True

        # basic variables
        self.penalty = "l2"
        self.solver = "liblinear"
        self.max_iter = 500
        self.tolerance = 1e-4 * 5

    def __str__(self):
        return "Logistic Regression Object"

    def get_model_description(self, model):
        desc = super().get_model_description(model)
        return "{}, C: {}, penalty: {}, solver: {}, max_iter: {}, tol: {}".format(
            desc, model.C, model.penalty, model.solver, model.max_iter, model.tol)

    def set_x_raw_variables(self):

        """
        This function returns the columns to use in training data from DataHelper.
        This is NOT actual variable names used in the model, due to the additional need of
        one hot encoding.
        """

        file = self.get_filename(self.filename_var)

        with open(file, 'r') as f:
            variables = json.load(f)
        return variables["no_missing_cols"]

    def fit_model(self, df, *args, **kwargs):

        """
        This function fits the l2 penalized logistic regression on a given training data set.
        Note that C=1/alpha needs to be passed as an arguments. It returns the fitted model object.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable]

        # fit l2 penalized logistic regression
        model = LogisticRegression(penalty=self.penalty, C=kwargs["C"], solver=self.solver,
                                   max_iter=self.max_iter, tol=self.tolerance)
        model.fit(x_data, y_data)
        return model


# noinspection SpellCheckingInspection
class ModelRandomForest(ModelClassification):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # setting
        self.handle_NA = False
        self.scale = False

        # add more variables - not sure we really need this 3??
        self.n_jobs = -1
        self.testMeasure = ["score", "OOB_score", "AUC"]

    def __str__(self):
        return "Random Forest Object"

    def get_model_description(self, model):
        desc = super().get_model_description(model)
        return "{}, n_tree/m_try/splits: {}/{}/{} ".format(desc, model.n_estimators,
                                                           model.max_features, model.min_samples_split)

    def set_x_raw_variables(self):

        """
        This function returns the columns to use in training data from DataHelper.
        This is NOT actual variable names used in the model, due to the additional need of
        one hot encoding.
        """

        file = self.get_filename(self.filename_var)

        with open(file, 'r') as f:
            variables = json.load(f)
        return variables["no_missing_cols"]

    def fit_model(self, df, *args, **kwargs):

        """
        This function fits the Random Forest on a given training data set.
        Note that n_estimators, max_features, and min_samples_split needs to be
        passed as an arguments. It returns the fitted model object.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable]

        # fit l2 penalized logistic regression
        model = RandomForestClassifier(n_estimators=kwargs["n_estimators"],
                                       max_features=kwargs["max_features"],
                                       min_samples_split=kwargs["min_samples_split"],
                                       n_jobs=self.n_jobs,
                                       oob_score=True)
        model.fit(x_data, y_data)

        return model


# noinspection SpellCheckingInspection
class ModelXGBClassfication(ModelClassification):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # setting
        self.handle_NA = True
        self.scale = True

        # add more measures
        self.n_jobs = 8
        self.objective = "binary:logistic"
        self.testMeasure = ["score", "AUC"]

    def __str__(self):
        return "XGBoost Classification Object"

    def get_model_description(self, model):
        desc = super().get_model_description(model)
        return "{}, eta: {}, num_rounds: {}, max_depth: {}, subsample: {}".\
            format(desc, model.learning_rate, model.n_estimators, model.max_depth, model.subsample)

    def set_x_raw_variables(self):

        """
        This function returns the columns to use in training data from DataHelper.
        Please note it returns full predictor column names including the new variables, after 2015.
        This is NOT actual variable names used in the model, due to the additional need of
        one hot encoding.
        """

        file = self.get_filename(self.filename_var)

        with open(file, 'r') as f:
            variables = json.load(f)
        return variables["interim_cols"]

    def fit_model(self, df, *args, **kwargs):

        """
        This function fits the XGBoost on a given training data set. Note that many varaibles
        need to be passed as an arguments. The number of jobs (CPU) and objective should be set as
        member variables. It returns the fitted model object.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable].values

        # fit XGBoost
        model = xgb.XGBClassifier(n_estimators=kwargs["n_estimators"],
                                  learning_rate=kwargs["learning_rate"],
                                  max_depth=kwargs["max_depth"],
                                  min_child_weight=kwargs["min_child_weight"],
                                  gamma=kwargs["gamma"],
                                  subsample=kwargs["subsample"],
                                  colsample_bytree=kwargs["colsample_bytree"],
                                  reg_alpha=kwargs["reg_alpha"],
                                  objective=self.objective,
                                  n_jobs=self.n_jobs,
                                  missing=None)
        model.fit(x_data, y_data)

        return model

    def fit_n_estimator(self, df, fold, stopping, **kwargs):

        """
        This function use xgboost cv to fit the n_estimators. This is convinient since
        We can set the early stopping rule for a quick calibration.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable].values

        # d-matrix
        d_train = xgb.DMatrix(x_data, label=y_data)

        # Cross Validation
        results = xgb.cv(kwargs, d_train, num_boost_round=kwargs["n_estimators"], nfold=fold,
                         metrics='auc', early_stopping_rounds=stopping)

        print("stop at {} test-mean-auc: {}, test-std-auc: {}".format(
            results.shape[0], round(results.iloc[-1, 2], 4), round(results.iloc[-1, 3], 4)))

        return results.shape[0]


# noinspection SpellCheckingInspection
class ModelRegression(ModelBase):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # object type
        self.type = "regression"

        # define classification specific variables
        self.testMeasure = ["score", "MSE"]
        self.y_variable = "log_return"

    def __str__(self):
        return "Regression Model Base"

    @abstractmethod
    def set_x_raw_variables(self):
        pass

    @abstractmethod
    def fit_model(self, df, *args, **kwargs):
        pass

    def test_model(self, df, model=None, plot=False):

        """
        This function test the fitted model. If a model object is not passed, then it looks
        for a stored model object (self.model). It returns the score we chose as dictionary.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable]

        if model is None:
            model = self.model

        results = {}
        if "score" in self.testMeasure:
            results["score"] = model.score(x_data, y_data)
        if "MSE" in self.testMeasure:
            results["MSE"] = self.get_mse(model, x_data, y_data)

        return results

    @staticmethod
    def get_mse(model, df_x, df_y):

        pred = model.predict(df_x)
        mse = metrics.mean_squared_error(df_y, pred, multioutput="uniform_average")

        return mse

    @staticmethod
    def best_grid(scores, measure):
        raw_scores = [x["scores"] for x in scores]
        avg_scores = [x["avg_score"][measure] for x in scores]
        grids = [x["grid"] for x in scores]

        if measure == "MSE":
            ind = int(np.argmin(avg_scores))
        elif measure == "score":
            ind = int(np.argmax(avg_scores))

        return raw_scores[ind], avg_scores[ind], grids[ind]


# noinspection SpellCheckingInspection
class ModelLinear(ModelRegression):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # setting
        self.handle_NA = False
        self.scale = True

    def __str__(self):
        return "Linear Regression Object"

    def get_model_description(self, model):
        desc = super().get_model_description(model)
        return desc

    def set_x_raw_variables(self):

        """
        This function returns the columns to use in training data from DataHelper.
        This is NOT actual variable names used in the model, due to the additional need of
        one hot encoding.
        """

        file = self.get_filename(self.filename_var)

        with open(file, 'r') as f:
            variables = json.load(f)
        return variables["no_missing_cols"]

    def fit_model(self, df, *args, **kwargs):

        """
        This function fits plain liner regression. This is used for benchmark model only.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable]

        # fit linear regression
        model = LinearRegression()
        model.fit(x_data, y_data)
        return model


# noinspection SpellCheckingInspection
class ModelLinearLasso(ModelRegression):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # setting
        self.handle_NA = False
        self.scale = True

        # basic variables
        self.max_iter = 1000
        self.tolerance = 1e-5

    def __str__(self):
        return "Lasso Linear Regression Object"

    def get_model_description(self, model):
        desc = super().get_model_description(model)
        return "{}, alpha: {}, max_iter: {}, tol: {}".format(desc, model.alpha, model.max_iter, model.tol)

    def set_x_raw_variables(self):

        """
        This function returns the columns to use in training data from DataHelper.
        This is NOT actual variable names used in the model, due to the additional need of
        one hot encoding.
        """

        file = self.get_filename(self.filename_var)

        with open(file, 'r') as f:
            variables = json.load(f)
        return variables["no_missing_cols"]

    def fit_model(self, df, *args, **kwargs):

        """
        This function fits the lasso linear regression on a given training data set.
        Note that alpha needs to be passed as an arguments. It returns the fitted model object.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable]

        # fit lasso linear regression
        model = Lasso(alpha=kwargs["alpha"], max_iter=self.max_iter, tol=self.tolerance)
        model.fit(x_data, y_data)
        return model


# noinspection SpellCheckingInspection
class ModelXGBRegression(ModelRegression):

    def __init__(self, filename=None):

        # call base init
        super().__init__(filename)

        # setting
        self.handle_NA = True
        self.scale = True

        # add more measures
        self.n_jobs = 8
        self.objective = "reg:linear"
        self.testMeasure = ["score", "MSE"]

    def __str__(self):
        return "XGBoost Regression Object"

    def get_model_description(self, model):
        desc = super().get_model_description(model)
        return desc

    def set_x_raw_variables(self):

        """
        This function returns the columns to use in training data from DataHelper.
        Please note it returns full predictor column names including the new variables, after 2015.
        This is NOT actual variable names used in the model, due to the additional need of
        one hot encoding.
        """

        file = self.get_filename(self.filename_var)

        with open(file, 'r') as f:
            variables = json.load(f)
        return variables["interim_cols"]

    def fit_model(self, df, *args, **kwargs):

        """
        This function fits the XGBoost on a given training data set. Note that many varaibles
        need to be passed as an arguments. The number of jobs (CPU) and objective should be set as
        member variables. It returns the fitted model object.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable].values

        # fit XGBoost
        model = xgb.XGBRegressor(n_estimators=kwargs["n_estimators"],
                                 learning_rate=kwargs["learning_rate"],
                                 max_depth=kwargs["max_depth"],
                                 min_child_weight=kwargs["min_child_weight"],
                                 gamma=kwargs["gamma"],
                                 subsample=kwargs["subsample"],
                                 colsample_bytree=kwargs["colsample_bytree"],
                                 reg_alpha=kwargs["reg_alpha"],
                                 objective=self.objective,
                                 n_jobs=self.n_jobs,
                                 missing=None)
        model.fit(x_data, y_data)

        return model

    def fit_n_estimator(self, df, fold, stopping, **kwargs):

        """
        This function use xgboost cv to fit the n_estimators. This is convinient since
        We can set the early stopping rule for a quick calibration.
        """

        # this makes the data independence to stored x-variables list
        x_data = df.drop(columns=[self.y_variable])
        y_data = df[self.y_variable].values

        # d-matrix
        d_train = xgb.DMatrix(x_data, label=y_data)

        # Cross Validation
        kwargs["objective"] = self.objective
        results = xgb.cv(kwargs, d_train, num_boost_round=kwargs["n_estimators"], nfold=fold,
                         metrics='rmse', early_stopping_rounds=stopping)

        print("stop at {} test-mean-rmse: {}, test-std-rmse: {}".format(
            results.shape[0], round(results.iloc[-1, 2], 4), round(results.iloc[-1, 3], 4)))

        return results.shape[0]

