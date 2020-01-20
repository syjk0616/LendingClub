import numpy as np
import pandas as pd
import os
import json
import datetime
from sklearn.model_selection import KFold

class DataHelper:

    ''' DataHelper

    This is main class for data helping to,
    1. get training data
    2. get test data
    3. get CV folds for training data
    4. get summary statistics
    5. clean the investment dataset

    Please make sure that the actual investment dataset will be called from LendingClub class,
    but the cleaning will be done here (NEED TO ADD THE CLEANING FUNCTION)
    '''

    filename_Var = "variable_selected.txt"
    filename_Data_Pre = "LoanStats"
    filename_map_var = "map_TrainLC.txt"

    def __init__(self, periodStart, periodEnd, transformer, lendingclub):

        self.startQuarter = periodStart[0]
        self.startYear = int(periodStart[1])
        self.endQuarter = periodEnd[0]
        self.endYear = int(periodEnd[1])
        self.term = " 36 months"
        self.bad_loans = ["Charged Off", "Default", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]
        self.states = ["CA", "TX", "NY", "FL", "IL", "NJ", "GA", "OH", "PA"]

        # This is separate object
        self.transformer = transformer
        self.lendingclub = lendingclub

        # To be set
        self.training = np.nan
        self.test = np.nan

    def set_training_dataset(self, verbose=False):

        '''
        This is the main function constructing training dataset. It reads files by looping,
        includes the data in-scope, and then filtering out the columns. It also calls final
        transformation function here. The resulted dataset is set to training and retuned.
        '''

        # Get all files
        file_names = self.get_training_files()

        # Loop through the files
        dataset = self.get_combined_data(file_names, forTrain=True, includeCurrent=False, verbose=verbose)

        # reset index
        dataset = dataset.reset_index(drop=True)

        # set training
        self.training = dataset

    def set_test_dataset(self, file_names, includeCurrent=False, verbose=False):

        '''
        This is the main function constructing test dataset. It reads files specified as input
        data. Exact same treatments with training dataset are applied.
        '''

        # Loop through the files
        dataset = self.get_combined_data(file_names, forTrain=False, includeCurrent=includeCurrent, verbose=verbose)

        # reset index
        dataset = dataset.reset_index(drop=True)

        # set test
        self.test = dataset

    def get_combined_data(self, file_names, forTrain=False, includeCurrent=False, verbose=False):

        '''
        This function loops over the filenames provided and appends if available
        '''

        dataset = pd.DataFrame()

        # loop through files
        for name in file_names:

            if verbose: print("Reading {}...".format(name))
            df = self.get_rawdata_from_file(name, forTrain=forTrain, includeCurrent=includeCurrent, verbose=verbose)

            if df.shape == (0, 0):
                if verbose: print("Excluded: Out of period")
                continue

            if dataset.shape == (0, 0):
                dataset = df
            else:
                dataset = dataset.append(df)

        return dataset

    def get_rawdata_from_file(self, file_name, forTrain=False, includeCurrent=False, verbose=False):

        '''
        This function reads data from single filename, basic filtering by term, issue_d, loan_status.
        And then applies basic cleaning and transformation by calling clean_rawdata function.
        '''

        # Read file
        file = self.get_filename(file_name)
        df = pd.read_csv(file, sep=",", skiprows=1, low_memory=False)

        # select 36m only
        df = df[df["term"] == self.term]

        # change issue_d to datetime type
        df["issue_d"] = [datetime.datetime.strptime(d, "%b-%Y") for d in df["issue_d"] if type(d) is str]

        # this is only for training
        if forTrain:
            min_d = np.min(df["issue_d"])
            max_d = np.max(df["issue_d"])
            min_Q = self.get_quarter(min_d)
            max_Q = self.get_quarter(max_d)
            min_Y = min_d.year
            max_Y = max_d.year

            # skip if out of scope
            if min_Y < self.startYear:
                return pd.DataFrame()
            elif min_Y == self.startYear and min_Q < self.startQuarter:
                return pd.DataFrame()
            elif max_Y > self.endYear:
                return pd.DataFrame()
            elif max_Y == self.endYear and max_Q > self.endQuarter:
                return pd.DataFrame()

        # remove live loans from the dataset
        if not includeCurrent:
            df = df[df["loan_status"] != "Current"]

        # add quarter and year
        df["issue_Q"] = df["issue_d"].map(self.get_quarter)
        df["issue_y"] = df["issue_d"].map(lambda x: x.year)

        # True/False if status is bad_loans - NOTE We probably don't need np.where here?
        df["bad_loan"] = np.where(df["loan_status"].map(lambda x: x in self.bad_loans), True, False)

        # PLACEHOLDER - Add RETURN - then add variable name in the file
        df["return"] = (df["total_pymnt"]-df["collection_recovery_fee"])/df["loan_amnt"]
        df["log_return"] = np.log(1+df["return"])

        # cleaning data
        df = self.clean_rawdata(df)

        # Filter out other columns
        cols, info_cols = self.get_variables()
        cols = cols + info_cols
        df = df[cols]

        return df

    def clean_rawdata(self, df):

        '''
        This function cleans and transforms x-vaariables. This should be uniformly applied to
        training, test, and listed loan dataset.
        '''

        # int_rate - change it to ratio from string
        df["int_rate"] = df["int_rate"].map(lambda x: np.float32(x.split("%")[0]) / 100)
        df["int_rate"] = list(map(float, df["int_rate"]))

        # revol_util - change it to ratio from string
        df.loc[df["revol_util"].notna(),"revol_util"] = \
            df.loc[df["revol_util"].notna(),"revol_util"].map(lambda x: np.float32(x.split("%")[0]) / 100)
        df["revol_util"] = list(map(float, df["revol_util"]))

        # home_ownership - delete ANY
        df = df[df["home_ownership"] != "ANY"]  # delete any

        # purpose - combine to other category
        df.loc[df["purpose"] == "educational", "purpose"] = "other"
        df.loc[df["purpose"] == "renewable_energy", "purpose"] = "other"
        df.loc[df["purpose"] == "wedding", "purpose"] = "other"

        # earliest_cr_line - change to days until issue_d and log trans
        df["earliest_cr_line"] = df["earliest_cr_line"].map(lambda x: datetime.datetime.strptime(x,"%b-%Y"))
        delta = df["issue_d"] - df["earliest_cr_line"]
        delta = delta.map(lambda x: x.days)
        df["earliest_cr_line"] = delta

        # emp_length
        df.loc[df["emp_length"].notna(),"emp_length"] = \
            df.loc[df["emp_length"].notna(),"emp_length"].map(self.emp_filter)

        df["emp_length"] = pd.to_numeric(df["emp_length"])

        # remove dti -1
        df = df[df["dti"] >= 0]

        # Application Type
        df["application_type"] = np.where(df["application_type"] == "Individual", 1, 0)

        # decreases states
        df["addr_state"] = df["addr_state"].map(lambda x: "Other" if x not in self.states else x)

        # change last_pymnt_d to dateformat
        # we need do this carefully, since this is always NaN if the loan_status is "charged-off"
        # as a conservative approach, we assume last_pymnt_d is last day of the loan maturity
        if self.term == " 36 months":
            year_to_add = 3
        else:
            year_to_add = 6

        ind = df["last_pymnt_d"].isnull()
        temp = df.loc[ind, "issue_d"] + datetime.timedelta(year_to_add * 365)
        df.loc[ind, "last_pymnt_d"] = [x.strftime("%b-%Y") for x in temp]
        df["last_pymnt_d"] = [datetime.datetime.strptime(d, "%b-%Y") for d in df["last_pymnt_d"] if type(d) is str]

        # Transform data
        # we take the transformer and apply transformation here
        df = self.transformer.transform(df)

        return df

    def get_listed_loandata(self):

        '''
        This function returns availble list of loans to invest from lendingclub api.
        It leverages "get_listed_loans" function in lendingclub object, and additionally
        applies cleaning and transformation.
        '''

        # get listed loans from lendingclub api
        df = self.lendingclub.get_listed_loans()
        df = pd.DataFrame(df["loans"])
        df = df[df["term"]==36]

        # get variable mapping
        file = self.get_filename(self.filename_map_var)

        with open(file, 'r') as f:
            mapper = json.load(f)

        filter_cols = list(mapper.keys())
        new_cols = list(mapper.values())

        # filter by columns
        df = df[filter_cols]

        # change column names to match training dataset
        df.columns = new_cols

        # clean/transform
        # int_rate - change it to ratio from string
        df["int_rate"] = df["int_rate"] / 100

        # revol_util - change it to ratio from string
        df["revol_util"] = df["revol_util"] / 100

        # home_ownership - delete ANY
        df = df[df["home_ownership"] != "ANY"]  # delete any

        # purpose - combine to other category
        df.loc[df["purpose"] == "educational", "purpose"] = "other"
        df.loc[df["purpose"] == "renewable_energy", "purpose"] = "other"
        df.loc[df["purpose"] == "wedding", "purpose"] = "other"

        # earliest_cr_line - change to days until issue_d and log trans
        df["earliest_cr_line"] = df["earliest_cr_line"].map(lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d"))
        delta = datetime.datetime.now() - df["earliest_cr_line"]
        delta = delta.map(lambda x: x.days)
        df["earliest_cr_line"] = delta

        # emp_length
        emplen = np.round(df.loc[df["emp_length"].notna(), "emp_length"] / 12)
        emplen[emplen >= 10] = 10
        emplen[emplen <= 0] = 0
        df.loc[df["emp_length"].notna(), "emp_length"] = emplen
        df["emp_length"] = pd.to_numeric(df["emp_length"])

        # remove dti -1
        df = df[df["dti"] >= 0]

        # Application Type
        df["application_type"] = np.where(df["application_type"] == "INDIVIDUAL", 1, 0)

        # decreases states
        df["addr_state"] = df["addr_state"].map(lambda x: "Other" if x not in self.states else x)

        # Transform data
        # we take the transformer and apply transformation here
        df = self.transformer.transform(df)

        return df

    def get_cross_validation_data(self, fold=5):

        # get training
        df = self.training.copy()

        # prepare empty array
        stores = [[np.array([], dtype="int"), np.array([], dtype="int")] for x in range(fold)]

        # get time marks
        time_marks = df.apply(lambda x: str(x["issue_y"]) + x["issue_Q"], axis=1)
        marks = time_marks.unique()

        # instantiate KFold
        kf = KFold(n_splits=fold)

        # CV by each time mark
        for mark in marks:
            tempdata = df.index[time_marks == mark].tolist()
            splits = kf.split(tempdata)
            splits = list(splits)
            for i, item in enumerate(splits):
                tmp1 = [tempdata[x] for x in item[0]]
                tmp2 = [tempdata[x] for x in item[1]]
                stores[i][0] = np.concatenate([stores[i][0], tmp1])
                stores[i][1] = np.concatenate([stores[i][1], tmp2])

        return stores

    def get_sumary_rawdata_loan_status(self, file_name):

        '''
        This function returns loan status of csv file without any cleaning.
        This is for a quick check whether to include or not a new csv file.
        '''

        file = self.get_filename(file_name)
        df = pd.read_csv(file, sep=",", skiprows=1, low_memory=False)
        df = df[df["term"] == self.term]

        # loan_status_summary
        group = df.groupby(["loan_status"]).agg("count")["loan_amnt"]
        group = pd.DataFrame({"count": group, "ratio": group / group.sum()})

        return group

    @staticmethod
    def get_summary_data(df):

        '''
        This function returns summary of the dataset. This can be a good on-going monitoring
        of the model. If something changed vastly, we need to revisit the investment strategy.
        '''

        df = df[["bad_loan", "sub_grade", "issue_Q", "issue_y"]]
        df.loc[:, "issue"] = df.apply(lambda x: "".join([str(x["issue_y"]), x["issue_Q"]]), axis=1)

        issues = list(df["issue"].unique())
        for issue in issues:
            df.loc[:, issue] = df["issue"] == issue

        cols = ["bad_loan"] + issues
        summary = df.loc[:,cols].groupby(["bad_loan"]).agg(np.sum)

        for col in summary.columns.values:
            summary.loc[:,col] = summary[col] / np.sum(summary[col])

        return summary

    def get_variables(self):
        '''Get full list of selected variables (columns)'''
        filename = self.get_filename(self.filename_Var)
        with open(filename, 'r') as f:
            variables = json.load(f)

        return variables["interim_cols"], variables["info_cols"]

    def get_training_files(self):
        '''Get full list of filenames candidate for training dataset'''
        path = self.get_filepath()
        files = [f for f in os.listdir(path) if f.endswith(".csv") and f.startswith(self.filename_Data_Pre)]
        files.sort()
        return files

    @staticmethod
    def get_quarter(x):
        if x.month <= 3:
            return "Q1"
        elif x.month <= 6:
            return "Q2"
        elif x.month <= 9:
            return "Q3"
        else:
            return "Q4"

    @staticmethod
    def emp_filter(x):
        '''Treating special format of emp_length variable'''
        if "<" in x:
            x = int(x[2])-1
        elif "+" in x:
            x = x.split(" ")[0].split("+")[0]
            x = int(x)
        else:
            x = int(x.split(" ")[0])

        return float(x)

    @staticmethod
    def get_filepath():
        return os.getcwd() + "/Data/"

    @staticmethod
    def get_filename(filename):
        filepath = os.getcwd() + "/Data/"
        return filepath + filename


class Transformer:

    ''' Transformer

    This is base class handling advanced transformation of the data. It contains every logic,
    but actual transformation method mapping for variables are stored in txt file. Each chile
    clss specifies the txt file name.
    '''

    def get_transform_map(self):

        '''Load JSON file containing mapping. Noe the filename_map is abstract variable'''

        file = self.get_filename(self.filename_Map)

        with open(file, 'r') as f:
            results = json.load(f)

        return results["transform"], results["cut"], results["fill"]

    def transform(self, dataset):

        '''Apply transformation - transform, cut, and fill'''

        trans_map, cut_map, fill_map = self.get_transform_map()
        dataset = self.variable_trans(dataset, trans_map)
        dataset = self.variable_Cut(dataset, cut_map)
        dataset = self.variable_fill(dataset, fill_map)
        return dataset

    @staticmethod
    def variable_trans(df, trans_map):
        for col, method in trans_map.items():
            if method == "log":
                df[col] = np.log(df[col])
            elif method == "1+log":
                df[col] = np.log(1 + df[col])
            elif method == "sqrt":
                df[col] = np.sqrt(df[col])

        return df

    @staticmethod
    def variable_Cut(df, cut_map):
        for col, cut in cut_map.items():
            if not np.isnan(cut):
                df.loc[df[col] >= cut,col] = cut
        return df

    @staticmethod
    def variable_fill(df, fill_map):
        for col, fill in fill_map.items():
            if not np.isnan(fill):
                df[col].fillna(fill, inplace=True)
        return df

    @staticmethod
    def get_filename(filename):
        filepath = os.getcwd() + "/Data/"
        return filepath + filename


class Transformer_full(Transformer):

    '''This is the child class applying full transformation'''

    def __init__(self):
        self.filename_Map = "transform_map.txt"

