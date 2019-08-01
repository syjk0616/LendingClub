import pandas as pd
import numpy as np
import datetime
import dateutil
from dateutil import relativedelta


# noinspection PyPep8Naming
class Backtester:

    def __init__(self, dataHelper, model, strategy, ntest):
        self.dataHelper = dataHelper
        self.model = model
        self.strategy = strategy
        self.ntest = ntest
        self.nfold = 5
        self.riskfree = 0.026
        self.bootstrap_sample = 0.7

        # variables for backtesting
        self.initialInvest = 10000

    def backtest_OOT(self, bootstrap=False, returnAll=False):

        # turn on returnAll - this is for analysis purpose
        returnAll *= (not bootstrap)

        # profits
        profits = []

        # get test dataset
        loans = self.dataHelper.test

        if bootstrap:
            indices = self.bootstrap(loans.shape[0])
            for index in indices:
                test = loans.iloc[index, :]
                profit = self.backtest(test)
                profits.append(profit)
        else:
            if returnAll:
                total, C, L, profit = self.backtest(loans, returnAll=returnAll)
                return total, C, L, profit
            else:
                profit = self.backtest(loans)

            profits.append(profit)

        return profits, self.avg_return(profits), self.sharp_ratio(profits)

    def backtest_InTime(self):

        # profits
        profits = []

        # get training dataset
        df = self.dataHelper.training

        # loop for ntest
        for n in range(self.ntest):

            # get CVs - we need to randomize this!!
            CVs = self.dataHelper.get_cross_validation_data(fold=self.nfold)

            # Backtest with k-fold CV data
            for i, CV in enumerate(CVs):

                # get tuned parameters
                params = self.model.model.get_params()

                # refit model on the CV
                training, test = self.model.get_data_for_model(df.loc[CV[0], :], df.loc[CV[1], :])
                model = self.model.fit_model(training, **params)

                # backtest for the left out
                profit = self.backtest(test, model=model)
                profits.append(profit)

        return self.sharp_ratio(profits)

    def backtest(self, loans, model=None, returnAll=False):

        # initialize
        total = pd.DataFrame()
        M = self.initialInvest
        I = 0

        month_start = min(loans["issue_d"])
        month_end = max(loans["issue_d"])

        # investing period
        r = relativedelta.relativedelta(month_end, month_start)
        J1 = r.months + r.years * 12
        J2 = J1 + 3 * 12

        # time span
        times = [month_start + relativedelta.relativedelta(months=t) for t in range(J2 + 1)]

        # total amount of loans invested
        L = np.zeros(J2 + 1)

        # cashflow generated from invested loans
        C = np.zeros(J2 + 1)

        # loop start
        j = 0
        selected = self.invest_loans(model, loans, times, j, M)
        if not len(selected):
            if returnAll:
                return selected, C, L, 0
            else:
                return 0
        L[j] = np.sum(selected["invest_amount"])
        C = self.update_cashflow(selected, C, j)
        M -= L[j]
        I += L[j]

        total = total.append(selected)

        for j in range(1, J1 + 1):
            M += C[j - 1]
            selected = self.invest_loans(model, loans, times, j, M)
            L[j] = np.sum(selected["invest_amount"])
            C = self.update_cashflow(selected, C, j)
            M -= L[j]
            I += L[j]

            total = total.append(selected)

        # rest of cashflow
        T = np.argmin(C[J1:]) - 1
        rf = self.riskfree / 12
        B = np.array([(1 + rf) ** t for t in range(0, T + 1)])[::-1]
        profit = sum(C[J1:J1 + T + 1] * B) + M * B[0]

        # annualize
        profit /= self.initialInvest
        profit = (profit-1) * 12 / (J1+T)

        if returnAll:
            return total, C, L, profit
        else:
            return profit

    def invest_loans(self, model, loans, times, j, M):

        # get strategy object
        strategy = self.strategy
        if model is None:
            model = self.model

        # reset the model
        strategy.model = model

        # find loans to invest for j month
        ind = loans["issue_d"].map(lambda x: x.month) == times[j].month
        loans = self.strategy.invest_loans(loans[ind], backtest=True)
        if not len(loans):
            return loans

        # cut by money
        spent = np.array([np.sum(loans["invest_amount"][0:x+1]) for x in range(0, loans.shape[0])])
        loans = loans[spent <= M]

        return loans

    @staticmethod
    def update_cashflow(loans, C, j):

        # get info
        rs = np.array(loans.apply(lambda x: relativedelta.relativedelta(x["last_pymnt_d"], x["issue_d"]), axis=1))
        ts = np.array([r.months + r.years * 12 for r in rs])
        ratio = np.array(loans["invest_amount"]/np.exp(loans["loan_amnt"]))

        # change it to mothly payment
        last = np.minimum(loans["total_pymnt"], loans["last_pymnt_amnt"]) * ratio
        recovery = np.array(loans["recoveries"]) * ratio
        monthly = np.array(loans["total_pymnt"] - loans["collection_recovery_fee"] -
                           loans["recoveries"] - loans["last_pymnt_amnt"])
        monthly = np.maximum(monthly, np.zeros(len(monthly))) * ratio / ts
        monthly[np.isnan(monthly)] = 0

        # sum up the cashflow to C
        for t, m, l, r in zip(ts, monthly, last, recovery):

            if t:
                add = np.zeros(t) + m
            else:
                add = np.array([m])

            # we add last payment and recovery at last
            add[-1] += (l + r)

            if t:
                C[(j + 1):(t + j + 1)] += add
            else:
                C[j] += add

        return C

    def bootstrap(self, n):

        # m = int(np.floor(n * 1/self.nfold))
        m = int(np.floor(n * self.bootstrap_sample))
        return [np.random.choice(n, m, replace=False) for x in range(self.ntest)]

    def sharp_ratio(self, profits):

        if len(profits) == 1:
            return profits[0]
        else:
            mean = np.average(profits)
            std = np.std(profits)
            return (mean-self.riskfree)/std

    @staticmethod
    def avg_return(profits):

        return np.average(profits)


