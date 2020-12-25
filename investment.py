import sys
import backtester
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class StrategyBase(ABC):

    def __init__(self, model):
        self.model = model
        self.FICO = 550

    def invest_loans(self, df, backtest=False):

        if not backtest:
            loans, pred = self.model.predict_model(df)
        else:
            # need original info for backtesting
            temp, pred = self.model.predict_model(df)
            loans = df.loc[temp.index, :]

        # apply investment logic
        investments = self.apply_strategy(loans, pred, backtest=backtest)

        return investments

    def apply_strategy(self, loans, pred, backtest=False):

        # select loans
        investments = self.select_loans(loans, pred)

        # find investment amounts
        investments = self.select_amounts(investments)

        # sort loans
        investments = self.sort_loans(investments)

        # filter loans
        investments = self.filter_loans(investments, backtest=backtest)

        return investments

    @abstractmethod
    def select_loans(self, loans, pred):

        # we need to append score columns here
        loans["score"] = np.array(pred)

        # any filter by logic goes to child
        return loans

    @abstractmethod
    def select_amounts(self, loans):

        # we need to append score columns in the function select_loans
        # we check this in here. If this is not satified, return error.
        if "score" not in loans.columns.values:
            print("Please append score columns at the function select_loans")
            sys.exit(1)

        # any amount selection logic goes to child
        return loans

    @abstractmethod
    def sort_loans(self, loans):

        # We need to sort the selected loans, so that we can grab the loans
        # from the top. This is for the case when we cannot invest on all of,
        # our selected loans due to the lack of the funds in the pocket.
        pass

    @abstractmethod
    def filter_loans(self, loans, backtest=False):

        # in backtest, FICO information is not available
        if not backtest:
            loans = loans[loans["fico_range_high"] >= self.FICO]

        # any explicit filter goes to child
        return loans

    @staticmethod
    @abstractmethod
    def description():

        # this is placeholder to add description string of each strategy
        pass


class StrategyClassSimple(StrategyBase):

    """
    This strategy is based on the simple cut method. Apply buckets on the test scores,
    then only invest on the loans falling to first few buckets. This is expected to generate
    stable income as long as we keep the "cut" as small.

    Actual value of cut should depend on the prediction model we use. We need to fine tune
    this empirically through the backtesting.
    e.g., xbgc - 0.02
    """

    def __init__(self, model):

        # call base init
        super().__init__(model)

        # we use fix amount for each loan
        self.invest_amount = 25

        # please note this cut are max score of the second bucket
        # please refer to "Strategy Investigation.ipynb".
        self.cut = 0.04

    def select_loans(self, loans, pred):

        # always call explicitly the parent node
        loans = super().select_loans(loans, pred)

        return loans[loans["score"] <= self.cut]

    def select_amounts(self, loans):

        # always call explicitly the parent node
        loans = super().select_amounts(loans)
        loans["invest_amount"] = self.invest_amount

        return loans

    def sort_loans(self, loans):

        # sort by score
        loans = loans.sort_values("score", ascending=True)

        return loans

    def filter_loans(self, loans, backtest=False):

        # always call explicitly the parent node
        loans = super().filter_loans(loans, backtest=backtest)

        return loans

    @staticmethod
    def description():

        return "This strategy chooses loans basaed on the simple score cut. In other words, it only " \
               "invests to the loans with the model scores lower than the cut. It further filters out the " \
               "selected loan by FICO score."


class StrategyClassComplex(StrategyBase):

    """
    This strategy is based on the expected return method. Find expected income (int - def) of
    loans, and then invest on the loans with the largest values. This is benchmark strategy to
    the simple one.

    We should find tune the "cut" and "grade" empirically through the backtesting.
    """

    def __init__(self, model):

        # call base init
        super().__init__(model)

        # we use fix amount for each loan
        self.invest_amount = 25

        # please note this cut are max score of the second bucket
        # please refer to "Strategy Investigation.ipynb".
        self.cut = 0.065
        self.grade = "C5"

    def select_loans(self, loans, pred):

        loans = super().select_loans(loans, pred)

        # get expected income
        int_rate = loans["int_rate"]/12
        int_rate = np.array([sum(x * np.array([1-i*1/36 for i in range(0, 36)])) for x in int_rate])
        expected = int_rate - loans["score"]

        return loans[expected >= self.cut]

    def select_amounts(self, loans):

        loans = super().select_amounts(loans)
        loans["invest_amount"] = self.invest_amount

        return loans

    def sort_loans(self, loans):

        # get expected income
        int_rate = loans["int_rate"]/12
        int_rate = np.array([sum(x * np.array([1-i*1/36 for i in range(0, 36)])) for x in int_rate])
        expected = int_rate - loans["score"]

        # sort by expected payoff
        index = np.argsort(expected)[::-1]
        loans = loans.iloc[index, :]

        return loans

    def filter_loans(self, loans, backtest=False):

        loans = super().filter_loans(loans, backtest=backtest)

        # we have additional filter by sub_grade
        loans = loans[loans["sub_grade"] <= self.grade]

        return loans

    @staticmethod
    def description():

        return "This strategy chooses loans basaed on the expected payoff. The expected payoff is calculated " \
               "by the interest rate minus the model score. If the calculated expected payoff is greater than " \
               "the predefined cut, then it selects a loan. It further filters out the selected loan by FICO " \
               "score and the loan grade."


class StrategyClassComplexII(StrategyClassComplex):

    """
    This strategy is improvement of StrategyClassComplex. It uses same logic with the parent,
    but differentiate the invested amount based on the expected payoff. Simple logistic function is leveraged
    to decide invest amounts.
    """

    def __init__(self, model):

        # call base init
        super().__init__(model)
        self.increment = 1
        self.amount_slope = 1
        self.amount_max = 1

    def select_amounts(self, loans):

        loans = super().select_amounts(loans)

        # sort by expected payoff
        int_rate = loans["int_rate"] / 12
        int_rate = np.array([sum(x * np.array([1-i*1/36 for i in range(0, 36)])) for x in int_rate])
        expected = int_rate - loans["score"]

        # additional loan amounts to add
        adder = self.amounts_func(expected)
        loans["invest_amount"] = loans["invest_amount"] + adder

        return loans

    def amounts_func(self, expected):

        """
        This is simle logistic function deciding additional amount to invest. p1 and p2 are needed to be tuned
        for each strategy. p1 controls the slope of the function, while p2 controls maximum additional amounts.
        """

        p1 = self.amount_slope
        p2 = self.amount_max
        cut = self.cut
        y = (1/(1+np.exp(-p1*(expected-cut))) - 0.5) * p2
        return np.round(y)

    @staticmethod
    def description():

        return "This strategy chooses loans basaed on the expected payoff. The expected payoff is calculated " \
               "by the interest rate minus the model score. If the calculated expected payoff is greater than " \
               "the predefined cut, then it selects a loan. It also increase the invest amount for each loan " \
               "based on the payoff. It further filters out the selected loan by FICO score and the loan grade."


class StrategyRegSimple(StrategyBase):

    """
    This strategy is based on the simple cut method. Cut loans by expected return.
    This is expected to generate aggressive income.

    Actual value of cut should depend on the prediction model we use. We need to fine tune
    this empirically through the backtesting.
    """

    def __init__(self, model):

        # call base init
        super().__init__(model)

        # we use fix amount for each loan
        self.invest_amount = 25

        # please note this cut are max score of the second bucket
        # please refer to "Strategy Investigation.ipynb".
        self.cut = 0.70

    def select_loans(self, loans, pred):

        # always call explicitly the parent node
        loans = super().select_loans(loans, pred)

        return loans[loans["score"] >= self.cut]

    def select_amounts(self, loans):

        # always call explicitly the parent node
        loans = super().select_amounts(loans)
        loans["invest_amount"] = self.invest_amount

        return loans

    def sort_loans(self, loans):

        # sort by score - should be descending
        loans = loans.sort_values("score", ascending=False)

        return loans

    def filter_loans(self, loans, backtest=False):

        # always call explicitly the parent node
        loans = super().filter_loans(loans, backtest=backtest)

        return loans

    @staticmethod
    def description():

        return "This strategy chooses loans basaed on the simple epected return cut. In other words, it only " \
               "invests to the loans with the model scores higher than the cut. It further filters out the " \
               "selected loan by FICO score."


class StrategyRegWeight(StrategyRegSimple):

    """
    This strategy is improvement of StrategyRegSimple. It uses same logic with the parent,
    but differentiate the invested amount based on the expected return.
    """

    def __init__(self, model):

        # call base init
        super().__init__(model)
        self.increment = 1

    def select_amounts(self, loans):

        loans = super().select_amounts(loans)

        # Assing increment loan amount
        score_map = np.array([0.05, 0.1, 0.15, 0.2]) + self.cut
        adder = np.array([sum(x>score_map) for x in loans["score"]])
        adder *= self.increment
        loans["invest_amount"] = loans["invest_amount"] + adder

        return loans

    @staticmethod
    def description():

        return "This strategy chooses loans basaed on the expected payoff. It also increase the " \
               "invest amount for each loan based on the payoff. It further filters out the selected " \
               "loan by FICO score and the loan grade."


# noinspection PyUnboundLocalVariable
class StrategyOptimizer:

    def __init__(self, DH, model, strat):

        self.DataHelper = DH
        self.model = model
        self.strat = strat

    def opt_cut_conserv(self, cuts, seed, plot=True):

        profits = []
        for cut in cuts:

            # set cut of strateg
            self.strat.cut = cut

            # set initial investment of Backtester
            Backtester = backtester.Backtester(self.DataHelper, self.model, self.strat, 1)
            Backtester.initialInvest = seed

            # run backtest
            _, r, _ = Backtester.backtest_OOT(returnAll=False)

            # append return
            profits.append(r)

        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(cuts, profits)
            plt.show()

        return profits

    def opt_seed_conserv(self, cuts, seeds, plot=True):

        results = []
        plt.figure(figsize=(10, 8))

        for seed in seeds:

            # call opt_cut_conserv
            profits = self.opt_cut_conserv(cuts, seed, plot=False)

            # plot
            plt.plot(cuts, profits)

            # append maximum
            ind = np.argmax(profits)
            results.append({"initialInvest": seed, "cut": cuts[ind], "return": profits[ind]})

        if plot:
            names = ["m:{}".format(x) for x in seeds]
            plt.legend(names, loc='upper left')
            plt.show()

        return results

    def opt_cut_aggr(self, cuts, grades, seed, plot=True):

        heatmap = np.zeros([len(cuts), len(grades)])

        for i, cut in enumerate(cuts):
            for j, grade in enumerate(grades):

                # set cut of strateg
                self.strat.cut = cut
                self.strat.grade = grade

                # set initial investment of Backtester
                Backtester = backtester.Backtester(self.DataHelper, self.model, self.strat, 1)
                Backtester.initialInvest = seed

                # append return
                _, r, _ = Backtester.backtest_OOT(returnAll=False)
                heatmap[i, j] = r

        # change it to df
        heatmap_df = pd.DataFrame(heatmap, index=cuts, columns=grades)

        if plot:
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(heatmap_df, cmap="YlGnBu")
            ax.set_title("aggressive strategy return with init invest {}".format(seed))
            plt.show()

        return heatmap_df

    def opt_seed_aggr(self, cuts, grades, seeds, plot=True):

        results = []
        for seed in seeds:
            print(seed)
            profits = self.opt_cut_aggr(cuts, grades, seed, plot=plot)

            # append maximum
            ind = np.unravel_index(np.argmax(profits.values, axis=None), profits.shape)
            grade = profits.columns[ind[1]]
            cut = profits.index[ind[0]]
            results.append({"return": profits.loc[cut, grade], "initialInvest": seed, "grade": grade, "cut": cut})

        return results

    def opt_amountfunc_aggr(self, slopes, maxs, cut, grade, seed, plot=False):

        # set cut of strateg
        self.strat.cut = cut
        self.strat.grade = grade

        heatmap = np.zeros([len(slopes), len(maxs)])

        for i, s in enumerate(slopes):
            for j, m in enumerate(maxs):

                # set cut of strateg
                self.strat.amount_slope = s
                self.strat.amount_max = m

                # set initial investment of Backtester
                Backtester = backtester.Backtester(self.DataHelper, self.model, self.strat, 1)
                Backtester.initialInvest = seed

                # append return
                _, r, _ = Backtester.backtest_OOT(returnAll=False)
                heatmap[i, j] = r

        # change it to df
        heatmap_df = pd.DataFrame(heatmap, index=slopes, columns=maxs)

        if plot:
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(heatmap_df, cmap="YlGnBu")
            ax.set_title("aggressive strategy return with init invest {}".format(seed))
            ax.set_ylabel('p1 - slope')
            ax.set_xlabel('p2 - max')
            plt.show()

        return heatmap_df

