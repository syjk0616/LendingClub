import investhelper
import lendingclub as lc
import warnings
import pandas as pd
import os
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

os.chdir("/home/jacob/Project/LendingClub")

# instantiate config
config = lc.ConfigData("config_data.ini")

# instantiate investment helper
helper = investhelper.investment_helper(config)

# loans
loans = helper.DH.get_listed_loandata()

# prediction
pred = helper.model.predict_model(loans)

# run strategy
targets = helper.strat.invest_loans(loans, backtest=False)

# print Path
print(helper.DH.get_listed_loandata().shape)

# get lendingcub object
LC = helper.LC

# get my loans
loans = LC.get_my_loans()
cash = LC.get_cash()
summary = LC.get_account_summary()

print("available cash: {}".format(cash))
print("Account Summary: {}".format(summary))