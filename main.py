import investhelper
import lendingclub as lc
import datetime
import warnings
import pandas as pd
import os
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

print("{} : running main.py".format(datetime.datetime.now()))

# set dir by hardcode to run in crontab
os.chdir("/home/jacob/Project/LendingClub/")

# instantiate config
config = lc.ConfigData("config_data.ini")

# instantiate investment helper
helper = investhelper.investment_helper(config)

# invest
results = helper.invest()

