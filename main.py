import investhelper
import lendingclub as lc
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# instantiate config
config = lc.ConfigData("config_data.ini")

# instantiate investment helper
helper = investhelper.investment_helper(config)

# invest
results = helper.invest()

