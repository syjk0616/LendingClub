import Data_Helper
import investment
import lendingclub
import prediction


class investment_helper:

    """
    This is an object having all investment process.
    1. it instaniates every object needed for investment.
    2. the main function "invest" get listed loans, select, and then invest

    All relevant parameters/information are coming from config object.
    """

    def __init__(self, config):

        self.LC = None
        self.DH = None
        self.model = None
        self.strat = None

        self.config = config
        self.initiate_objects()

    def initiate_objects(self):

        """
        This function instantiates, DataHelper, LendingClub, Prediction, and Investment
        Every relevant infomation is stored in Config object
        """

        # initialize lendingclup api object
        self.LC = self.get_lendingclub_object()

        # initialize DataHelper
        self.DH = self.get_datahelper_object()

        # instantiate model
        self.model = self.get_model_object()

        # instantiate Strategy
        self.strat = self.get_strat_object()

    def get_lendingclub_object(self):

        LC = lendingclub.LendingClub(self.config)
        LC.set_logger()
        LC.set_portfolio_id()

        return LC

    def get_datahelper_object(self):

        # we can use dummy value, doesn't matter for actual investing
        periodStart = ("Q1", "2014")
        periodEnd = ("Q4", "2016")
        transformer = Data_Helper.Transformer_full()
        DH = Data_Helper.DataHelper(periodStart, periodEnd, transformer, self.LC)

        return DH

    def get_model_object(self):

        modelType = self.config.modelType
        modelFile = self.config.modelFile

        # call model
        if modelType == "LINEAR":
            model = prediction.ModelRegression(modelFile)
        elif modelType == "XGBR":
            model = prediction.ModelXGBRegression(modelFile)
        elif modelType == "RFC":
            model = prediction.ModelRandomForest(modelFile)
        elif modelType == "XGBC":
            model = prediction.ModelXGBClassfication(modelFile)
        elif modelType == "LOGISTIC":
            model = prediction.ModelLogistic(modelFile)
        else:
            # default to logistic
            model = prediction.ModelLogistic(modelFile)

        # set model
        model.set_model_from_file()

        return model

    def get_strat_object(self):

        stratType = self.config.stratType

        # call strategy
        if stratType == "Reg_Simple":
            strat = investment.StrategyRegSimple(self.model)
        elif stratType == "Reg_Weight":
            strat = investment.StrategyRegWeight(self.model)
        elif stratType == "Class_Complex":
            strat = investment.StrategyClassComplex(self.model)
        elif stratType == "Class_Simple":
            strat = investment.StrategyClassSimple(self.model)
        else:
            # default to simple classification strategy
            strat = investment.StrategyClassSimple(self.model)

        # setting
        strat.cut = self.config.stratCut
        strat.grade = self.config.stratGrade
        strat.FICO = self.config.stratFICO
        strat.invest_amount = self.config.stratAmount

        return strat

    def invest(self):

        """
        Main function for actual investment
        """

        # get listed loan
        loans = self.DH.get_listed_loandata()

        # choose loans to invest
        targets = self.strat.invest_loans(loans, backtest=False)

        # invest
        results = self.LC.post_orders(targets)

        return results


