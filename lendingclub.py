import requests
import json
import numpy as np
import logging
import configparser
import datetime
import os


class ConfigData:

    def __init__(self, filename):

        self.configFileName = filename
        config = configparser.ConfigParser()
        config.optionform = str
        config.read(filename)

        # config data for lendingclub
        self.accountId = config.get("AccountData", "accountId")
        self.authKey = config.get("AccountData", "authKey")
        self.portfolioName = config.get("AccountData", "portfolioName")
        self.portfolioDesc = config.get("AccountData", "portfolioDesc")
        self.logFileName = config.get("Logging", "FileName")

        # config data for model
        self.modelType = config.get("Model", "modelType")
        self.modelFile = config.get("Model", "modelFile")

        # config data for strategy
        self.stratType = config.get("Strategy", "stratType")
        self.stratCut = float(config.get("Strategy", "stratCut"))
        self.stratGrade = config.get("Strategy", "stratGrade")
        self.stratFICO = int(config.get("Strategy", "stratFICO"))
        self.stratAmount = float(config.get("Strategy", "stratAmount"))


class LendingClub:

    """
    This is the main object interacting with lending club account through REST api.
    Please make sure call set_portfolio_id before executing the order.
    This authentication key and account id can be found in the config file (config_data.ini)
    """

    version = 'v1'
    baseUrl = 'https://api.lendingclub.com/api/investor/'

    def __init__(self, config):

        # Basic
        self.config = config
        self.header = {"Authorization": self.config.authKey, 'Content-Type': "application/json"}
        self.portfolioId = np.nan

        # Accounts
        self.acctSummaryURL = self.baseUrl + self.version + '/accounts/' + str(self.config.accountId) + '/summary'
        self.portfoliosURL = self.baseUrl + self.version + '/accounts/' + str(self.config.accountId) + '/portfolios'
        self.acctLoansURL = self.baseUrl + self.version + '/accounts/' + str(self.config.accountId) + '/notes'

        # Listed Loans
        self.loanListURL = self.baseUrl + self.version + '/loans/listing'

        # orders
        self.ordersURL = self.baseUrl + self.version + '/accounts/' + str(self.config.accountId) + '/orders'

        # logging - we need to call logger separately
        self.logger = None
        self.time = None

    def set_logger(self):

        """
        If we want to log the process, then we should call this setter first
        """

        # instantiate logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # set file handler
        formatter = logging.Formatter('%(name)s:%(message)s')
        file = "{}/{}".format(os.getcwd(), self.config.logFileName)
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # set logger
        self.logger = logger

        # add log
        self.logger.info("{}: LendingClub object is constructed".format(datetime.datetime.now()))

    def get_account_summary(self):
        resp = requests.get(self.acctSummaryURL, headers=self.header)
        resp.raise_for_status()
        return resp.json()

    def get_cash(self):
        summary = self.get_account_summary()
        return float(summary["availableCash"])

    def get_my_loans(self):
        resp = requests.get(self.acctLoansURL, headers=self.header)
        resp.raise_for_status()
        return resp.json()["myNotes"]

    def get_my_loan_ids(self):
        loans = self.get_my_loans()
        return [f["loanId"] for f in loans]

    def get_listed_loans(self):

        """
        This calls the list of listed loans to invest. However, don't call this directly.
        Rather call "get_listed_loandata" of Data Helper object which leverages this function.
        """

        resp = requests.get(self.loanListURL, headers=self.header, params={'showAll': 'true'})
        resp.raise_for_status()
        return resp.json()

    def set_portfolio_id(self):

        """
        It takes the portfolioName of config file, and finds corresponding portfolio id.
        If no portfolio id found, then it construct portfolio with a given name and a description.
        Make sure to call this setter always as soon as instantiating the lendingclub object.
        """

        ports = self.get_portfolio()
        if not len(ports):
            port_id = self.post_portfolio()
        elif self.config.portfolioName not in [p["portfolioName"] for p in ports]:
            port_id = self.post_portfolio()
        else:
            ind = [i for i, p in enumerate(ports) if p["portfolioName"] == self.config.portfolioName][0]
            port_id = ports[ind]["portfolioId"]

        self.portfolioId = port_id

    def get_portfolio(self):
        resp = requests.get(self.portfoliosURL, headers=self.header)
        resp.raise_for_status()
        result = resp.json()

        if len(result):
            return result["myPortfolios"]
        else:
            return result

    def post_portfolio(self):

        """
        This function constructs new portfolio.
        """

        payload = json.dumps({
            "actorId": self.config.accountId,
            "portfolioName": self.config.portfolioName,
            "portfolioDescription": self.config.portfolioDesc
        })

        resp = requests.post(self.portfoliosURL, headers=self.header, data=payload)
        result = resp.json()

        if 'errors' in result:
            if self.logger is not None:
                for error in result['errors']:
                    self.logger.error('Error during portfolio construction: ' + error['message'])
            return 0
        else:
            if self.logger is not None:
                msg = "Portfolio {}, {} is constructed".format(self.config.portfolioName, result["portfolioId"])
                self.logger.info(msg)
            return result["portfolioId"]

    def post_order(self, loan_id, amount):

        """
        This function post order by taking loanId and amount to invest.
        This is the key function of this object. It returns NaN when error happens, rather than
        halting whole process.
        """

        payload = json.dumps({'aid': self.config.accountId,
                              'orders': [{
                                  'loanId': loan_id,
                                  'requestedAmount': float(amount),
                                  'portfolioId': self.portfolioId
                              }]})

        try:
            resp = requests.post(self.ordersURL, headers=self.header, data=payload)
            result = resp.json()
        except Exception as e:
            if self.logger is not None:
                self.logger.error("Error during post order for loanId: {}, amount: {}".format(loan_id, amount))
            return np.nan
        else:
            if 'errors' in result:
                if self.logger is not None:
                    for error in result['errors']:
                        self.logger.error('Error during post order: ' + error['message'])

                # in case of error, return NaN, rather than halting whole process
                return np.nan
            else:
                confirmation = result['orderConfirmations'][0]

                if self.logger is not None:
                    order_msg = "OrderId {}: ${} invested in LoanId {}".format(
                        result['orderInstructId'], confirmation['investedAmount'], confirmation['loanId'])
                    self.logger.info(order_msg)

                # return confirmation
                return confirmation

    def post_orders(self, loans):

        """
        This function post orders for the loans selected by Stratgy object.
        The input should be the dataframe coming from Strategy Object, with id and invest_amount columns.
        """

        # no loans to invest
        if loans.shape[0] == 0 and self.logger is not None:
            self.logger.info("There is no loans to invest")
            return 0

        # select loans based on the budget
        budget = self.get_cash()
        amounts = loans["invest_amount"].cumsum()
        loans = loans[amounts <= budget]

        # post order
        if loans.shape[0] > 0:
            results = []
            for loanId, investAmount in zip(loans["id"], loans["invest_amount"]):
                if loanId not in self.get_my_loan_ids():
                    results.append(self.post_order(loanId, investAmount))
                else:
                    if self.logger is not None:
                        self.logger.info("LoanId {} is already in the portfolio".format(loanId))
            return results
        else:
            if self.logger is not None:
                self.logger.info("No budget left to invest ({})".format(budget))
            return 0

