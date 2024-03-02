# -*- coding: utf-8 -*-
"""
IBAPI - Request intraday historical data for each traded stock at entry date and exit date

@author: RazorForce
"""

""" IB API INCLUDES  """
import sys
#sys.path.insert(0, 'C:\\Users\\moham\\.conda\\envs\\Py310\\Lib\\site-packages\\ibapi')
sys.path.append("C:\\X\\Workspaces\\ALGO\\software\\src_app")

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
""" LIBRARY INCLUDES """
from datetime import datetime
import pandas as pd
import threading
import time
from dateutil import parser
""" ALGO INCLUDES    """
import os
import json
from file_handling import *

class TradingApp(EWrapper, EClient):
    def __init__(self,event):
        self.event = event
        EClient.__init__(self,self)
        self.histdata_dict = { } #empty dictionary NO KEYS NO VALUES

    def error(self, reqId, errorCode:int, errorString:str, advancedOrderRejectJson = ""):
        print(f"Error RequestID {reqId} / ErrCode {errorCode} /  ErrStr{errorString}")
        #pass

    def connectAck(self):
        print("ConnState = CONNET ACK")

    def nextValidId(self, orderId):
        """This function Receives next valid order id
        triggered every time connection to TWS is established.
        triggered every time reqIds() is called """
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        print(f"Next Valid Id {orderId}")

    def historicalData(self, reqId, bar):
        """ triggered by reqHistoricalData """
        #slice bar.date to disregard timezone information
        bar.date = datetime.strptime(bar.date[:16], "%Y%m%d %H:%M:%S")

        if reqId in self.histdata_dict:
            self.histdata_dict[reqId].append(
                                              {"Date": bar.date, "Open": bar.open, "High": bar.high,
                                                "Low": bar.low, "Close": bar.close, "Volume": bar.volume}
                                              )

        if reqId not in self.histdata_dict:
            # add new key to dict then make a list
            self.histdata_dict[reqId] = [
                                          {"Date": bar.date, "Open": bar.open, "High": bar.high,
                                          "Low": bar.low, "Close": bar.close, "Volume": bar.volume}
                                        ]

        #print("Historicaldata_reqID {}: Date={}, Open={}, High={}, Low={}, Close={}, Volume={}"
        #      .format(reqId, bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume))

    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print(f"historicalDataEnd. ReqId = {reqId} ,start from =  {start}, end at = {end}")
        self.event.set()

""" ******************* GLOBAL VARIABLES ******************* """

""" ******************************************************** """




class Intra_Histdata_Request():

    def __init__(self, event,app_obj):
        self.event = event
        self.app = app_obj

    def websocket_con(self):
        self. app.run()

    def ContractDetails(self, symbol, conId = 0, secType = "STK", currency = "USD", exchange ="SMART"):
        contract = Contract() #create contract object
        contract.symbol = symbol
        contract.secType = secType
        contract.currency = currency
        contract.exchange = exchange
        contract.conId = conId
        return contract

    def get_HistoricalData(self, reqNum, contract, endDate, duration, candle_size, use_RTH=1):
        #Eclient function to request Historical Data
        self.app.reqHistoricalData(reqId=reqNum,
                               contract = contract,
                               endDateTime = endDate,
                               durationStr = duration,
                               barSizeSetting = candle_size,
                               whatToShow = 'TRADES', #First open/Last close/Highest High/Lowest Low prices
                               useRTH = use_RTH, #Regular Trading Hours Only
                               formatDate = 1, #to obtain the bars' time as YYYYMMDD hh:mm:ss
                               keepUpToDate = 0,#do not received continuous updates
                               chartOptions = [])

    def store_DataFrame(self, req_map, date_map, reqId, hist_dict,tickers):

        #even number are entry dates
        if reqId % 2 == 0:
            new_key = req_map[reqId]+"-"+date_map[reqId]+"-entry"
            hist_dict[new_key]=pd.DataFrame(self.app.histdata_dict[reqId])
        # odd numbers are exit dates
        else:
            new_key= req_map[reqId]+"-"+date_map[reqId]+"-exit"
            hist_dict[new_key]=pd.DataFrame(self.app.histdata_dict[reqId])

        # set index
        hist_dict[new_key] = hist_dict[new_key].set_index('Date')
        #store csv file
        store_file(f"{new_key}_Intraday_Bars.csv", hist_dict[new_key], "csv", "Historical\\Intraday")

    def request(self, tickers):

        #*****************************************************************************


        self.app.connect("127.0.0.1", 7496, clientId=2)
        """    ************* Start connection_thread *************    """
        # starting a separate daemon thread to execute the websocket connection
        con_thread = threading.Thread(target=self.websocket_con, daemon=True)
        con_thread.start()
        time.sleep(1) # some latency added to ensure that the connection is established
        """    ************* Resume main Thead  *****************    """

        cwd = "C:\\X\\Workspaces\\ALGO\\Data"

        file_path = os.path.join(cwd, "Scanner")

        reqId = 0
        req_map = {}
        date_map = {}
        conId = read_file("contractId.json", "Scanner")
        # Convert JSON string to dictionary
        conId = json.loads(conId)

        #for ticker in tickers:
        for sublist in tickers:
            ticker = sublist[0]
            date = sublist[1]

            self.event.clear()

            self.get_HistoricalData(reqId,self.ContractDetails(ticker, conId[ticker]),
                               date+' 19:59:00 US/Eastern','1 D','1 min')
            self.event.wait()

            req_map[reqId] = ticker
            date_map[reqId] = date

            reqId += 1
            #time.sleep(1)

        hist_dict = {}

        for i in range(0,reqId):
            self.store_DataFrame(req_map, date_map, i, hist_dict,tickers, )
        time.sleep(1)


        #print("khalas")
        self.app.disconnect()