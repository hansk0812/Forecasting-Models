import os
import yfinance as yf

"""
AAPL Apple Inc. MCD McDonald’s Corporation
ABT Abbott Laboratories MSFT Microsoft Corporation
AEM Agnico Eagle Mines Limited ORCL Oracle Corporation
AFG American Financial Group, Inc. WWD Woodward, Inc.
APA Apache Corporation T AT&T Inc.
CAT Caterpillar Inc. UTX --> RTX United Technologies Corporation
"""
stocks = ["AAPL", "MCD", "ABT", "MSFT", "AEM", "ORCL", 
          "AFG", "WWD", "APA", "T", "CAT", "RTX"]

for stock in stocks:
    if not os.path.exists("./%s.csv" % stock):
        data = yf.download(stock, start="1996-07-26", end="2026-05-03") #period="10y")
        data.to_csv("./%s.csv" % stock)
