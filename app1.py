import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from openai import OpenAI
from flask import Flask,render_template ,jsonify,request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

def get_financial_statements(ticker):
    url1 = f"https://finance.yahoo.com/quote/{ticker}/balance-sheet?p={ticker}"
    header1 = {'Connection': 'keep-alive',
                'Expires': '-1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                }
        
    r1 = requests.get(url1, headers=header1)
    html1 = r1.text
    soup1 = BeautifulSoup(html1, "html.parser")

    div1 = soup1.find_all('div', attrs={'class': 'D(tbhg)'})
    if len(div1) < 1:
        print("Fail to retrieve table column header")
        exit(0)

    col = []
    for h in div1[0].find_all('span'):
        text = h.get_text()
        if text != "Breakdown":
            col.append( datetime.strptime(text, "%m/%d/%Y") )
    
    df1 = pd.DataFrame(columns=col)
    for div1 in soup1.find_all('div', attrs={'data-test': 'fin-row'}):
        i = 0
        idx = ""
        val = []
        for h in div1.find_all('span'):
            if i == 0:
                idx = h.get_text()
            else:
                num = int(h.get_text().replace(",", "")) * 1000
                val.append( num )
            i += 1
        if len(val) == 1:
            val.extend([0,0,0])
        row = pd.DataFrame([val], columns=col, index=[idx] )
        # print(row)
        df1 = pd.concat([df1, row], ignore_index=True)

    # print(df1)

    url2 = f"https://finance.yahoo.com/quote/{ticker}/cash-flow?p={ticker}"
    header2 = {'Connection': 'keep-alive',
                'Expires': '-1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                }
        
    r2 = requests.get(url2, headers=header2)
    html2 = r2.text
    soup2 = BeautifulSoup(html2, "html.parser")

    div2 = soup2.find_all('div', attrs={'class': 'D(tbhg)'})
    if len(div2) < 1:
        print("Fail to retrieve table column header")
        exit(0)

    col = []
    for h in div2[0].find_all('span'):
        text = h.get_text()
        if text != "Breakdown":
            if(text=="ttm"):
                col.append('TTM')
                continue
            col.append( datetime.strptime(text, "%m/%d/%Y") )
    
    df2 = pd.DataFrame(columns=col)
    for div2 in soup2.find_all('div', attrs={'data-test': 'fin-row'}):
        i = 0
        idx = ""
        val = []
        for h in div2.find_all('span'):
            if i == 0:
                idx = h.get_text()
            else:
                num = int(h.get_text().replace(",", "")) * 1000
                val.append( num )
            i += 1

        if len(val) < 5:
            val.extend([0]*(5-len(val)))
        row = pd.DataFrame([val], columns=col, index=[idx] )
        # print(row)
        df2 = pd.concat([df2, row], ignore_index=True)

    # print(df2)


    url3 = f"https://finance.yahoo.com/quote/{ticker}/financials?p={ticker}"
    header3 = {'Connection': 'keep-alive',
                'Expires': '-1',
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                }
        
    r3 = requests.get(url3, headers=header3)
    html3 = r3.text
    soup3 = BeautifulSoup(html3, "html.parser")

    div3 = soup3.find_all('div', attrs={'class': 'D(tbhg)'})
    if len(div3) < 1:
        print("Fail to retrieve table column header")
        exit(0)

    col = []
    for h in div3[0].find_all('span'):
        text = h.get_text()
        if text != "Breakdown":
            if(text=="ttm"):
                col.append('TTM')
                continue
            col.append( datetime.strptime(text, "%m/%d/%Y") )
    
    df3 = pd.DataFrame(columns=col)
    for div3 in soup3.find_all('div', attrs={'data-test': 'fin-row'}):
        i = 0
        idx = ""
        val = []
        for h in div3.find_all('span'):
            if i == 0:
                idx = h.get_text()
            else:
                num = int(h.get_text().replace(",", "")) * 1000
                val.append( num )
            i += 1
        if len(val) < 5:
            val.extend(np.zeros(5-len(val)))
        row = pd.DataFrame([val], columns=col, index=[idx])
        df3 = pd.concat([df3, row], ignore_index=True)

    # print(df3)

    return df1,df2,df3

def calculate_free_cash_flow(balance_sheet, cash_flow_statement, income_statement):
    # Calculate Free Cash Flow (FCF)
    # operating_cash_flow = cash_flow_statement['Operating Cash Flow']
    operating_cash_flow = cash_flow_statement.iloc[0,1:].tolist()
    print(operating_cash_flow)
    capital_expenditure = cash_flow_statement.iloc[5,1:].tolist()
    net_income = income_statement.iloc[17,1:].tolist()

    free_cash_flow = [operating_cash_flow - capital_expenditure for (operating_cash_flow,capital_expenditure) in zip(operating_cash_flow,capital_expenditure)]

    
    # Alternative method: FCF = Net Income + Depreciation & Amortization - Capital Expenditure
    # free_cash_flow = net_income + income_statement['Depreciation & Amortization'] - capital_expenditure

    # Calculate Discount Rate
    cost_of_equity = [calculate_cost_of_equity(income_statement)]*4
    cost_of_debt = calculate_cost_of_debt(income_statement, balance_sheet)
    weight_equity = calculate_weight_equity(balance_sheet)
    weight_debt = calculate_weight_debt(balance_sheet)

    print(cost_of_equity,cost_of_debt,weight_equity,weight_debt)

    l1 = [cost_of_equity/weight_equity for (cost_of_equity,weight_equity) in zip(cost_of_equity,weight_equity)]
    l2 = [cost_of_debt/weight_debt for (cost_of_debt,weight_debt) in zip(cost_of_debt,weight_debt)]
    wacc = [l1+ l2 for (l1,l2) in zip(l1,l2)]
    return free_cash_flow, wacc

def calculate_cost_of_equity(income_statement):
    # Calculate cost of equity using CAPM or other models
    risk_free_rate = 0.03  # Example risk-free rate
    market_return = 0.08   # Example market return
    beta = 1.2             # Example beta

    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    return cost_of_equity

def calculate_cost_of_debt(income_statement,balance_sheet):
    # Calculate cost of debt using yields on existing debt or other methods
    interest_expense = income_statement.iloc[20,1:].tolist()
    total_debt = balance_sheet.iloc[10,:].tolist()

    interest_rate = [interest_expense/total_debt for (interest_expense,total_debt) in zip(interest_expense,total_debt)]
    return interest_rate

def calculate_weight_equity(balance_sheet):
    # Calculate the weight of equity in the capital structure
    total_equity = balance_sheet.iloc[2,:].tolist()
    total_assets = balance_sheet.iloc[0,:].tolist()

    weight_equity = [total_equity/total_assets for (total_equity,total_assets) in zip(total_equity,total_assets)]
    return weight_equity

def calculate_weight_debt(balance_sheet):
    # Calculate the weight of debt in the capital structure
    total_debt = balance_sheet.iloc[10,:].tolist()
    total_assets = balance_sheet.iloc[0,:].tolist()

    weight_debt = [total_debt/total_assets for (total_debt,total_assets) in zip(total_debt,total_assets)]
    return weight_debt


def calculate_intrinsic_value(free_cash_flows, discount_rate):
    intrinsic_value = 0
    for i, cash_flow in enumerate(free_cash_flows):
        intrinsic_value += cash_flow / ((1 + discount_rate[i]) ** (i + 1))
    return intrinsic_value/10**9

def get_market_cap(ticker_symbol):
    # Retrieve stock information
    stock_info = yf.Ticker(ticker_symbol)

    # Get current price per share
    current_price = stock_info.history(period="1d").iloc[-1]['Close']

    # Get total number of shares outstanding (market capitalization is usually provided in billions)
    market_cap = stock_info.info['marketCap'] / 10**9  # Convert to billions

    return market_cap


def gpt_prompt(ticker):

    client = OpenAI(
        api_key =  ""
    )

    balance_sheet_df, cash_flow_statement_df, income_statement_df = get_financial_statements(ticker)
    free_cash_flow, discount_rate = calculate_free_cash_flow(balance_sheet_df, cash_flow_statement_df, income_statement_df)
    print("Free Cash Flow:", free_cash_flow)
    print("Discount Rate (WACC):", discount_rate)

    intrinsic_val = calculate_intrinsic_value(free_cash_flow, discount_rate)
    print("Intrinsic Value :", intrinsic_val)

    market_price = get_market_cap(ticker)
    print("Price of outstanding shares: ",market_price)

    prompt = f"The intrinsic value of {ticker} is {intrinsic_val}. Market Price of outstanding shares is: {market_price}. Give a 2-liner brief about the stock analysis and also a conclusion in form of 'BUY' 'HOLD' 'SELL'"

    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role":"user",
                "content":prompt
            }
        ],
        model = "gpt-3.5-turbo"
    )

    text_output = chat_completion.choices[0].message.content
    return text_output
    # print(chat_completion.choices[0].message.content)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker_symbol = data['ticker_symbol']

    message = gpt_prompt(ticker_symbol)

    try:
      response = {
            'message': message
        }
      
    except Exception as e:
        response = {
            'message': str(e)
        }

    return jsonify(response)

@app.route('/stock-data')
def get_stock_data():
    symbol = request.args.get('symbol')
    data = yf.download(symbol, period='1y')['Close'].reset_index()
    # Convert dates to strings and prices to integers
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    data['Close'] = data['Close'].astype(int)
    # Convert DataFrame to list of dictionaries
    stock_data = data.to_dict(orient='records')
    return jsonify(stock_data)

if __name__ == '__main__':
    app.run(debug=True)









# def ratio_analysis(stock_symbol, start_date, end_date):
#     # Fetch historical market data
#     data = yf.download(stock_symbol, start=start_date, end=end_date)

#     # Initialize arrays to store ratios for each year
#     pe_ratios = []
#     pb_ratios = []
#     debt_equity_ratios = []
#     current_ratios = []

#     # Iterate over each year and calculate ratios
#     for year, df_year in data.groupby(data.index.year):
#         # Calculate PE ratio (Price-to-Earnings Ratio)
#         pe_ratio = df_year['Adj Close'].mean() / df_year['Earnings'].mean()

#         # Calculate PB ratio (Price-to-Book Ratio)
#         pb_ratio = df_year['Adj Close'].mean() / df_year['Book Value'].mean()

#         # Calculate Debt-to-Equity ratio
#         debt_equity_ratio = df_year['Total Debt'].mean() / df_year['Total Equity'].mean()

#         # Calculate Current ratio
#         current_ratio = df_year['Total Current Assets'].mean() / df_year['Total Current Liabilities'].mean()

#         # Append ratios to respective arrays
#         pe_ratios.append(pe_ratio)
#         pb_ratios.append(pb_ratio)
#         debt_equity_ratios.append(debt_equity_ratio)
#         current_ratios.append(current_ratio)

#     return {
#         'PE Ratios': pe_ratios,
#         'PB Ratios': pb_ratios,
#         'Debt-to-Equity Ratios': debt_equity_ratios,
#         'Current Ratios': current_ratios
#     }
# def fundamental_analysis(stock_ticker, discount_rate=0.1, annual_report_url=None):
#     # Perform fundamental analysis on the given stock
    
#     # Step 1: Get financial data
#     balance_sheet = get_balance_sheet_from_yfinance_web(stock_ticker)
#     print(balance_sheet)
#     # print(financials, balance_sheet)
#     # Step 2: Calculate ratios
#     ratios = calculate_ratios(financials, balance_sheet)
    
#     # Step 3: Intrinsic value calculation
#     intrinsic_value = intrinsic_value_calculation(financials, discount_rate)
    
#     # Step 4: Parse annual report (if provided)
#     if annual_report_url:
#         parsed_data = parse_annual_report(annual_report_url)
#     else:
#         parsed_data = None
    
#     return ratios, intrinsic_value, parsed_data


# get_financial_statements("AAPL")
# # Example usage:
# stock_ticker = 'GOOG'
# ratios, intrinsic_value, parsed_data = fundamental_analysis(stock_ticker)
# print("Ratios:", ratios)
# print("Intrinsic Value:", intrinsic_value)
# print("Parsed Data:", parsed_data)