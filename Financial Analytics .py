#!/usr/bin/env python
# coding: utf-8

# # FINANCIAL ANALYTICS OF TOP 10 FORTUNE 500 COMPANIES IN RETAIL SECTOR
# 

# In[3]:


import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Importing all the required libraries for data cleaning, web scraping and analysis and visualization

# In[4]:


df = pd.read_csv(r'G:\OWN PROJECTS\PERSONAL PROJECTS\Finanical Analytics\fortune_global500_2024.csv')
df


# importing csv file to jupyter notebook for analysis. Data Source:Kaggle:https://www.kaggle.com/datasets/sneharangole/2024-fortune-global-500-companies

# ## EXPLORATORY DATA ANALYSIS

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isna().sum()


# In[8]:


df.sort_values(by='PROFITS ($M)',ascending=False)


# In[9]:


df.sort_values(by='EMPLOYEES')


# In[10]:


df['PROFITS ($M)'].describe()


# It seems i need to merge another dataset so that i can do thorough EDA as this Dataset does not provide enough information. Ive found another dataset relating to Fortune 1000 companies but from 2022 but it has more columns with industrial, geographical and financial information required to pull from Yahoo finance. 

# In[11]:


df2 = pd.read_csv(r'G:\OWN PROJECTS\PERSONAL PROJECTS\Finanical Analytics\Fortune_1000.csv')
df2


# In[12]:


df2.columns


# In[13]:


columns_to_drop=['rank', 'rank_change', 'revenue', 'profit',
       'num. of employees','newcomer','ceo_founder', 'ceo_woman', 'profitable', 'prev_rank', 'CEO', 'Website','Market Cap']

df2= df2.drop(columns=columns_to_drop)


# In[14]:


df2


# In[15]:


merged_df = df.merge(df2, how='left', left_on='NAME', right_on='company')
merged_df


# In[16]:


merged_df.isna().sum()


# Although there are 370 missing values across 5 columns in the dataset, I've verified that I have complete details for the top 5 retail companies in North America, according to the Fortune 500. Therefore, I'll proceed by focusing on those companies and ignore the missing values in the rest of the dataset.

# In[17]:


main = merged_df[merged_df['sector']=='Retailing'].head(5)
main


# top 5 fortune 500 retail companies in north america

# In[18]:


main = main.drop(columns=['REVENUES ($M)', 'REVENUE PERCENT CHANGE',
       'PROFITS ($M)', 'PROFITS PERCENT CHANGE', 'ASSETS ($M)', 'EMPLOYEES',
       'CHANGE IN RANK', 'YEARS ON GLOBAL 500 LIST', 'company', 'sector',
       'city', 'state'])
main


# Removed unnecessary columns

# In[19]:


df=main


# In[20]:


# Initialize a dictionary to store the results
results = {name: {} for name in df['NAME']}

# Fetch financial data for each ticker
for index, row in df.iterrows():
    ticker = row['Ticker']
    stock = yf.Ticker(ticker)

    # Get financial statements
    try:
        income_stmt = stock.quarterly_financials  # Last 4 quarters (TTM)
        balance_sheet = stock.quarterly_balance_sheet
        cash_flow = stock.quarterly_cashflow

        # Extract TTM metrics
        results[row['NAME']]['Total Revenue'] = income_stmt.loc['Total Revenue'].iloc[:4].sum()
        results[row['NAME']]['Gross Profit'] = income_stmt.loc['Gross Profit'].iloc[:4].sum()
        results[row['NAME']]['Operating Income'] = income_stmt.loc['Operating Income'].iloc[:4].sum()
        results[row['NAME']]['Net Income'] = income_stmt.loc['Net Income'].iloc[:4].sum()
        results[row['NAME']]['EBITDA'] = income_stmt.loc['EBITDA'].iloc[:4].sum()
        results[row['NAME']]['Diluted EPS'] = stock.info.get('trailingEps', None)
        results[row['NAME']]['Total Expenses'] = income_stmt.loc['Total Expenses'].iloc[:4].sum()
        results[row['NAME']]['Interest Expense'] = income_stmt.loc['Interest Expense'].iloc[:4].sum()
        
        # Extract balance sheet items (most recent quarter)
        results[row['NAME']]['Total Assets'] = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
        results[row['NAME']]['Total Liabilities'] = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
        results[row['NAME']]['Total Equity'] = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] if 'Total Equity Gross Minority Interest' in balance_sheet.index else None
        results[row['NAME']]['Current Assets'] = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else None
        results[row['NAME']]['Current Liabilities'] = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None
        
        # Net Debt calculation (Total Debt - Cash and Cash Equivalents) from most recent quarter
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else None
        cash_and_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else None
        results[row['NAME']]['Net Debt'] = total_debt - cash_and_equivalents if total_debt is not None and cash_and_equivalents is not None else None
        
        results[row['NAME']]['Working Capital'] = results[row['NAME']]['Current Assets'] - results[row['NAME']]['Current Liabilities'] if results[row['NAME']]['Current Assets'] is not None and results[row['NAME']]['Current Liabilities'] is not None else None

        # Cash flow metrics based on TTM
        results[row['NAME']]['Operating Cash Flow'] = cash_flow.loc['Operating Cash Flow'].iloc[:4].sum() if 'Operating Cash Flow' in cash_flow.index else None
        results[row['NAME']]['Free Cash Flow'] = cash_flow.loc['Free Cash Flow'].iloc[:4].sum() if 'Free Cash Flow' in cash_flow.index else None
        results[row['NAME']]['Capital Expenditure'] = cash_flow.loc['Capital Expenditures'].iloc[:4].sum() if 'Capital Expenditures' in cash_flow.index else None
        results[row['NAME']]['Cash Flow From Financing Activities'] = cash_flow.loc['Cash Flow From Financing Activities'].iloc[:4].sum() if 'Cash Flow From Financing Activities' in cash_flow.index else None

    except Exception as e:
        print(f"Error retrieving data for {row['NAME']}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
results_df


# Pulled all financial information required for my analysis using yfinance module. These will provide solid ground for the rest of my analysis

# In[21]:


results_df.columns


# In[22]:


# Drop specified columns from the results DataFrame
results_df = results_df.drop(columns=['Capital Expenditure', 'Cash Flow From Financing Activities'])


# In[23]:


results_df.info()


# In[24]:


# List of columns to divide by a billion
columns_to_divide = [
    'Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income',
    'EBITDA', 'Total Expenses', 'Interest Expense',
    'Total Assets', 'Total Liabilities', 'Total Equity', 
    'Current Assets', 'Current Liabilities', 'Net Debt', 
    'Working Capital', 'Operating Cash Flow', 'Free Cash Flow'
]

# Divide specified columns by a billion
for column in columns_to_divide:
    if column in results_df.columns:
        results_df[column] = (results_df[column] / 1_000_000_000).round(2)  # Dividing by a billion

# Display the updated DataFrame
results_df


# In[25]:


results_df.iloc[:, 0]


# **removed unnecessary columns and divided everything by 1 billion

# In[26]:


results_df.describe()


# In[27]:


(results_df['Gross Profit']/results_df['Total Revenue'])*100


# In[28]:


results_df.reset_index(inplace=True)

# Rename the index column to 'Company'
results_df.rename(columns={'index': 'Company'}, inplace=True)
results_df


# **Profitability Analysis**

# In[29]:


# Calculating the margins
results_df['Gross Margin'] = results_df['Gross Profit'] / results_df['Total Revenue'] * 100
results_df['Operating Margin'] = results_df['Operating Income'] / results_df['Total Revenue'] * 100
results_df['Net Profit Margin'] = results_df['Net Income'] / results_df['Total Revenue'] * 100
results_df['EBITDA Margin'] = results_df['EBITDA'] / results_df['Total Revenue'] * 100

# Define colors and patterns for each bar plot
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
patterns = ['/', 'o', '\\', '*', 'x']

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Margin Analysis for Companies", fontsize=16)

# Gross Margin
axes[0, 0].bar(results_df['Company'], results_df['Gross Margin'], color=colors[0], edgecolor='black', hatch=patterns[0])
axes[0, 0].set_title('Gross Margin (%)', fontsize=14, color=colors[0])
axes[0, 0].set_ylabel('Percentage (%)')

# Operating Margin
axes[0, 1].bar(results_df['Company'], results_df['Operating Margin'], color=colors[1], edgecolor='black', hatch=patterns[1])
axes[0, 1].set_title('Operating Margin (%)', fontsize=14, color=colors[1])
axes[0, 1].set_ylabel('Percentage (%)')

# Net Profit Margin
axes[1, 0].bar(results_df['Company'], results_df['Net Profit Margin'], color=colors[2], edgecolor='black', hatch=patterns[2])
axes[1, 0].set_title('Net Profit Margin (%)', fontsize=14, color=colors[2])
axes[1, 0].set_ylabel('Percentage (%)')

# EBITDA Margin
axes[1, 1].bar(results_df['Company'], results_df['EBITDA Margin'], color=colors[3], edgecolor='black', hatch=patterns[3])
axes[1, 1].set_title('EBITDA Margin (%)', fontsize=14, color=colors[3])
axes[1, 1].set_ylabel('Percentage (%)')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In this analysis, **Home Depot** and **Amazon** emerge as leaders in profitability and operational efficiency, showing strong gross profit, operating, and EBITDA margins, with Home Depot leading in net profit margin as well. **Costco** operates with the lowest margins, consistent with its high-volume, low-margin strategy, prioritizing price competitiveness. **Target** and **Walmart** display stable but moderate profitability across metrics, reflecting balanced operations. Overall, Home Depot and Amazon’s high margins underscore strong cash generation and cost control, while Costco’s margins highlight its lean, price-focused approach, and Walmart and Target’s metrics suggest solid, consistent performance.

# **Expense Analysis**

# In[30]:


# Calculating the margins
results_df['Expense Ratio'] = results_df['Total Expenses'] / results_df['Total Revenue'] 
results_df['Interest Coverage Ratio'] = results_df['EBITDA'] / results_df['Interest Expense'] 

# Define colors and patterns for each bar plot
colors = ['#E15759', '#76B7B2']
patterns = ['*', 'x']

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Change to 1 row and 2 columns
fig.suptitle("Expense Analysis for Companies", fontsize=16)

# Expense Ratio
axes[0].bar(results_df['Company'], results_df['Expense Ratio'], color=colors[0], edgecolor='black', hatch=patterns[0])
axes[0].set_title('Expense Ratio', fontsize=14, color=colors[0])
axes[0].set_ylabel('Billion $')

# Interest Coverage Ratio
axes[1].bar(results_df['Company'], results_df['Interest Coverage Ratio'], color=colors[1], edgecolor='black', hatch=patterns[1])
axes[1].set_title('Interest Coverage Ratio', fontsize=14, color=colors[1])
axes[1].set_ylabel('Billion $')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# **Liquidity Ratios**

# In[31]:


# Calculate liquidity ratios
results_df['Current Ratio'] = results_df['Current Assets'] / results_df['Current Liabilities']
results_df['Net Debt to EBITDA'] = results_df['Net Debt'] / results_df['EBITDA']

# Prepare data for plotting
plot_data = results_df[['Company', 'Current Ratio', 'Net Debt to EBITDA']].melt(id_vars='Company', 
                                                                                  var_name='Liquidity Ratio', 
                                                                                  value_name='Value')

# Create a clustered bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Company', y='Value', hue='Liquidity Ratio', data=plot_data, palette='viridis')
plt.title('Liquidity Ratios Comparison')
plt.xlabel('Company')
plt.ylabel('Ratio Value')
plt.xticks(rotation=45)
plt.legend(title='Liquidity Ratio')
plt.tight_layout()

# Show the plot
plt.show()


# *Current Ratio Performance:*
# 
# Companies like Amazon and Home Depot may exhibit current ratios above 1, indicating they can easily meet their short-term liabilities with available assets.
# Walmart and Target, with current ratios below 1, may face challenges in covering short-term obligations, suggesting potential liquidity risks.
# 
# *Net Debt to EBITDA Analysis:*
# 
# Amazon shows a lower Net Debt to EBITDA ratio, indicating a favorable debt position and suggesting that it can pay off its net debt relatively quickly with its earnings.
# Home Depot has a higher ratio, which may indicate more reliance on debt, potentially posing risks in a downturn.
# Costco’s negative ratio may suggest minimal debt levels compared to its earnings, highlighting strong financial health.
# 

# In[32]:


# Calculate cash flow metrics
results_df['Operating Cash Flow to Net Income'] = results_df['Operating Cash Flow'] / results_df['Net Income']
results_df['Free Cash Flow to Revenue'] = results_df['Free Cash Flow'] / results_df['Total Revenue']

# Set the style
sns.set(style="whitegrid")

# Bar Chart for Operating Cash Flow to Net Income
plt.figure(figsize=(10, 6))
sns.barplot(x='Company', y='Operating Cash Flow to Net Income', data=results_df, palette='viridis')
plt.title('Operating Cash Flow to Net Income Ratio by Company')
plt.ylabel('Ratio')
plt.xlabel('Company')
plt.ylim(0, results_df['Operating Cash Flow to Net Income'].max() + 0.2)  # Adjust the y-axis for better visibility
plt.axhline(1, color='red', linestyle='--', label='Ideal Ratio (1.0)')
plt.legend()
plt.show()

# Pie Chart for Free Cash Flow to Revenue
plt.figure(figsize=(8, 8))
plt.pie(results_df['Free Cash Flow to Revenue'], labels=results_df['Company'], autopct='%1.1f%%', startangle=140)
plt.title('Free Cash Flow to Revenue Distribution by Company')
plt.show()


# Overall, Amazon and Home Depot emerge as strong performers in terms of cash generation relative to their income and revenue. Their metrics suggest solid operational efficiency and ample cash available for future growth.
# Walmart shows robust cash conversion efficiency but needs to focus on improving its free cash flow relative to revenue.
# Costco, Target, and Home Depot should monitor their cash flow metrics closely to ensure they have sufficient liquidity to capitalize on growth opportunities without compromising operational stability.

# **BRINGING IN HISTORICAL DATA FOR THE 5 COMPANIES**

# In[33]:


company = main.drop(columns=['RANK'])


# 

# In[34]:


company


# In[35]:


#Loop through each ticker to get info keys
for ticker in company['Ticker']:
    stock = yf.Ticker(ticker)  # Correct instantiation
    info_keys = stock.info.keys()  # Access info keys
    print(f"Keys for {ticker}:")
    print(list(info_keys))
    print("\n")  # Add a newline for readability


# In[36]:


# Define the key metrics
metrics = [
    'ebitda', 'totalDebt', 'totalRevenue', 'freeCashflow',
    'operatingCashflow', 'netIncomeToCommon', 'revenueGrowth',
    'earningsGrowth'
]
# Empty list to store the results
historical_data = []

# Loop through each company to retrieve data
for ticker in company["Ticker"]:
    stock = yf.Ticker(ticker)

    # Retrieve the financials (income statement, balance sheet, cash flow)
    income_statement = stock.financials.T  # Transpose to get years as rows
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T

    # Convert index to datetime
    income_statement.index = pd.to_datetime(income_statement.index, errors='coerce')
    balance_sheet.index = pd.to_datetime(balance_sheet.index, errors='coerce')
    cash_flow.index = pd.to_datetime(cash_flow.index, errors='coerce')

    # Filter for the last five years
    last_five_years = sorted(income_statement.index.year.unique(), reverse=True)[:5]
    
    # Loop through each year
    for year in last_five_years:
        year_str = str(year)  # Convert year to string for consistent indexing
        
        # Extract key metrics from each statement
        data_row = {
            "Ticker": ticker,
            "Year": year_str,
            "Total Revenue": (income_statement.loc[income_statement.index.year == year, "Total Revenue"].values[0] if "Total Revenue" in income_statement.columns and not income_statement.loc[income_statement.index.year == year, "Total Revenue"].empty else None),
            "Gross Profit": (income_statement.loc[income_statement.index.year == year, "Gross Profit"].values[0] if "Gross Profit" in income_statement.columns and not income_statement.loc[income_statement.index.year == year, "Gross Profit"].empty else None),
            "Operating Income": (income_statement.loc[income_statement.index.year == year, "Operating Income"].values[0] if "Operating Income" in income_statement.columns and not income_statement.loc[income_statement.index.year == year, "Operating Income"].empty else None),
            "Net Income": (income_statement.loc[income_statement.index.year == year, "Net Income"].values[0] if "Net Income" in income_statement.columns and not income_statement.loc[income_statement.index.year == year, "Net Income"].empty else None),
            "EBITDA": (income_statement.loc[income_statement.index.year == year, "EBITDA"].values[0] if "EBITDA" in income_statement.columns and not income_statement.loc[income_statement.index.year == year, "EBITDA"].empty else None),
            "Diluted EPS": stock.info.get("trailingEps"),
            "Total Assets": (balance_sheet.loc[balance_sheet.index.year == year, "Total Assets"].values[0] if "Total Assets" in balance_sheet.columns and not balance_sheet.loc[balance_sheet.index.year == year, "Total Assets"].empty else None),
            "Total Liabilities": (balance_sheet.loc[balance_sheet.index.year == year, "Total Liabilities Net Minority Interest"].values[0] if "Total Liabilities Net Minority Interest" in balance_sheet.columns and not balance_sheet.loc[balance_sheet.index.year == year, "Total Liabilities Net Minority Interest"].empty else None),
            "Total Equity": (balance_sheet.loc[balance_sheet.index.year == year, "Common Stock Equity"].values[0] if "Common Stock Equity" in balance_sheet.columns and not balance_sheet.loc[balance_sheet.index.year == year, "Common Stock Equity"].empty else None),
            "Revenue Growth": stock.info.get('revenueGrowth'),
            "Earnings Growth": stock.info.get('earningsGrowth'),
        }
        
        historical_data.append(data_row)

# Convert to DataFrame for easy analysis
historical_df = pd.DataFrame(historical_data)
historical_df.set_index(["Ticker", "Year"], inplace=True)
historical_df


# In[37]:


# Assuming your DataFrame is named `historical_df`

# Reset the index to make Ticker a regular column
historical_df.reset_index(inplace=True)

# Fill down the ticker names
historical_df['Ticker'] = historical_df['Ticker'].ffill()

# Set the index back to Ticker and Year
historical_df.set_index(['Ticker', 'Year'], inplace=True)

# Rename the 'Ticker' column to 'Company' (This step is redundant since Ticker is already filled down)
# If you want to rename the index level instead, you can use:
historical_df.index = historical_df.index.rename(['Company', 'Year'])

# Display the updated DataFrame
historical_df


# In[38]:


import pandas as pd

# Assuming your DataFrame is named `historical_df`
# Sort the DataFrame by Company and Year
historical_df_sorted = historical_df.sort_index(level=['Company', 'Year'])

# Reset the index if desired
historical_df_sorted.reset_index(inplace=True)

# Optionally, fill missing values, for example with forward fill
# historical_df_sorted.fillna(method='ffill', inplace=True)

# Display the sorted DataFrame
historical_df_sorted


# In[39]:


data = {
    'c_code':[1001,1002,1003,1004,1005],
    'NAME': ['Walmart', 'Amazon', 'Costco Wholesale', 'Home Depot', 'Target'],
    'Ticker': ['WMT', 'AMZN', 'COST', 'HD', 'TGT'],
    
}
company_dm = pd.DataFrame(data)
company_dm


# In[40]:


historical_df = historical_df_sorted.merge(company_dm,how='left',left_on='Company',right_on='Ticker')
historical_df = historical_df.drop(columns=['NAME','Ticker','Company'])
new_order = ['c_code'] + [col for col in historical_df.columns if col != 'c_code']
historical_df = historical_df[new_order]

# Display the updated DataFrame
historical_df


# In[41]:


historical = historical_df.dropna()


# In[42]:


historical.to_csv(r"G:\OWN PROJECTS\PERSONAL PROJECTS\Finanical Analytics\historical_dm.csv")
company_dm.to_csv(r"G:\OWN PROJECTS\PERSONAL PROJECTS\Finanical Analytics\company_dm.csv")


# I need efficiency ratios for my powerbi dashboard

# In[43]:


company_dm


# In[87]:


metrics = ['Inventory', 'Accounts Receivable', 'Total Noncurrent Assets', 'Revenue', 'COGS']
# Create an empty DataFrame to store results
results = pd.DataFrame()

for ticker in company_dm['Ticker']:
    # Fetch the financial data
    company = yf.Ticker(ticker)

    # Retrieve and transpose the balance sheet data
    balance_sheet = company.balance_sheet.T
    # Retrieve and transpose the income statement data
    income_statement = company.financials.T
    
    # Check available columns to adapt the column names for common metrics
    print(f"Available columns for {ticker} (Balance Sheet):", balance_sheet.columns)
    print(f"Available columns for {ticker} (Income Statement):", income_statement.columns)

    # Attempting common variations of required metrics names
    metric_mapping = {
        'Inventory': ['Inventory', 'Inventories'],
        'Accounts Receivable': ['Accounts Receivable', 'Receivables'],
        'Total Noncurrent Assets': ['Total Noncurrent Assets', 'Total Non Current Assets', 'Noncurrent Assets', 'Long Term Assets'],
        'Revenue': ['Total Revenue', 'Revenue'],
        'COGS': ['Cost of Goods Sold', 'COGS', 'Cost Of Revenue']
    }

    # Filter for the required metrics based on available columns
    selected_data = pd.DataFrame(index=balance_sheet.index)
    
    # Loop to get metrics from balance sheet
    for metric, options in metric_mapping.items():
        for option in options:
            if option in balance_sheet.columns:
                selected_data[metric] = balance_sheet[option]
                break
        # Get Revenue and COGS from the income statement
        if metric in ['Revenue', 'COGS']:
            for option in options:
                if option in income_statement.columns:
                    selected_data[metric] = income_statement[option]
                    break

    # Add the ticker information
    selected_data['Ticker'] = ticker
    
    # Append data to the main results DataFrame
    results = pd.concat([results, selected_data])

# Reset index and organize columns
results.reset_index(inplace=True)
results.rename(columns={'index': 'Year'}, inplace=True)

# Display the final data
results


# In[88]:


results.to_csv(r'G:\OWN PROJECTS\PERSONAL PROJECTS\Finanical Analytics\efficiency metrics.csv')


# In[ ]:





# In[ ]:




