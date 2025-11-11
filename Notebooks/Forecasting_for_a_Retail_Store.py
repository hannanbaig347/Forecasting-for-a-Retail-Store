# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1 - Data Exploration & Cleaning, EDA

# %% [markdown]
# ### 1.1 Load the Historical Data

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
# datetime objects represent specific points in time
# timedelta objects represent durations.    

np.random.seed(42)

start_date = datetime(2020, 1, 1)
num_months = 48 # 4 years of data (business cycle)

date_rng = pd.date_range(start_date, periods = num_months, freq = 'MS') 
# Creates a sequence of monthly timestamps, starting at 2020-01-01, with 48 entries, each aligned to the start of the month ('MS' = Month Start).
# MS = Month Start. It means the dates will be aligned to the start of each month (e.g., Jan 1, Feb 1, Mar 1, etc.).

sales_data = pd.DataFrame(date_rng, columns = ['Date'])


# Base sales with a slight upward trend
base_sales = 10000 + np.arange(num_months) * 50 # create a NumPy array named base_sales
# This line is generating a sequence of increasing monthly base sales, starting from 10,000 and increasing by 50 units per month
# np.arange() is a NumPy function that returns an array of evenly spaced values within a specified range.
# Starts the sales at 10,000 units in the first month (January 2020), and increases by 50 each subsequent month.
# sales grow over time due to various factors (e.g., market expansion, product maturity).


# We're creating a seasonal pattern — like in sales, temperature, or web traffic — that repeats predictably throughout the year
seasonal_component = 2000 * np.sin(2 * np.pi * (sales_data['Date'].dt.month - 1) / 12 + np.pi/2) \
                   + 1500 * np.sin(2 * np.pi * (sales_data['Date'].dt.month - 1) / 6)
# This results in a seasonal pattern like:
    #High: Oct–Dec
    #Low: Jan–Feb
    #Smaller bumps: May–Jun, Aug–Sep

# The number 2000 just means: “Make this wave big enough to matter in sales.”


# Generating random fluctuations — like tiny unpredictable changes in sales (e.g., weather, supply hiccups, customer behavior).
noise = np.random.normal(0, 500, num_months)  
# This uses a normal distribution (bell curve) with:
    # Mean = 0 → centered around zero (so noise can go positive or negative)
    # Standard deviation = 500 → most noise will fall between -500 and +500
    # num_months = how many months of noise you want

sales_data['SalesAmount'] = (base_sales + seasonal_component + noise).astype(int)
sales_data['SalesAmount'] = sales_data['SalesAmount'].clip(lower=2000) #Ensure no negative sales



# Add promotional flags (randomly)
sales_data['Promotion'] = np.random.choice([0,1], size = num_months, p = [0.8, 0.2])

# You randomly mark whether there was a promotion each month:
    # this generates 48 random choices
    # 80% chance of picking 0 (no promotion)
    # 20% chance of picking 1 (promotion active)
    # So in 48 months:
        # Around 38 months will have no promotion
        # Around 10 months will have a promotion
    # This simulates marketing events (like discounts, campaigns) that happen occasionally.


# Increase sales during promotions
sales_data.loc[sales_data['Promotion'] == 1, 'SalesAmount'] *= np.random.uniform(1.1, 1.3, size = (sales_data['Promotion'] == 1).sum())

# If a month had a promotion, it increases sales by 10% to 30%, using a random multiplier between 1.1 and 1.3.


# Add Holiday Flags

sales_data["HolidayMonth"] = (sales_data['Date'].dt.month == 12).astype(int)  # December is a holiday month
#sales_data.loc[sales_data["HolidayMonth"] == 1, "SalesAmount"] *= np.random.uniform(1.15, 1.4) # Increase sales in December by 15% to 40% during holidays
sales_data.loc[sales_data["HolidayMonth"] == 1, "SalesAmount"] *= np.random.uniform(1.15, 1.4, size=(sales_data["HolidayMonth"] == 1).sum())


sales_data["SalesAmount"] = sales_data["SalesAmount"].astype(int)

sales_data.to_csv("retail_sales_mock_data.csv", index=False)



# %% [markdown]
# ### 1.2 Convert date columns to datetime objects and set as index.

# %%
import pandas as pd

sales_data = pd.read_csv("retail_sales_mock_data.csv")

sales_data['Date'] = pd.to_datetime(sales_data['Date'])

sales_data.set_index("Date", inplace=True)
sales_data


# %% [markdown]
# ### 1.3 EDA

# %%
sales_data.info()

# %%
sales_data.describe()

# %% [markdown]
# #### Check for missing values

# %%
sales_data.isnull().sum()

# %% [markdown]
# #### Visualize missing values

# %%
import missingno as msno
msno.bar(sales_data)   


# %% [markdown]
# #### Visualize Sales over Time

# %%
# Your job is to convert sales history into actionable insight:
# “Here’s when to increase stock.”
# “Here’s when to launch campaigns.

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.plot(sales_data.index, sales_data['SalesAmount'])

plt.title("Sales Amount Over Time")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# %% [markdown]
# #### Check for trends, seasonality, and stationarity (e.g., using decomposition plots, ACF/PACF plots, Dickey-Fuller test).

# %% [markdown]
# #### Check Trends through Decomposition

# %% [markdown]
# ##### (a) Classical Decomposition Plot

# %%
#Separates data into
    # Trend:A long-term increase or decrease in the data over time
    # Seasonality: A pattern that repeats at regular intervals (e.g., weekly, monthly, yearly).
    # Stationarity: A property of a time series where statistical properties (like mean and variance) remain constant over time.
        # tells you: “Can I rely on past data to predict the future, or is the game changing?”

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Ensure 'SalesAmount' is a float type (required by statsmodels)
sales_data['SalesAmount'] = sales_data['SalesAmount'].astype(float)

# Decompose using additive model (common for sales data)
decomposition = seasonal_decompose(sales_data['SalesAmount'], model='additive', period=12)

# Plot the decomposition
decomposition.plot()

plt.suptitle('Time Series Decomposition of Sales Data', fontsize=16)
plt.tight_layout()
plt.show()


# %% [markdown]
# ##### (b) STL (Seasonal and Trend decomposition using Loess) Decomposition

# %%
from statsmodels.tsa.seasonal import STL

# STL expects float values
sales_data['SalesAmount'] = sales_data['SalesAmount'].astype(float)

# STL decomposition with period=12 for yearly seasonality
stl = STL(sales_data['SalesAmount'], period=12, robust=True)
result = stl.fit()

# Plotting
fig = result.plot()
plt.suptitle("STL Decomposition of Sales Data", fontsize=16)
plt.tight_layout()
plt.show()


# %% [markdown]
# #### Check Seasonality through ACF and PACF Plots

# %% [markdown]
# #### ACF: AutoCorrelation Function Plot: Shows total correlation with past values
# #### PACF: Partial AutoCorrelation Function: Shows direct correlation with past values

# %%
# Correlation is a way to measure how two things (numbers or variables) move together.
# Correlation is between two different things
# Correlation is used to understand relationships between variables (e.g., ads vs. sales).
# A correlation of +1 means a perfect positive relationship, -1 means a perfect negative relationship, and 0 means no linear relationship


# Autocorrelation is between the same thing at different times
# Autocorrelation is used to understand patterns in time (e.g., seasonal trends or repeated behaviors).

# ACF: Shows how current values are related to previous values
       # Helps detect repeating patterns, trends, or cycles.
       # Used to figure out the MA (Moving Average) part of forecasting models.

#PACF: Similar to ACF, but it removes the influence of the values in between.
       # Helps find the true relationship between the current value and a past value, without the “middlemen”.
       # Used to figure out the AR (Autoregressive) part of forecasting models.

# When building models like ARIMA:
       # PACF helps decide how many past values (lags) to use → this is AR part.
       # ACF helps decide how many past errors to use → this is MA part.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

#ACF plot
plot_acf(sales_data["SalesAmount"], lags=24) #  We are comparing the SalesAmount column to itself at different time shifts (lags).
# with lags=24, you're checking how well current sales correlate with sales from each of the past 24 months, one lag at a time.
plt.title("Autocorrelation Function (ACF) of Sales Amount")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.show()


#PACF Plot
plot_pacf(sales_data["SalesAmount"], lags=24)
plt.title("Partial Autocorrelation Function (PACF) of Sales Amount")
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.grid()
plt.show()

# %% [markdown]
# #### Check Stationarity through Dickey-Fuller test

# %% [markdown]
# #### The Augmented Dickey-Fuller (ADF) test checks if a unit root is present in the series, which would mean the series is non-stationary.

# %%
# p > 0.05 → Non-stationary;  Apply transformations: differencing, log, seasonal adjustment
# p <= 0.05 → Stationary; No transformations needed. Proceed with ARIMA, SARIMA, or other time series models

from statsmodels.tsa.stattools import adfuller

result = adfuller(sales_data['SalesAmount'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Number of lags used: ", result[2])
print("Number of observations used for ADF regression and critical values calculation:", result[3])
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value}")

#    METRIC	        VALUE	                               INTERPRETATION
# ADF Statistic  	-4.333890	                 Very negative → strong evidence against unit root
# p-value	        0.00038	                   < 0.05 → Reject the null hypothesis (H₀)
# Conclusion	The data is stationary	          You don’t need differencing before modeling

# %%
sales_data.isnull().sum()

# %%
sales_data["SalesAmount"].isnull().sum()

# %% [markdown]
# # 2 - Feature Engineering

# %% [markdown]
# ### 2.1 Create Lag Features (A lag feature is the value of the target variable from a previous time step.)

# %% [markdown]
# #### Function that creates lag features (reusable)

# %%
import pandas as pd

def create_lag_features(df, target_column: str, lags: list, drop_na: bool = True) -> pd.DataFrame: #type hints have been used
    # Now I am starting a docstring: a description of what the function does, how to use it, and what it returns
    """
    Adds lag features to the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame (must be time-ordered with a datetime index).
    target_column : str
        The column name to create lags for (e.g., 'SalesAmount').
    lags : list
        A list of integer lag periods (e.g., [1, 2, 3, 6, 12]).
    drop_na : bool
        If True, drops rows with NaN introduced by lagging.

    Returns:
    -------
    df : pd.DataFrame
        DataFrame with new lag columns added.
    """

    df = df.copy()  
    # non-destructive programmimg; Makes a copy of the original data table so we don’t accidentally change the original data. 

    for x in lags:
        lag_col = f'{target_column}_lag_{x}'
        df[lag_col] = df[target_column].shift(x)
        # pandas.shift() moves the values in a Series or DataFrame up or down by a specified number of periods while keeping the original index unchanged.

    if drop_na:
        df.dropna(inplace=True) #This is a built-in Pandas function used to remove rows (or columns) with missing values (NaN) from a DataFrame.

    return df



# %%
sales_data.info()

# %%
import pandas as pd

# create_lag_features(df, target_column: str, lags: list, drop_na: bool = True)

lags_to_create = [1,2,3,6,12]

"""
1, 2, 3: Sales from the previous 1, 2, and 3 days.

6: Sales from 6 days ago (useful for weekly patterns, as it's close to a week).

12: Sales from 12 days ago (often used to capture bi-weekly effects or just a longer history).
"""

sales_data_with_lags = create_lag_features(df = sales_data, target_column= 'SalesAmount', lags = lags_to_create, drop_na = True)

sales_data_with_lags.to_csv("retail_sales_with_lags.csv", index = True)


# %%
sales_data_with_lags.info()

# The original data had 48 months of sales.
# But when I create lag features — like "What were the sales 12 months ago?" — I can't fill in that info for the first 12 months, because there's no earlier data to look back on. 
# So those early rows end up with missing values and get removed.
# That's why the new table has only 36 months of complete data, starting from the 13th month (January 2021).

# %%
sales_data_with_lags.head()

# %% [markdown]
# ### Check Stationarity of this new data with lags using Dickey Fuller Test

# %%
from statsmodels.tsa.stattools import adfuller

result = adfuller(sales_data_with_lags['SalesAmount'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("Number of lags used: ", result[2])
print("Number of observations used for ADF regression and critical values calculation:", result[3])
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value}")

# Data is still stationary after adding lag features
# Now we can use the lagged features to build predictive models, like ARIMA or SARIMA, to forecast future sales.


# %% [markdown]
# #### ACF and PACF Plots again

# %% [markdown]
# #### ACF (Memory of Past Errors) - Moving Average (MA)
# ####  PACF (Memory of Past Sales) - Auto Regressive (AR)

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plot_acf(sales_data_with_lags["SalesAmount"], lags = 17)
plt.show()


plot_pacf(sales_data_with_lags["SalesAmount"], lags = 17)
plt.show()

# %% [markdown]
# # 3.	Model Selection and Training

# %% [markdown]
# ### 3.1 •	Split data into training and validation sets (respecting temporal order).

# %%
trainingSet = sales_data_with_lags.iloc[:-6] # First 30 months for training (Jan 2021-Jun 2023)
validationSet = sales_data_with_lags.iloc[-6:] # Last 6 months for validation (Jul 2023 - Dec 2023)

print("Training Set: " , trainingSet.shape)
print("Validation Set: ", validationSet.shape)

# %% [markdown]
# ### 3.2 Implement ARIMA (Autoregressive Integrated Moving Average) 1,0,1

# %%
from statsmodels.tsa.arima.model import ARIMA

arima_model = ARIMA(trainingSet['SalesAmount'], order = (1,0,1))

# AR (p) = 1 → use sales from 1 month ago / Use the last month’s sales
# I (d) = 0 → don’t difference (because data is stationary)
# MA (q) = 1 → use noise from 1 month ago / Use the last month's error in prediction


arima_result = arima_model.fit()
# This is where the machine “learns” the best-fit values for the AR and MA components.
# It finds parameters that minimize the error (difference between predicted vs. actual sales) on the training data.

print(arima_result.summary())

# log likelihood (-261.472) = The closer to zero (less negative), the better.
    # -261.472 is not terrible, but it’s not excellent either.
    #  It tells us the model does an “okay” job at explaining the data — not super tight, but not wildly off.


# AIC (Akaike Information Criterion (530.945) = Lower is better. 
    # AIC tries to balance two goals:
        # Accuracy (does the model fit the data well?)
        # Simplicity (is it unnecessarily complicated?)
    # AIC = 530.945 is not meaningful on its own. It only means something when you compare it to other models on the same dataset:


# BIC (Bayesian Information Criterion (536.549) = Lower is better. Similar to AIC but penalizes complexity more.

# HQIC (Hannan-Quinn Information Criterion (532.738) = Lower is better. Another model selection criterion.



# values (AIC 530, BIC 536, HQIC 532) are all moderate — not bad, but you won’t know if they’re good until you try another model to compare.


# ar.L1 = 0.5054, p = 0.074: "How much does last month’s sales affect this month?" / Past sales’ impact
      # 0.5054 → moderately positive relationship
             # → If last month was high, this month tends to be high too.
      # AR(1) has some predictive strength (p ≈ 0.07) — it’s likely contributing to short-term trends.

# ma.L1 = 0.4049, p = 0.160: "How much does last month’s prediction error affect this month?" / Past error’s impact
        # 0.4049 → moderate positive relationship
                 # → If last month’s prediction was off, this month tends to follow that pattern.
        # p = 0.160 → Not significant at all (you typically want < 0.05)
# This MA term is probably not helping much. You could try removing it (i.e., test ARIMA(1,0,0)) and see if performance holds.

# p-value for AR/MA terms (0.074, 0.160)	Not < 0.05 → ⚠️	Suggests these coefficients may not help much


# σ² (sigma squared) = 2,116,000
# This is the variance of the error term. It tells you: “How far off is the model, on average, from the actual sales?”
# This value being large (~2 million) means your model's predictions can still be off by a big margi
# This tells us: there's still a lot of variance in actual sales not explained by your model.

# %% [markdown]
# ##### This ARIMA(1,0,1) model provides a moderate fit to your retail sales data — it captures some short-term dependencies (like last month's sales) but doesn't fully explain all variation, as indicated by high residual variance. The autoregressive term (AR) is borderline significant, while the moving average term (MA) likely adds little value, suggesting a simpler model might work just as well or better. Overall, the model is functional but not optimal, and should be refined or compared against alternatives for better forecasting reliability.

# %% [markdown]
# #### Forecast on Validation Set

# %%
# Forecast next 6 steps
forecast1 = arima_result.forecast(steps=6)
forecast1.index = validationSet.index  # Align index

# Compare with actuals
comparison_df = pd.DataFrame({
    'Actual': validationSet['SalesAmount'],
    'Forecast': forecast1
})

print(comparison_df)


'''
Forecast table:
 Month	  ActualSales	Forecast	% Error

Jul-2023	10,042	     9,404	   6.36% under
Aug-2023	11,566	     10,522	   9.02% under
Sep-2023	11,759	     11,087	   5.7% under
Oct-2023	11,890	     11,373	   4.34% under
Nov-2023	11,770	     11,517	   2.15% under
Dec-2023	18,289	     11,590	   36.6% under 

The model is under-predicting sales, especially in December, which is a holiday month with higher sales.
# This suggests the model may not fully capture seasonal spikes, especially during holidays.

'''

# %% [markdown]
# #### Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np



mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Forecast'])
print("Mean Absolute Error (MAE) ARIMA 1,0,1: ", mae)

mse = mean_squared_error(comparison_df['Actual'], comparison_df['Forecast'])
print(f"Mean Squared Error (MSE) ARIMA 1,0,1: {mse:.2f}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) ARIMA 1,0,1: {rmse:.2f}")


mape = mean_absolute_percentage_error(comparison_df['Actual'], comparison_df['Forecast'])
print(f"Mean Absolute Percentage Error (MAPE) ARIMA 1,0,1: {mape:.2f}%")


'''
MEAN ABSOLUTE ERROR (MAE)

Date	    Actual	Forecast	 Error	  Absolute Error   
2023-07-01	10042	9404.25	    -637.75	     637.75
2023-08-01	11566	10522.26	-1043.74	 1043.74
2023-09-01	11759	11087.36	-671.64	     671.64
2023-10-01	11890	11372.99	-517.01	     517.01
2023-11-01	11770	11517.35	-252.65   	 252.65
2023-12-01	18289	11590.32	-6698.68	 6698.68

MAE= 637.75 + 1043.74 + 671.64 + 517.01 + 252.65 + 6698.68 = 9821.47

9821.47 / 6 = 1636.91

"ARIMA(1,0,1) model makes an average absolute forecasting error of ~1637 sales units over the validation period."


MEAN SQUARED ERROR (MSE):

Date	    Actual	  Forecast	   (Actual - Forecast)²
2023-07-01	10042	9404.249277	    (637.750723)² =     406725.75
2023-08-01	11566	10522.267519	(1043.732481)² =    1089205.27
2023-09-01	11759	11087.363305	(671.636695)² =     451096.78
2023-10-01	11890	11372.987668	(517.012332)² =     267305.76
2023-11-01	11770	11517.354833	(252.645167)² =     63831.48
2023-12-01	18289	11590.324373	(6698.675627)² =    44881612.40

MSE: Sum of squared errors: 406725.75 + 1089205.27 + 451096.78 + 267305.76 + 63831.48 + 44881612.40 = 47614777.44

MSE = 47614777.44 / 6 = 7935796.24
"ARIMA(1,0,1) model has a mean squared error of ~7935796.24 sales units over the validation period."



'''

# %% [markdown]
# #### Plot Forecast vs Actuals

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(validationSet['SalesAmount'], label='Actual', marker='o')
plt.plot(forecast1, label='Forecast', marker='x')
plt.title('ARIMA Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.legend()
plt.grid()
plt.show()

# Completely misses non-linear events like the December sales spike (e.g., holidays, promotions, end-of-year push



# %% [markdown]
# ### 3.3 Lets try again by implementing ARIMA (Autoregressive Integrated Moving Average) 1,0,0

# %%
# trainingSet = sales_data_with_lags.iloc[:-6] # First 30 months for training (Jan 2021-Jun 2023)
# validationSet = sales_data_with_lags.iloc[-6:] # Last 6 months for validation (Jul 2023 - Dec 2023)

from statsmodels.tsa.arima.model import ARIMA
arima_model2 = ARIMA(trainingSet['SalesAmount'], order = (1,0,0))
arima_result2 = arima_model2.fit()
print(arima_result2.summary()) 

# %%
forecast2 = arima_result2.forecast(steps = 6)
forecast2.index = validationSet.index

a = pd.DataFrame({
    'Actual': validationSet['SalesAmount'],
    'Forecast': forecast2
})

print(a)

# %% [markdown]
# #### Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae2 = mean_absolute_error(a['Actual'], a['Forecast'])
mse2 = mean_squared_error(a['Actual'], a['Forecast'])
RMSE2 = np.sqrt(mse2)   
MAPE2 = mean_absolute_percentage_error(a['Actual'], a['Forecast'])

print(f"Mean Absolute Error (ARIMA(1,0,0)): {mae2:.2f}")
print(f"Mean Squared Error (ARIMA(1,0,0)): {mse2:.2f}") 
print(f"Root Mean Squared Error (ARIMA(1,0,0)): {RMSE2:.2f}")
print(f"Mean Absolute Percentage Error (ARIMA(1,0,0)): {MAPE2:.2f}%")

# %%
import matplotlib.pyplot as plt
plt.plot(validationSet['SalesAmount'], label = 'Actual', marker = 'o')
plt.plot(forecast2, label = 'Forecast', marker = 'x')
plt.title('ARIMA(1,0,0) Forecast vs Actuals')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.show()

# %% [markdown]
# ### 3.4 Lets try again by implementing ARIMA (Autoregressive Integrated Moving Average) 0,0,1

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_model3 = ARIMA(trainingSet['SalesAmount'], order = (0,0,1))
arima_result3 = arima_model3.fit()
print(arima_result3.summary())

# %%
forecast3 = arima_result3.forecast(steps = 6)
forecast3.index = validationSet.index

b = pd.DataFrame({'Actual': validationSet['SalesAmount'], 'Forecast': forecast3})
print(b)

# %% [markdown]
# #### Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae3 = mean_absolute_error(b['Actual'], b['Forecast'])
mse3 = mean_squared_error(b['Actual'], b['Forecast'])
RMSE3 = np.sqrt(mse3)
MAPE3 = mean_absolute_percentage_error(b['Actual'], b['Forecast'])

print(f"Mean Absolute Error (ARIMA(0,0,1)): {mae3:.2f}")
print(f"Mean Squared Error (ARIMA(0,0,1)): {mse3:.2f}")
print(f"Root Mean Squared Error (ARIMA(0,0,1)): {RMSE3:.2f}")
print(f"Mean Absolute Percentage Error (ARIMA(0,0,1)): {MAPE3:.2f}%")



# %%
import matplotlib.pyplot as plt
plt.plot(validationSet['SalesAmount'], label = 'Actual', marker = 'o')
plt.plot(forecast3, label = 'Forecast', marker = 'x')
plt.title("ARIMA Forecast vs Actuals 0,0,1")
plt.xlabel ("Date")
plt.ylabel('Sales Amount')
plt.legend()
plt.plot()

# %% [markdown]
# ### 3.4 Lets try again by implementing ARIMA (Autoregressive Integrated Moving Average) 0,0,0

# %%
from statsmodels.tsa.arima.model import ARIMA
arima_model4 = ARIMA(trainingSet["SalesAmount"], order = (0,0,0))
arima_result4 = arima_model4.fit()
print(arima_result4.summary())

# %%
forecast4 = arima_result4.forecast(steps = 6)
forecast4.index = validationSet.index
c = pd.DataFrame({"Actual" : validationSet['SalesAmount'],
                   'Forecast': forecast4})
print(c)

# %% [markdown]
# #### Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae4 = mean_absolute_error(c['Actual'], c['Forecast'])
mse4 = mean_squared_error(c['Actual'], c['Forecast'])
RMSE4 = np.sqrt(mse4)
MAPE4 = mean_absolute_percentage_error(c['Actual'], c['Forecast'])

print(f"Mean Absolute Error ARIMA(0,0,0): {mae4: .2f}")
print(f"Mean Squared Error ARIMA(0,0,0): {mse4:.2f}")
print(f"Root Mean Squared Error ARIMA(0,0,0): {RMSE4:.2f}")
print(f"Mean Absolute Percentage Error ARIMA(0,0,0): {MAPE4:.2f}%")

# %%
import matplotlib.pyplot as plt
plt.plot(validationSet["SalesAmount"], label = 'Actual', marker = 'o')
plt.plot(forecast4, label = 'Forecast', marker = 'x')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ### 3.4 Lets try again by implementing ARIMA (Autoregressive Integrated Moving Average) 1,1,1

# %%
from statsmodels.tsa.stattools import adfuller
result = adfuller(trainingSet['SalesAmount'])
print(result[1])  # p-value should be < 0.05


# %%
from statsmodels.tsa.arima.model import ARIMA
arima_model5 = ARIMA(trainingSet['SalesAmount'], order = (1,1,1))
arima_result5  = arima_model5.fit()
print(arima_result5.summary())



# %%
forecast5 = arima_result5.forecast(steps = 6)
forecast5.index = validationSet.index
d = pd.DataFrame({'Actual': validationSet['SalesAmount'], 'Forecast': forecast5})
print(d)

# %% [markdown]
# #### Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae5= mean_absolute_error(d['Actual'], d['Forecast'])
mse5= mean_squared_error(d['Actual'], d['Forecast'])
RMSE5= np.sqrt(mse2)   
MAPE5= mean_absolute_percentage_error(d['Actual'], d['Forecast'])

print(f"Mean Absolute Error (ARIMA(1,0,0)): {mae5:.2f}")
print(f"Mean Squared Error (ARIMA(1,0,0)): {mse5:.2f}")
print(f"Root Mean Squared Error (ARIMA(1,0,0)): {RMSE5:.2f}")
print(f"Mean Absolute Percentage Error (ARIMA(1,0,0)): {MAPE5:.2f}%")


# %%
import matplotlib.pyplot as plt
plt.plot(validationSet["SalesAmount"], label = 'Actual', marker = 'o')
plt.plot(forecast5, label = 'Forecast', marker = 'x')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ### 3.3 Implement SARIMA (Seasonal Autoregressive Integrated Moving Average) 

# %% [markdown]
# #### Check Trends through Decomposition

# %%
# Separates data into
    # Trend:A long-term increase or decrease in the data over time
    # Seasonality: A pattern that repeats at regular intervals (e.g., weekly, monthly, yearly).
    # Stationarity: A property of a time series where statistical properties (like mean and variance) remain constant over time.
        # tells you: “Can I rely on past data to predict the future, or is the game changing?”


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# ensure 'salesAmount' is a float type

trainingSet['SalesAmount'] = trainingSet['SalesAmount'].astype(float)

decomp = seasonal_decompose(trainingSet['SalesAmount'], model  = 'additive', period = 12)

# period=12 implies monthly seasonality (e.g., one full cycle per year if the data is monthly).

decomp.plot()

plt.suptitle("Time Series Decomposition of Sales Data of The Training Set - Before SARIMA")
plt.tight_layout()
plt.xticks(rotation = 90)
plt.show()

'''

1. Observed (Top Panel: "SalesAmount")
This is my original time series: raw monthly sales data (Jan 2021 – Jun 2023).
It fluctuates — there's clear seasonality, a growing trend, and some noise.

2. Trend
This line shows the underlying direction of the data by smoothing out seasonal and random effects.
We can observe a gentle upward movement in sales over time — likely due to the base sales increasing with time (+50 every month in your code).
The trend gets cut off at the start and end (because moving averages need several points to compute).

3. Seasonal
Captures repeating patterns that occur every 12 months.
In my case, the pattern is cyclical and symmetric, peaking around October–December and dipping around January–February — 
exactly as I intended with your custom sinusoidal seasonal components in the code.

SARIMA can model this repeating seasonality directly with the seasonal component (S in SARIMA).

4. Residual (or "Noise") - Residuals are the "what we couldn’t explain" part
Residuals are the unexpected changes — small bumps and dips that don’t follow any regular pattern.
What’s left after removing both trend and seasonality: ideally, random noise.
The residuals are relatively small and mostly hover around zero, suggesting my data is well-explained by trend + seasonality.
However, there are some spikes (e.g., near late 2022 or early 2023), which may hint at unusual events (like promotions or holidays

'''



# %% [markdown]
# #### Check Seasonality through ACF and PACF Plots - Before SARIMA

# %%
# PACF helps decide how many past values (lags) to use → this is AR part.
# ACF helps decide how many past errors to use → this is MA part


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


#ACF Plot
plot_acf(trainingSet["SalesAmount"], lags = 24) #  We are comparing the SalesAmount column to itself at different time shifts (lags).
# with lags=24, you're checking how well current sales correlate with sales from each of the past 24 months, one lag at a time.

plt.title("ACF Plot for Sales Amount of Training Set - Before SARIMA")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.show()


#PACF Plot

plot_pacf(trainingSet["SalesAmount"], lags = 15)
plt.title("PACF Plot for Sales Amount of Training Set - Before SARIMA")
plt.xlabel("Lags")
plt.ylabel("PACF")
plt.grid()
plt.show()


# How the comparison happens in the plot: Each vertical line (spike) on the ACF plot represents the calculated autocorrelation coefficient for a specific lag.
# The height of the line tells you the strength of the correlation, and its direction (above or below zero) tells you if it's positive or negative.

# %% [markdown]
# #### Check Stationarity through Dickey-Fuller Test - Before SARIMA

# %%
# p > 0.05 → Non-stationary;  Apply transformations: differencing, log, seasonal adjustment
# p <= 0.05 → Stationary; No transformations needed. Proceed with ARIMA, SARIMA, or other time series models

from statsmodels.tsa.stattools import adfuller

r = adfuller(trainingSet['SalesAmount'])

print("ADF Statistic:", r[0])
print("p-value: ", r[1])
print("Number of Lags used: ", r[2])
print("Critical Values:")

for key, value in r[4].items():
    print(f"   {key}: {value}")

'''
The combination of the ACF, PACF plots, and the ADF test provides a clear picture of the SalesAmount training set:

Trend: The slow decay in the ACF (even after considering seasonality) and the non-stationary result from the ADF test suggest an underlying trend.

Seasonality: Both ACF and PACF clearly show strong annual seasonality (spikes at lag 12 and its multiples). The presence of a strong spike at lag 12 in the PACF indicates a direct seasonal autoregressive component.

Non-Stationarity: The ADF test confirms that the series is non-stationary at a typical 5% significance level.

'''


# %% [markdown]
# #### Check Missing Values in the entire Training Set

# %%
trainingSet.isnull().sum()

# %% [markdown]
# #### Check Missing Values in the entire Sales Amount Column of Training Set

# %%
trainingSet["SalesAmount"].isnull().sum()

# %% [markdown]
# #### SARIMA (1,1,1)(1,1,1,12)

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

sarima_model = SARIMAX(trainingSet['SalesAmount'], order = (1,1,1), seasonal_order = (1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit()

print(sarima_result.summary())

# The SARIMA (1,1,1)(1,1,1,12) is technically running fine, but most coefficients are not statistically significant (p-values >> 0.05).
# This means the model is not very confident that these particular lags (both trend and seasonality) are contributing meaningfully to improving the prediction.
# Statistically, themodel is syntactically valid but weak in predictive power — possibly due to
    # Small dataset (only 30 observations)
    # Overfitting seasonal terms
    # Lack of external (exogenous) features like promotions or holidays





# %%
forecast_sarima = sarima_result.forecast(steps = 6)

forecast_sarima.index = validationSet.index

comparison = pd.DataFrame({'Actual': validationSet['SalesAmount'], 'Forecast': forecast_sarima})
print(comparison)

comparison.plot(title="SARIMA Forecast vs Actual Sales", figsize=(10,5))


# %% [markdown]
# #### Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_sarima = mean_absolute_error(comparison['Actual'], comparison['Forecast'])
mse_sarima = mean_squared_error(comparison['Actual'], comparison['Forecast'])
rmse_sarima = np.sqrt(mse_sarima)
mape_sarima = mean_absolute_percentage_error(comparison['Actual'], comparison['Forecast'])

print(f"Mean Absolute Error for SARIMA (1,1,1)(1,1,1,12): , {mae_sarima:.2f}")
print(f"Mean Squared Error for SARIMA (1,1,1)(1,1,1,12): , {mse_sarima:.2f}")
print(f"Root Mean Squared Error for SARIMA (1,1,1)(1,1,1,12): , {rmse_sarima:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) SARIMA (1,1,1)(1,1,1,12): {mape_sarima:.2f}")

# %% [markdown]
# #### SARIMA (1,1,1)(1,1,1,12) with exogenous features (promotion and holiday month)

# %%
exog_features = trainingSet[['Promotion', 'HolidayMonth']]
model1 = SARIMAX(trainingSet['SalesAmount'], 
                exog=exog_features, 
                order=(1,1,1), 
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False, 
                enforce_invertibility=False)
sarima_exog_result = model1.fit()
print(sarima_exog_result.summary())


# %%
# Create exogenous values for the forecast horizon (next 6 months)
future_exog = validationSet[['Promotion', 'HolidayMonth']]

# Forecast with future exogenous features
forecast_sarima_exog = sarima_exog_result.forecast(steps=6, exog=future_exog)

# Align forecast index
forecast_sarima_exog.index = validationSet.index

# Compare forecast with actual
comparison1 = pd.DataFrame({
    'Actual': validationSet['SalesAmount'],
    'Forecast': forecast_sarima_exog
})
print(comparison1)

# Plot
comparison1.plot(title="SARIMA (1,1,1)(1,1,1,12) with Exogenous - Forecast vs Actual Sales", figsize=(10,5))


# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_sarima_exog = mean_absolute_error(comparison1['Actual'], comparison1['Forecast'])
mse_sarima_exog = mean_squared_error(comparison1['Actual'], comparison1['Forecast'])
rmse_sarima_exog = np.sqrt(mse_sarima_exog)
mape_sarima_exog = mean_absolute_percentage_error(comparison1['Actual'], comparison1['Forecast'])

print(f"Mean Absolute Error for SARIMA exog (1,1,1)(1,1,1,12): {mae_sarima_exog:.2f}")
print(f"Mean Squared Error for SARIMA exog (1,1,1)(1,1,1,12): , {mse_sarima_exog:.2f}")
print(f"Root Mean Squared Error for SARIMA exog (1,1,1)(1,1,1,12): , {rmse_sarima_exog:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) SARIMA exog (1,1,1)(1,1,1,12): {mape_sarima_exog:.2f}")

# %% [markdown]
# ### SARIMA (1,1,2)(1,0,1,12) with exogenous features (promotion and holiday month)

# %%
exog_features2 = trainingSet[['Promotion', 'HolidayMonth']]
model2 = SARIMAX(trainingSet['SalesAmount'], exog = exog_features2, order = (1,1,2),
                  seasonal_order=(1,0,1,12), enforce_invertibility=False, enforce_stationarity=False) 

sarima_exog_result2 = model2.fit()
print(sarima_exog_result2.summary())

# %%
future_exog2 = validationSet[['Promotion', 'HolidayMonth']]

forecast_sarima_exog2 = sarima_exog_result2.forecast(steps = 6, exog = future_exog2)

forecast_sarima_exog2.index = validationSet.index


comparison2 = pd.DataFrame({'Actual' : validationSet['SalesAmount'], 'Forecast': forecast_sarima_exog2})

print(comparison2)

comparison2.plot(title = "SARIMA with Exogenous - Forecast vs Actual Sales - SARIMA (1,1,2)(1,0,1,12)")

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_sarima_exog2 = mean_absolute_error(comparison2['Actual'], comparison2['Forecast'])
mse_sarima_exog2 = mean_squared_error(comparison2['Actual'], comparison2['Forecast'])
rmse_sarima_exog2 = np.sqrt(mse_sarima_exog2)
mape_sarima_exog2 = mean_absolute_percentage_error(comparison2['Actual'], comparison2['Forecast'])

print(f"Mean Absolute Error for SARIMA exog (1,1,2)(1,0,1,12): {mae_sarima_exog2:.2f}")
print(f"Mean Squared Error for SARIMA exog (1,1,2)(1,0,1,12): , {mse_sarima_exog2:.2f}")
print(f"Root Mean Squared Error for SARIMA exog (1,1,2)(1,0,1,12): {rmse_sarima_exog2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) SARIMA exog (1,1,2)(1,0,1,12): {mape_sarima_exog2:.2f}%")



# %% [markdown]
# ### SARIMA (2,1,2)(0,1,0,12) with exogenous features (promotion and holiday month)

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

exog_features3 = trainingSet[['Promotion', 'HolidayMonth']]

model3 = SARIMAX(trainingSet['SalesAmount'], exog = exog_features, order = (2,1,2),
                  seasonal_order=(0,1,0,12), enforce_invertibility=False, enforce_stationarity=False)


sarima_exog_result3 = model3.fit()

print(sarima_exog_result3.summary())

# %%
future_exog3 = validationSet[['Promotion', 'HolidayMonth']]

forecast_sarima_exog3 = sarima_exog_result3.forecast(steps = 6, exog = future_exog3)

forecast_sarima_exog3.index = validationSet.index

comparison3 = pd.DataFrame({'Actual': validationSet['SalesAmount'], 'Forecast' : forecast_sarima_exog3})

print(comparison3)


comparison3.plot(title = "SARIMA (2,1,2)(0,1,0,12) with exogenous features (promotion and holiday month) ")
plt.show()

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_sarima_exog3 = mean_absolute_error(comparison3['Actual'], comparison3['Forecast'])
mse_sarima_exog3 = mean_squared_error(comparison3['Actual'], comparison3['Forecast'])
rmse_sarima_exog3 = np.sqrt(mse_sarima_exog3)
mape_sarima_exog3 = mean_absolute_percentage_error(comparison3['Actual'], comparison3['Forecast'])

print(f"Mean Absolute Error for SARIMA exog (2,1,2)(0,1,0,12): {mae_sarima_exog3:.2f}")
print(f"Mean Squared Error for SARIMA exog (2,1,2)(0,1,0,12): , {mse_sarima_exog3:.2f}")
print(f"Root Mean Squared Error for SARIMA exog (2,1,2)(0,1,0,12): , {rmse_sarima_exog3:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for SARIMA exog (2,1,2)(0,1,0,12): {mape_sarima_exog3:.2f}")


# %% [markdown]
# ### Prophet Implementation

# %% [markdown]
# ##### Pre reqs for Prophet
# ###### 1.	Proper Date-Time Formatting
# ###### Your dataset must have a datetime column in standard format (YYYY-MM-DD) named "ds".
# ######	The column representing the values to forecast must be named "y" (e.g., sales, revenue, demand).
#

# %%
sales_data

'''
Observations
Promotions Are Sparse: Only 6 promotional months out of 48 (12.5%), which is lower than the 20% expected from your generation code.
 This may weaken the statistical signal of the Promotion variable.

Holiday Effect Appears Real: December sales are consistently higher:

2020-12: 14,761

2021-12: 13,966

2022-12: 15,643

2023-12: 18,289

This supports holiday seasonality, even though your SARIMA models showed mixed results for HolidayMonth significance.

Trend + Seasonality: There's a mild upward trend and some seasonal patterns (e.g., dips around summer, spikes around winter), aligning with your earlier synthetic generation logic.

'''

# %%
sales_data.info()

# %%
import pandas as pd
import numpy as np
from prophet import Prophet


df_prophet = sales_data.reset_index()[['Date', 'SalesAmount', 'Promotion', 'HolidayMonth']].copy()

# .copy() creates a new DataFrame that is a copy of the original, so any changes made to df_prophet won't affect sales_data.

# Your dataset must have a datetime column in standard format (YYYY-MM-DD) named "ds".
# Prophet doesn't require a DataFrame with datetime as the index. 
# It just needs a column named ds with dates, so we first flatten the DataFrame.

df_prophet.rename(columns = {'Date':'ds', 'SalesAmount':'y'}, inplace = True)
# inplace=True modifies the original DataFrame directly, renaming the columns to match Prophet's requirements:

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
# This line ensures that the "ds" column is in datetime format (YYYY-MM-DD) — essential for time-based modeling.

# Define Holiday Calendar
# Only December is a "holiday month" in my context
# Defining a holiday calendar in Prophet is recommended when you want Prophet’s internal holiday modeling features.
# A regressor with the same info can be used as a shortcut, but it does not fully replace the built-in holiday mechanism


# Prophet’s holiday calendar is a built-in special mechanism that:
# Treats holidays as categorical events with possible multiple days effects (via lower_window and upper_window).
# enables Prophet’s built-in holiday effect machinery

december_holidays = pd.DataFrame({
    'holiday': 'december_bump',
    'ds': df_prophet[df_prophet['ds'].dt.month == 12]['ds'],
    'lower_window': 0,
    'upper_window': 0
})

'''
lower_window: Number of days before the holiday date to include in the holiday effect.
upper_window: Number of days after the holiday date to include in the holiday effect.

In my code, both are set to 0, meaning the holiday effect applies only on the exact date(s) in December, with no extension before or after.

'''


# Outlier Detection using z-score method
# The z-score (also known as the standard score) measures how many standard deviations a data point is from the mean of a dataset.
# It tells you whether a value is typical or unusual compared to the rest of the data
# a z-score of:
#       0 means the value is exactly at the mean.
#      +1 means it’s 1 standard deviation above the mean.
#      −2 means it’s 2 standard deviations below the mean.

from scipy.stats import zscore
df_prophet['z_score'] = zscore(df_prophet['y'])
outliers = df_prophet[np.abs(df_prophet['z_score']) > 3] # np.abs() — Returns the absolute value of a number or array of numbers. The absolute value of a number is its distance from zero — without considering direction.
# This line filters the DataFrame to find rows where the absolute z-score is greater than 3, indicating potential outliers.
# A z-score greater than 3 or less than -3 is often considered an outlier in many statistical analyses.

# print("Outliers detected based on z-score: \n", outliers[['ds', 'y', 'z_score']])
print("Outliers detected based on z-score:", len(outliers))

# z-score column is not needed for Prophet, so we can drop it
df_prophet.drop(columns = ['z_score'], inplace = True)

prophet_model = Prophet(
    yearly_seasonality = False,
    weekly_seasonality = False,
    daily_seasonality = False,
    holidays = december_holidays,
    seasonality_mode= 'additive').add_seasonality(name = 'yearly_custom', period = 12, fourier_order = 5)

# This initializes a Prophet model with no built-in seasonality (yearly, weekly, daily) but adds a custom yearly seasonality with a 12-month period.
# Because the data is monthly, we want to capture yearly patterns without the default daily/weekly noise.

prophet_model.add_regressor('Promotion')
prophet_model.add_regressor('HolidayMonth')




# %%
print(df_prophet)

# %%
print(december_holidays)

# %% [markdown]
# #### Train Prophet - 1st Attempt - Seasonality Mode: Additive - Fourier_order = 5

# %%
train_df = df_prophet.iloc[:42].copy() # Selects the first 42 rows of df_prophet as the training set (likely Jan 2020 to June 2023).
evaluate_df = df_prophet[['ds', 'Promotion', 'HolidayMonth']].iloc[42:].copy() # Selects the last 6 rows (row 42 to end), i.e., July to December 2023, as the evaluation (future) set.

prophet_model.fit(train_df)

future = evaluate_df.copy()

forecast = prophet_model.predict(future)
forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Selects only the most relevant columns from the full forecast:
    # ds: date
    # yhat: predicted sales
    # yhat_lower, yhat_upper: lower and upper bounds of the forecast interval (confidence range).

comparison_prophet= pd.concat([
    df_prophet[['ds', 'y']].iloc[42:].reset_index(drop=True),
    forecast[['yhat']].round(2)
], axis=1)

comparison_prophet.columns = ['ds', 'Actual', 'Forecast']



print(comparison_prophet)
# Here’s what really happened (Actual), and here’s what my model thought would happen (Forecast).”
# It's the final scoreboard for evaluating how well Prophet predicted reality

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_prophet = mean_absolute_error(comparison_prophet['Actual'], comparison_prophet['Forecast'])
mse_prophet = mean_squared_error(comparison_prophet['Actual'], comparison_prophet['Forecast'])
rmse_prophet = np.sqrt(mse_prophet)
mape_prophet = mean_absolute_percentage_error(comparison_prophet['Actual'], comparison_prophet['Forecast'])

print(f"Mean Absolute Error for Prophet: {mae_prophet:.2f}")
print(f"Mean Squared Error for Prophet: {mse_prophet:.2f}")
print(f"Root Mean Squared Error for Prophet: {rmse_prophet:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for Prophet: {mape_prophet:.2f}%")




# %%
# comparison_prophet.plot(title="Prophet Forecast vs Actual Sales", x = 'ds', y = ,figsize=(10,5))
comparison_prophet.plot(title="Prophet Forecast vs Actual Sales", x='ds', y=['Actual', 'Forecast'], figsize=(10, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend(["Actual Sales", "Forecasted Sales"])
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.title("Prophet Forecast vs Actual Sales")
plt.show()

# %% [markdown]
# #### Train Prophet - 2nd Attempt - Seasonality Mode: Additive - Fourier order = 10

# %%
import pandas as pd
import numpy as np
from prophet import Prophet


df_prophet2 = sales_data.reset_index()[['Date', 'SalesAmount', 'Promotion', 'HolidayMonth']].copy()

df_prophet2.rename(columns = {'Date':'ds', 'SalesAmount':'y'}, inplace = True)

df_prophet2['ds'] = pd.to_datetime(df_prophet2['ds'])

december_holidays2 = pd.DataFrame({
    'holiday': 'december_bump',
    'ds': df_prophet2[df_prophet2['ds'].dt.month == 12]['ds'],
    'lower_window': 0,
    'upper_window': 0
})

from scipy.stats import zscore
df_prophet2['z_score'] = zscore(df_prophet2['y'])
outliers2 = df_prophet2[np.abs(df_prophet2['z_score']) > 3] 
print("Outliers detected based on z-score:", len(outliers2))


prophet_model2 = Prophet(
    yearly_seasonality = False,
    weekly_seasonality = False,
    daily_seasonality = False,
    holidays = december_holidays2,
    seasonality_mode= 'additive').add_seasonality(name = 'yearly_custom', period = 12, fourier_order = 10)


prophet_model2.add_regressor('Promotion')
prophet_model2.add_regressor('HolidayMonth')


train_df2 = df_prophet2.iloc[:42].copy() # Selects the first 42 rows of df_prophet as the training set (likely Jan 2020 to June 2023).
evaluate_df2 = df_prophet2[['ds', 'Promotion', 'HolidayMonth']].iloc[42:].copy() # Selects the last 6 rows (row 42 to end), i.e., July to December 2023, as the evaluation (future) set.
prophet_model2.fit(train_df2)

future2 = evaluate_df2.copy()

forecast2 = prophet_model2.predict(future)
forecast2 = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


comparison_prophet2= pd.concat([
    df_prophet2[['ds', 'y']].iloc[42:].reset_index(drop=True),
    forecast2[['yhat']].round(2)
], axis=1)

comparison_prophet2.columns = ['ds', 'Actual', 'Forecast']
print(comparison_prophet2)



# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_prophet2 = mean_absolute_error(comparison_prophet2['Actual'], comparison_prophet2['Forecast'])
mse_prophet2 = mean_squared_error(comparison_prophet2['Actual'], comparison_prophet2['Forecast'])
rmse_prophet2 = np.sqrt(mse_prophet2)
mape_prophet2 = mean_absolute_percentage_error(comparison_prophet2['Actual'], comparison_prophet2['Forecast'])

print(f"Mean Absolute Error for Prophet: {mae_prophet2:.2f}")
print(f"Mean Squared Error for Prophet: {mse_prophet2:.2f}")
print(f"Root Mean Squared Error for Prophet: {rmse_prophet2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for Prophet: {mape_prophet2:.2f}%")


# comparison_prophet.plot(title="Prophet Forecast vs Actual Sales", x = 'ds', y = ,figsize=(10,5))
comparison_prophet2.plot(title="Prophet Forecast vs Actual Sales", x='ds', y=['Actual', 'Forecast'], figsize=(10, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend(["Actual Sales", "Forecasted Sales"])
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.title("Prophet Forecast vs Actual Sales")
plt.show()


# %% [markdown]
# #### Train Prophet - 3rd Attempt - Seasonality Mode: Multiplicative - Fourier order = 5

# %%
import pandas as pd
import numpy as np
from prophet import Prophet


df_prophet3 = sales_data.reset_index()[['Date', 'SalesAmount', 'Promotion', 'HolidayMonth']].copy()

df_prophet3.rename(columns = {'Date':'ds', 'SalesAmount':'y'}, inplace = True)

df_prophet3['ds'] = pd.to_datetime(df_prophet3['ds'])

december_holidays3 = pd.DataFrame({
    'holiday': 'december_bump',
    'ds': df_prophet3[df_prophet3['ds'].dt.month == 12]['ds'],
    'lower_window': 0,
    'upper_window': 0
})

from scipy.stats import zscore
df_prophet3['z_score'] = zscore(df_prophet3['y'])
outliers3 = df_prophet3[np.abs(df_prophet3['z_score']) > 3] 
print("Outliers detected based on z-score:", len(outliers3))


prophet_model3 = Prophet(
    yearly_seasonality = False,
    weekly_seasonality = False,
    daily_seasonality = False,
    holidays = december_holidays3,
    seasonality_mode= 'multiplicative').add_seasonality(name = 'yearly_custom', period = 12, fourier_order = 5)


prophet_model3.add_regressor('Promotion')
prophet_model3.add_regressor('HolidayMonth')


train_df3 = df_prophet3.iloc[:42].copy() # Selects the first 42 rows of df_prophet as the training set (likely Jan 2020 to June 2023).
evaluate_df3 = df_prophet3[['ds', 'Promotion', 'HolidayMonth']].iloc[42:].copy() # Selects the last 6 rows (row 42 to end), i.e., July to December 2023, as the evaluation (future) set.
prophet_model3.fit(train_df3)

future3 = evaluate_df3.copy()

forecast3 = prophet_model3.predict(future)
forecast3 = forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


comparison_prophet3= pd.concat([
    df_prophet3[['ds', 'y']].iloc[42:].reset_index(drop=True),
    forecast3[['yhat']].round(2)
], axis=1)

comparison_prophet3.columns = ['ds', 'Actual', 'Forecast']
print(comparison_prophet3)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_prophet3 = mean_absolute_error(comparison_prophet3['Actual'], comparison_prophet3['Forecast'])
mse_prophet3 = mean_squared_error(comparison_prophet3['Actual'], comparison_prophet3['Forecast'])
rmse_prophet3 = np.sqrt(mse_prophet3)
mape_prophet3 = mean_absolute_percentage_error(comparison_prophet3['Actual'], comparison_prophet3['Forecast'])

print(f"Mean Absolute Error for Prophet: {mae_prophet3:.2f}")
print(f"Mean Squared Error for Prophet: {mse_prophet3:.2f}")
print(f"Root Mean Squared Error for Prophet: {rmse_prophet3:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for Prophet: {mape_prophet3:.2f}%")


# comparison_prophet.plot(title="Prophet Forecast vs Actual Sales", x = 'ds', y = ,figsize=(10,5))
comparison_prophet3.plot(title="Prophet Forecast vs Actual Sales", x='ds', y=['Actual', 'Forecast'], figsize=(10, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend(["Actual Sales", "Forecasted Sales"])
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.title("Prophet Forecast vs Actual Sales")
plt.show()

# %% [markdown]
# #### Train Prophet - 4th Attempt - Seasonality Mode: Multiplicative - Fourier order = 10

# %%
import pandas as pd
import numpy as np
from prophet import Prophet


df_prophet4 = sales_data.reset_index()[['Date', 'SalesAmount', 'Promotion', 'HolidayMonth']].copy()

df_prophet4.rename(columns = {'Date':'ds', 'SalesAmount':'y'}, inplace = True)

df_prophet4['ds'] = pd.to_datetime(df_prophet4['ds'])

december_holidays4 = pd.DataFrame({
    'holiday': 'december_bump',
    'ds': df_prophet4[df_prophet4['ds'].dt.month == 12]['ds'],
    'lower_window': 0,
    'upper_window': 0
})

from scipy.stats import zscore
df_prophet4['z_score'] = zscore(df_prophet4['y'])
outliers4 = df_prophet4[np.abs(df_prophet4['z_score']) > 3] 
print("Outliers detected based on z-score:", len(outliers4))


prophet_model4 = Prophet(
    yearly_seasonality = False,
    weekly_seasonality = False,
    daily_seasonality = False,
    holidays = december_holidays4,
    seasonality_mode= 'multiplicative').add_seasonality(name = 'yearly_custom', period = 12, fourier_order = 10)


prophet_model4.add_regressor('Promotion')
prophet_model4.add_regressor('HolidayMonth')


train_df4 = df_prophet4.iloc[:42].copy() # Selects the first 42 rows of df_prophet as the training set (likely Jan 2020 to June 2023).
evaluate_df4 = df_prophet4[['ds', 'Promotion', 'HolidayMonth']].iloc[42:].copy() # Selects the last 6 rows (row 42 to end), i.e., July to December 2023, as the evaluation (future) set.
prophet_model4.fit(train_df4)

future4 = evaluate_df4.copy()

forecast4 = prophet_model4.predict(future)
forecast4 = forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


comparison_prophet4= pd.concat([
    df_prophet4[['ds', 'y']].iloc[42:].reset_index(drop=True),
    forecast4[['yhat']].round(2)
], axis=1)

comparison_prophet4.columns = ['ds', 'Actual', 'Forecast']
print(comparison_prophet4)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_prophet4 = mean_absolute_error(comparison_prophet4['Actual'], comparison_prophet4['Forecast'])
mse_prophet4 = mean_squared_error(comparison_prophet4['Actual'], comparison_prophet4['Forecast'])
rmse_prophet4 = np.sqrt(mse_prophet4)
mape_prophet4 = mean_absolute_percentage_error(comparison_prophet4['Actual'], comparison_prophet4['Forecast'])

print(f"Mean Absolute Error for Prophet: {mae_prophet4:.2f}")
print(f"Mean Squared Error for Prophet: {mse_prophet4:.2f}")
print(f"Root Mean Squared Error for Prophet: {rmse_prophet4:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for Prophet: {mape_prophet4:.2f}%")


# comparison_prophet.plot(title="Prophet Forecast vs Actual Sales", x = 'ds', y = ,figsize=(10,5))
comparison_prophet4.plot(title="Prophet Forecast vs Actual Sales", x='ds', y=['Actual', 'Forecast'], figsize=(10, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend(["Actual Sales", "Forecasted Sales"])
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.title("Prophet Forecast vs Actual Sales")
plt.show()

# %% [markdown]
# #### Train Prophet - 4th Attempt - Stick with Additive + Fourier=5: It gave the best MAE, RMSE, and MAPE.

# %% [markdown]
# #### Experimenting with change points - changepoint_prior_scale = VARYING (BY HANNAN)

# %%
import pandas as pd
import numpy as np
from prophet import Prophet


df_prophet5 = sales_data.reset_index()[['Date', 'SalesAmount', 'Promotion', 'HolidayMonth']].copy()

df_prophet5.rename(columns = {'Date':'ds', 'SalesAmount':'y'}, inplace = True)

df_prophet5['ds'] = pd.to_datetime(df_prophet5['ds'])

december_holidays5 = pd.DataFrame({
    'holiday': 'december_bump',
    'ds': df_prophet5[df_prophet5['ds'].dt.month == 12]['ds'],
    'lower_window': 0,
    'upper_window': 0
})

from scipy.stats import zscore
df_prophet5['z_score'] = zscore(df_prophet5['y'])
outliers5 = df_prophet5[np.abs(df_prophet5['z_score']) > 3] 
print("Outliers detected based on z-score:", len(outliers4))


# Adding manual change points to the Prophet model
# Change points are specific dates where the model can adjust its trend.
manual_change_points = ['2020-10-01', '2020-11-01', '2020-12-01', '2021-10-01', '2021-11-01', '2021-12-01', '2022-10-01', '2022-11-01', '2022-12-01']

prophet_model5 = Prophet(
    changepoints = manual_change_points,
    changepoint_prior_scale = 0.5,
    yearly_seasonality = False,
    weekly_seasonality = False,
    daily_seasonality = False,
    holidays = december_holidays4,
    seasonality_mode= 'additive').add_seasonality(name = 'yearly_custom', period = 12, fourier_order = 5)


prophet_model5.add_regressor('Promotion')
prophet_model5.add_regressor('HolidayMonth')


train_df5 = df_prophet5.iloc[:42].copy() # Selects the first 42 rows of df_prophet as the training set (likely Jan 2020 to June 2023).
evaluate_df5 = df_prophet5[['ds', 'Promotion', 'HolidayMonth']].iloc[42:].copy() # Selects the last 6 rows (row 42 to end), i.e., July to December 2023, as the evaluation (future) set.
prophet_model5.fit(train_df5)

future5 = evaluate_df5.copy()

forecast5 = prophet_model5.predict(future)
forecast5 = forecast5[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


comparison_prophet5= pd.concat([
    df_prophet5[['ds', 'y']].iloc[42:].reset_index(drop=True),
    forecast5[['yhat']].round(2)
], axis=1)

comparison_prophet5.columns = ['ds', 'Actual', 'Forecast']
print(comparison_prophet5)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

mae_prophet5 = mean_absolute_error(comparison_prophet5['Actual'], comparison_prophet5['Forecast'])
mse_prophet5 = mean_squared_error(comparison_prophet5['Actual'], comparison_prophet5['Forecast'])
rmse_prophet5 = np.sqrt(mse_prophet5)
mape_prophet5 = mean_absolute_percentage_error(comparison_prophet5['Actual'], comparison_prophet5['Forecast'])

print(f"Mean Absolute Error for Prophet: {mae_prophet5:.2f}")
print(f"Mean Squared Error for Prophet: {mse_prophet5:.2f}")
print(f"Root Mean Squared Error for Prophet: {rmse_prophet5:.2f}")
print(f"Mean Absolute Percentage Error (MAPE) for Prophet: {mape_prophet5:.2f}%")


# comparison_prophet.plot(title="Prophet Forecast vs Actual Sales", x = 'ds', y = ,figsize=(10,5))
comparison_prophet5.plot(title="Prophet Forecast vs Actual Sales", x='ds', y=['Actual', 'Forecast'], figsize=(10, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend(["Actual Sales", "Forecasted Sales"])
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.title("Prophet Forecast vs Actual Sales")
plt.show()

# %% [markdown]
# # 4 - Selecting  SARIMA(2,1,2)(0,1,0,12) + exog  the best model and retrain it on the entire historical dataset.

# %% [markdown]
# ####  SARIMA(2,1,2)(0,1,0,12) + exog  on the entire historical dataset i.e all 48 months of historical data (Jan 2020 – Dec 2023). 
# #### use this fully trained model to forecast the next 6 months (Jan 2024 – Jun 2024), ideally including confidence intervals

# %%
sales_data

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

exog_features_final_SARIMA = sales_data[['Promotion', 'HolidayMonth']]

final_SARIMA_model = SARIMAX(sales_data['SalesAmount'], exog = exog_features_final_SARIMA, order = (2,1,2), seasonal_order = (0,1,0,12),
                             enforce_invertibility = False, enforce_stationarity =False)

final_SARIMA_result = final_SARIMA_model.fit()

print(final_SARIMA_result.summary())




# %%
future_exog_dates= pd.date_range(start = '2024-01-01', periods = 6, freq = 'MS')

future_exog_data = pd.DataFrame({
    'Promotion': [0,0,1,0,1,0], # Example pattern: Promotions in Mar and May
    'HolidayMonth': [1, 0, 0, 0, 0, 0] # Example: January is a holiday month
}, index = future_exog_dates)

forecast_final_SARIMA = final_SARIMA_result.get_forecast(steps=6, exog=future_exog_data)
forecast_ci = forecast_final_SARIMA.conf_int()
forecast_mean = forecast_final_SARIMA.predicted_mean

comparison_final_SARIMA = pd.DataFrame({
    'Forecast': forecast_mean,
    'Lower CI': forecast_ci.iloc[:, 0],
    'Upper CI': forecast_ci.iloc[:, 1]
}, index=future_exog_dates)

print(comparison_final_SARIMA)


comparison_final_SARIMA['Forecast'].plot(title="Final SARIMA Forecast vs Actual Sales", figsize=(10,5), marker='o')
import matplotlib.pyplot as plt
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend(["Forecast", "Lower CI", "Upper CI"])
plt.grid()
plt.fill_between(comparison_final_SARIMA.index,
                 comparison_final_SARIMA['Lower CI'],
                 comparison_final_SARIMA['Upper CI'],
                 color='skyblue', alpha=0.3, label='95% Confidence Interval')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# %% [markdown]
# #### Visualize the forecasts against historical data.

# %%
forecast_mean
forecast_ci

plt.plot(sales_data.index, sales_data['SalesAmount'], label = "Historical Sales", marker = 'o')
plt.plot(forecast_mean.index, forecast_mean.values, label = "Forecasted Sales", marker = 'o', color='orange')
plt.fill_between(conf_int.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='skyblue', alpha=0.3, label='95% Confidence Interval')

plt.axvline(pd.to_datetime('2023-12-31'),color = 'red', linestyle = "--", label = "Forecast Start")

plt.title("Final SARIMA Forecast vs Historical Sales")
plt.xlabel("Date")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# %% [markdown]
# ##### The final sales forecast data (e.g., in a CSV file).

# %%
# The final sales forecast data (e.g., in a CSV file) downloadable

comparison_final_SARIMA.to_csv('final_sarima_forecast.csv') 

