# Retail Sales Forecasting: Beating the Holiday Rush with SARIMAX

I built this project to tackle a classic, retail problem: how to accurately **forecast sales**. Simple models just don't cut it when you have to account for wild holiday swings, marketing promos, and long-term growth all at once.

My goal was to create a tool that could actually be used by an operations or marketing team. The final model, a fine-tuned **SARIMAX**, doesn't just guess—it **quantifies the impact of promotions** and gives a clear picture of the months ahead, helping teams optimize inventory and stop losing money.

---

## The Business Problem: Why This Matters

Every retailer knows the pain of getting the forecast wrong. It's a constant balancing act that leads to two costly mistakes:

* **Overstocking:** You buy too much, and your cash gets tied up in products that will eventually be sold at a steep discount.
* **Stockouts:** You buy too little, miss out on sales during your busiest season, and send customers to your competitors.

This project was all about building a **reliable solution** to avoid those problems.

---

##  What This Project Can Do

* **Analyzes the Full Picture:** It breaks down the sales data to see the real trend, the 12-month seasonal patterns, and the impact of business decisions.
* **Quantifies What Matters:** It measures the exact sales lift from things like **marketing promotions**.
* **Compares Models Head-to-Head:** I didn't just pick one model. I tested **ARIMA, SARIMAX, and Prophet** to find the undisputed winner.
* **Delivers Actionable Forecasts:** The final model gives a **6-month forecast with a 95% confidence range**, so you're not just getting a single number—you're getting a clear guide for managing risk.
* **Includes the Data:** I've included the Python script I used to generate the realistic 4-year (48-month) **synthetic dataset**, complete with trends, seasonality, and promo flags.

---

## My Workflow: From Data to Forecast

I followed a structured process to make sure the model was built on a solid foundation.

### 1. Data Generation & Prep

First, I needed good data. I built a 4-year synthetic dataset from scratch using `pandas` and `numpy`. This let me create a realistic baseline (`retail_sales_mock_data.csv`) that included:

* A steady **upward trend** through "base_sales = 10000 + np.arange(num_months) * 50" (depicts the company growth).
* Strong **12-month seasonality** (the holiday rush).
* Binary flags for **Promotions** and **Holidays**.
* A bit of random **noise** through np.random.normal (like tiny unpredictable changes in sales such as weather, supply hiccups, customer behavior).

After generating it, I loaded, cleaned, and set up the `Date` index for time series analysis.

### 2. Exploratory Data Analysis (EDA)

Before trying to model anything, I had to understand the data.
* **![Visualizing Sales over Time](figures/Visualizing_Sales_over_Time.png)**
* **Decomposition :** To properly understand the sales data, I needed to break it down into its main components:
* 1. **Trend**: A long-term increase or decrease in the data over time,
  2. **Seasonality**: A pattern that repeats at regular intervals (e.g., weekly, monthly, yearly)
  3. **Residuals**: The leftover noise

* **![Classical Decomposition Plot](figures/Classical_Decomposition_Plot.png)**

* **![ STL (Seasonal and Trend decomposition using Loess) Decomposition](figures/STL_Decomposition.png)**.

* I generated two decomposition plots (shown above) —one classical and STL method. Both plots showed a consistent upward trend, which suggests that my sales are experiencing steady, long-term growth. Critically, the seasonal component in both decompositions revealed significant, repeating yearly patterns, peaking strongly towards the end of the year which is entirely typical for retail holiday rushes. The residuals from the STL decomposition looked even cleaner, appearing randomly distributed around zero. **This robustness of the STL method is important because it confirms the model effectively captured the trend and seasonality, leaving less unexplained noise**

  
* **Stationarity (ADF Test):** I ran the Augmented Dickey-Fuller test to see if the data was stable or if it needed transformations. This confirmed I'd need **differencing** ($d=1$, $D=1$) to get it ready for modeling.
* **Autocorrelation (ACF/PACF):** These plots were my guide for picking the initial AR and MA parameters for the models.

### 3. The Model Gauntlet: Finding a Winner

I split the data into a **30-month training set** and a **6-month validation set** to see which model could perform best on unseen data.

* **Attempt 1: Basic ARIMA**
    * *Result:* A total failure. I tried multiple ARIMA setups, and they all fell flat. They were completely **blind to the 12-month seasonality** and couldn't predict the December sales spike to save their lives. This proved non-seasonal models were useless for this problem.

* **Attempt 2: Facebook Prophet**
    * *Result:* Much better. Prophet is great with holidays, so I gave it a shot. By tuning its `changepoint_prior_scale` and even setting manual changepoints for Oct-Dec, I got a decent model with an **RMSE of 2,027.45**. It's easy to read, but the accuracy still wasn't where I needed it to be.

* **Attempt 3: SARIMAX (The Champion)**
    * *Result:* This is where the magic happened. SARIMAX is built for this. It combines seasonal components ($S$) with the ability to add **external regressors** ($X$). By feeding it the **Promotion** and **HolidayMonth** flags, the model could finally see the *why* behind the numbers.
    * The final tuned model, **SARIMAX(2, 1, 2)(0, 1, 0, 12) + Exogenous**, blew the others away.

---

## 🏆 Model Performance: The Final Tally

The results speak for themselves. The SARIMAX model wasn't just a little better—it cut the error in half compared to the next best option.

| Model | Best Parameters | Best Metric (RMSE) | My Takeaway |
| :--- | :--- | :--- | :--- |
| ARIMA | (Multiple) | ~2,700 - 3,600 | Inadequate. Fails to model seasonality. |
| Prophet | Manual Changepoints, prior_scale=0.5 | 2,027.45 | Good and interpretable, but not the most accurate. |
| **SARIMAX** | **(2, 1, 2)(0, 1, 0, 12) + Exogenous** | **1,017.34** | **WINNER.** Best accuracy & it explains why. |

---

## 🚀 The Payoff: What This Model Delivers

After finding the winner, I retrained the SARIMAX model on the entire 48-month dataset to make it as smart as possible. This final model provides two huge advantages for a business:

* **You Can Finally Measure Marketing:** The model learned the real impact of our promos. The final summary showed that a single promotion adds **+2,457 units to sales** in a given month. The marketing team can now calculate the **precise ROI** for their campaigns.
* **No More Guesswork for Inventory:** I used the model to generate a **6-month rolling forecast**. It gives the operations team a **95% confidence interval**, which is huge. Instead of one number, they get a practical range for planning. This means they can confidently order stock, knowing they've minimized the risk of stockouts and overstocking.

This project provides a clear path from reactive, gut-feel decisions to **proactive, data-driven planning.**

---

## 🛠️ Tech & Libraries Used

* **Python 3.x**
* **pandas:** For all the data wrangling and time series magic.
* **numpy:** For numerical operations and building the dataset.
* **statsmodels:** The powerhouse for time series analysis (ADF, STL, ACF/PACF) and the **SARIMAX model**.
* **prophet:** For building the Prophet comparison model.
* **matplotlib:** For all the visualizations.
* **missingno:** For a quick visual check on missing data.

---

## 📂 Check out the work:

* The main analysis and model building is in the **Retail_Forecasting_Analysis.ipynb** notebook.
* The raw data I generated is **retail_sales_mock_data.csv**.
* The data with engineered lag features is **retail_sales_with_lags.csv**.
