# Retail Sales Forecasting: Beating the Holiday Rush with SARIMAX

I built this project to tackle a classic retail problem: how to accurately **forecast sales**. Simple models just don't work when you have to account for holiday sales, marketing promos, and long-term growth all at once.

My goal was to create a tool that could actually be used by an operations or marketing team. The final model, a fine-tuned **SARIMAX**, doesn't just guess— it **quantifies the impact of promotions** and gives a clear picture of the months ahead, helping teams optimize inventory and stop losing money.

---

## Table of Contents

* [The Business Problem: Why This Matters](#the-business-problem-why-this-matters)
* [Tech Stack & Installation](#tech-stack-installation)
* [How to Run](#how-to-run)
* [The Winning Model & Final Forecast](#the-winning-model-final-forecast)
* [Summary of Analysis & Model Comparison](#summary-of-analysis-model-comparison)
* [Project Capabilities](#project-capabilities)
---

## The Business Problem: Why This Matters

Every retailer knows the pain of getting the forecast wrong. It's a constant balancing act that leads to two costly mistakes:

* **Overstocking:** You buy too much, and your cash gets tied up in products that will eventually be sold at a steep discount.
* **Stockouts:** You buy too little, miss out on sales during your busiest season, and send customers to your competitors.

This project was all about building a **reliable solution** to avoid those problems.

---

## 🛠️ Tech Stack & Installation
<br>

* **Python 3.12.0**
* **pandas:** For all the data wrangling and time series
* **numpy:** For numerical operations and building the dataset.
* **statsmodels:** For time series analysis (ADF, STL, ACF/PACF) , ARIMA, **SARIMAX model**.
* **Scikit-learn** For evaluation metrics
* **Prophet:** For building the Prophet comparison model.
* **matplotlib:** For all the visualizations.
* **missingno:** For a quick visual check on missing data.

To set up, I recommend using a virtual environment.

1.  Clone this repository:
    ```bash
    git clone [https://github.com/hannanbaig347/Forecasting-for-a-Retail-Store](https://github.com/hannanbaig347/Forecasting-for-a-Retail-Store)
    cd your-repo-name
    ```
2.  Create and activate a virtual environment:
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    venv\Scripts\activate
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```


---

## How to Run

The entire analysis, from data generation to final model, is contained in one Jupyter Notebook.

1.  Make sure your virtual environment is activated.
2.  Launch Jupyter Notebook from your terminal:
    ```bash
    jupyter notebook
    ```
3.  In the browser window that opens, click on **`Forecasting_for_a_Retail_Store.ipynb`**.
4.  You can run all cells from top to bottom to reproduce the full analysis, see the model comparisons, and generate the final forecast.

---
---

## The Winning Model & Final Forecast

The final, best-performing model was retrained on the entire 4-year dataset to make a 6-month forecast (Jan 2024 - June 2024). This forecast isn't just a single number; it's a guide for managing risk.

By inputting a planned promotion schedule, the model delivers a 95% confidence interval (CI) that translates directly into business actions:

* **Actionable Inventory Planning:** For a planned March promotion, the model predicts **16,759** in sales but provides a CI of **[15,182, 18,338]**. This tells an operations manager the *exact range* to plan for, providing a clear, data-driven margin for safety stock.
* **Preserving Cash Flow:** The forecast shows an expected dip in June (predicting **9,698** sales). This is a strong signal to the purchasing department to strategically decrease orders and conserve working capital during a known slow month.

This model moves the business from reactive "gut-feel" decisions to **proactive, data-driven planning.**

---

## Summary of Analysis & Model Comparison

Before landing on the winner, I followed a structured process. (The full, detailed exploration is in the notebook and the PDF report — this is just the summary).

### 1. Data & EDA
I generated a 4-year synthetic dataset (`retail_sales_mock_data.csv`) that realistically includes:
* A steady upward **trend** (company growth).
* Strong **12-month seasonality** (the holiday rush).
* Binary flags for **Promotions** and **Holidays**.

Exploratory Data Analysis (using STL decomposition and the ADF test) confirmed these patterns and showed the data was **stationary**, meaning it was ready for modeling without differencing.

### 2. Feature Engineering
To give the model historical context, I engineered several **lag features** (sales from 1, 2, 3, 6, and 12 months prior).

### 3. Head-to-Head Model Comparison
I tested three classes of models to find the one that could best handle seasonality *and* external factors.

| Model | Best Parameters | Best Metric (RMSE) | My Takeaway |
| :--- | :--- | :--- | :--- |
| ARIMA | (Multiple) | ~2,700 - 2800 | **Total failure.** Completely blind to seasonality and useless for this problem. |
| Prophet | (Tuned) | 2,096.18 | Good and interpretable, but not the most accurate. |
| **SARIMAX** | **(2, 1, 2)(0, 1, 0, 12) + Exog** | **1,017.34** | **WINNER.** The combination of seasonal components ($S$) and exogenous features ($X$) made it the most accurate by a wide margin. |

The full notebook shows the experimentation, but the conclusion is clear: **SARIMAX** was the only model capable of accurately capturing all the complex drivers of retail sales.

---

## ✨ Project Capabilities

* **Analyzes the Full Picture:** Breaks down sales data to see the real trend, the 12-month seasonal patterns, and the impact of business decisions.
* **Quantifies What Matters:** Measures the exact sales lift from things like marketing promotions.
* **Compares Models Head-to-Head:** Tests ARIMA, SARIMAX, and Prophet to find the undisputed winner.
* **Delivers Actionable Forecasts:** Gives a 6-month forecast with a 95% confidence range, providing a clear guide for managing risk.
* **Includes the Data:** The synthetic dataset and the data generation script are all in the notebook.
