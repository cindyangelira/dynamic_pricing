import streamlit as st
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import datetime
from typing import List, TypedDict, NamedTuple, DefaultDict
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import gamma

# # Define FASTAPI app
# app = FastAPI()

# # Define a class for the request body
# class PriceParams(BaseModel):
#     price: float
#     alpha: float
#     beta: float
       
# Streamlit
def main():
    st.title("Price Optimization")
    st.markdown("This is a simple price optimization app built with Streamlit. The app uses the Beta Geometric Negative Binomial Distribution to model the customer purchase behavior. The app then uses the model to find the optimal price that maximizes the revenue.")
    def load_data():    
        df = pd.read_csv("data/data.csv", parse_dates=['InvoiceDate'], encoding='latin1')
        # filter out StockCode that appears < 10 times
        vc = df['StockCode'].value_counts()
        df = df[df['StockCode'].isin(vc[vc > 50].index)]
        df.sort_values(by=['CustomerID', 'InvoiceDate'], inplace=True)
        return df

    def load_uploaded_data(uploaded_file):
        df = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'])
        df.sort_values(by=['CustomerID', 'InvoiceDate'], inplace=True)
        return df

    #df = load_data()
    # Sidebar options
    data_selection = st.sidebar.selectbox("Select data source", ["Demo Data", "Upload Data"], index=0)
    # Load data based on user selection
    if data_selection == "Upload Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            df = load_uploaded_data(uploaded_file)
        else:
            df = load_data()
    else:
        df = load_data()

    # Sidebar inputs for column selection
    if data_selection == "Upload Data" and uploaded_file is not None:
        selected_customer_id = st.sidebar.selectbox("Select Customer ID Column", df.columns)
        selected_price = st.sidebar.selectbox("Select Price Column", df.columns)
        selected_order_id = st.sidebar.selectbox("Select Order ID Column", df.columns)
        selected_timestamp = st.sidebar.selectbox("Select Timestamp Column", df.columns)
    else:
        selected_customer_id = "CustomerID"
        selected_price = "UnitPrice"
        selected_order_id = "InvoiceNo"
        selected_timestamp = "InvoiceDate"

    def btyd(df, category:str):
        df2 = df.copy()
        df2 = df2[df2['Description'] == category]
        today = df.InvoiceDate.max().strftime('%Y-%m-%d')
        rfm = summary_data_from_transaction_data(df, 'CustomerID', 'InvoiceDate', monetary_value_col='UnitPrice', observation_period_end=today, freq='D',)
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])
        rfm_repeat = rfm[rfm['frequency']>0]
        # ggf = GammaGammaFitter(penalizer_coef = 0.01)
        # ggf.fit(rfm_repeat['frequency'], rfm_repeat['monetary_value'])
        return bgf

    category = st.sidebar.selectbox("Select a category", df['Description'].unique())
    bgf = btyd(df, str(category))

    class OptimalPriceResult(NamedTuple):
        price: float
        price_index: int

    def get_optimal_price(prices: List[float], demands: List[float]) -> OptimalPriceResult:
        """
        Identify the optimal prices. Note that this is a Greedy Decision.
        """
        index = np.argmax(prices * demands)
        return OptimalPriceResult(price_index=index, price=prices[index])

    # def sample_demands_from_model(p_lambdas: List[dict]) -> List[float]:
    #     """
    #     Samples demands from the gamma models.
    #     """
    #     return list(map(lambda v: np.random.gamma(v['alpha'], 1/v['beta']), p_lambdas))

    def sample_demands_from_model(p_lambdas: List[dict]) -> List[float]:
        """
        Samples demands from the gamma models.
        """
        demands = []
        for v in p_lambdas:
            alpha = v.get('alpha')
            beta = v.get('beta')
            if alpha is None or beta is None:
                # Handle the case where 'alpha' or 'beta' key is missing
                raise ValueError("Invalid price_meta dictionary. Missing 'alpha' or 'beta' key.")
            demands.append(np.random.gamma(alpha, 1/beta))
        return demands

    # Define the prices to be tested
    # create multiple inputs for prices
    st.sidebar.title("Price Inputs")
    st.write("Please enter the prices you want to test")
    prices_to_test = []
    for i in range(5):
        price = st.sidebar.number_input(f"Price {i+1}", min_value=0.01, max_value=100.0, step=0.01, value=2.49)
        prices_to_test.append(price)

    prices_to_test = np.array(prices_to_test)
    # Define the prior values for the alpha and beta that define a gamma distribution
    # Feel free to change the values of alpha and beta and experiment with them. They are not set in stone.
    alpha_0 = bgf.params_[1]
    beta_0 = bgf.params_[0]

    def sample_true_demand(price: float) -> float:
        """
        np.poisson.random -> https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
        """
        theta_1 = bgf.params_[2]
        theta_2 = bgf.params_[3]
        demand = theta_1 + theta_2 * price
        return np.random.poisson(demand, 1)[0]

    class priceParams(NamedTuple):
        price: float
        alpha: float
        beta: float

    # Build a list of priceParams for all the prices to be tested.
    p_lambdas: List[dict] = []
    for price in prices_to_test:
        p_lambdas.append(
            priceParams(
                price=price, 
                alpha=alpha_0, 
                beta=beta_0
            )._asdict()
        )

    # Track the counts of selected prices
    price_counts = defaultdict(lambda: 0)

    # Run the simulation for 500 iterations
    st.header("Simulation Results")
    for t in range(500):
        # Sample demands from the model
        demands = sample_demands_from_model(p_lambdas)

        # Pick the price that maximizes the revenue
        optimal_price_res = get_optimal_price(prices_to_test, demands)

        # Increase the count for the price
        price_counts[optimal_price_res.price] += 1

        # Offer the selected price and observe demand
        demand_t = sample_true_demand(optimal_price_res.price)

        # Update model parameters / Update our Belief
        v = p_lambdas[optimal_price_res.price_index]
        v['alpha'] += demand_t
        v['beta'] += 1

        if t == 50 or t == 100 or t == 200 or t == 300 or t == 400 or t == 500:
                fig, ax = plt.subplots()
                x = np.linspace(0, 30, 100)

                for p_lambda in p_lambdas:
                    alpha = p_lambda['alpha']
                    beta = p_lambda['beta']
                    pdf = gamma.pdf(x, alpha, scale=1/beta)
                    ax.plot(x, pdf, label=f"Price: {p_lambda['price']}")

                ax.set_xlabel('Demand')
                ax.set_ylabel('Probability Density')
                ax.set_title(f'PDF after {t} iterations')
                ax.legend()

                st.pyplot(fig)

    # # Print the price counts and selected prices and show as a table, highlight the optimal price
    # for price, count in price_counts.items():
    #     st.write(f"Price: {price} | Count: {count}")

    # Display the price counts and selected prices in a table, with optimal price highlighted
    st.header("Price Counts")
    df_counts = pd.DataFrame({'Price': list(price_counts.keys()), 'Count': list(price_counts.values())})
    df_counts = df_counts.style.highlight_max(subset='Count', color='lightgreen')
    st.dataframe(df_counts)

    # Mark the optimal price
    optimal_price = max(price_counts, key=price_counts.get)
    st.markdown(f"**Optimal Price: {optimal_price}**")

# # Define the API endpoint
# @app.post("/optimal_price")
# def optimal_price(price_params: PriceParams):
#     # """
    # Returns the optimal price based on the price_params.
    # """
    # # Define the prices to be tested
    # prices_to_test = np.array([price_params.price])

    # # Define the prior values for the alpha and beta that define a gamma distribution
    # # Feel free to change the values of alpha and beta and experiment with them. They are not set in stone.
    # alpha_0 = price_params.alpha
    # beta_0 = price_params.beta



    # def sample_true_demand(price: float) -> float:
    #     """
    #     np.poisson.random -> https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
    #     """
    #     theta_1 = bgf.params_[2]
    #     theta_2 = bgf.params_[3]
    #     demand = theta_1 + theta_2 * price
    #     return np.random.poisson(demand, 1)[0]

    # class priceParams(NamedTuple):
    #     price: float
    #     alpha: float
    #     beta: float

    # # Build a list of priceParams for all the prices to be tested.
    # p_lambdas: List[dict] = []
    # for price in prices_to_test:
    #     p_lambdas.append(
    #         priceParams(
    #             price=price, 
    #             alpha=alpha_0, 
    #             beta=beta_0
    #         )._asdict()
    #     )

    # # Track the counts of selected prices
    # price_counts = defaultdict(lambda: 0)

    # # Run the simulation for 500 iterations
    # for t in range(500):
    #     # Sample demands from the model
    #     demands = sample_demands_from_model(p_lambdas)

    #     # Pick the price that maximizes the revenue
    #     optimal_price_res = get_optimal_price(prices_to_test, demands)

    #     # Increase the count for the price
    #     price_counts[optimal_price_res.price] += 1

    #     # Offer the selected price and observe demand
    #     demand_t = sample_true_demand(optimal_price_res.price)

    #     # Update model parameters / Update our Belief
    #     v = p_lambdas[optimal_price_res.price_index]
    #     v['alpha'] += demand_t
    #     v['beta'] += 1

    # # Return the optimal price
    # optimal_price = max(price_counts, key=price_counts.get)
    # return {"optimal_price": optimal_price}  

# Run the FastAPI app
if __name__ == "__main__":
    # Optional: Add any Streamlit setup code here (if needed)
    # For example, loading data, models, etc.

    # # Start the FastAPI app with Uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8501)
    main()