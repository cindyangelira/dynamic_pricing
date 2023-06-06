# Awesome Dynamic Pricing

This is part of [Square Developer Hackhaton](https://square2023.devpost.com/).

# Dynamic Pricing

This is a pricing strategy platform that enable seller to evaluate and find their optimal pricing of each product.\
What's make it different? I employed the combination of [BG/NBD](https://lifetimes.readthedocs.io/en/latest/Quickstart.html) model and [Thompson sampling](https://towardsdatascience.com/thompson-sampling-fc28817eacb8). Thus, sellers can can decide which price to set given their budget constraints. 

## Requirements

Make sure you have Python >= 3.4

## Setup

### Install the Python client library

1. Make sure you have Python >= 3.8 installed from [python.org](https://www.python.org/).

2. Clone this repository and create new environment `python3 -m venv env`

2. Run the following command to install required dependencies:

   `pip install -r requirements.txt` or `pip3 install -r requirements.txt`

### Provide required credentials

Create a `config.ini` file at the root directory by copying the contents of the `config.ini.example` file and populate it. Note that there's sandbox and production credentials. Use `is_prod` (true/false) to choose between them.
Do not use quotes around the strings in the `config.ini` file.
(**WARNING**: never upload `config.ini` with your credentials/access_token.)

If you're just testing things out, it's recommended that you use your _sandbox_ credentials for now. See
[this article](https://developer.squareup.com/docs/testing/sandbox)
for more information on the API sandbox.

## Running the sample

From the sample's root directory, run:

    python run deployment.py

## Application Flow

This pricing platform was originally built on streamlit.

If there's no data uploaded, it will analyze e-commerce public dataset from [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)\
Given the available dataset and choosen category item, BG/NBD model will train the data to find the best Beta and Gamma for Gamma Distribution.\
After that, seller can input their preferenced range of budget. Thompson sampling will do the job upto 500 iterations to find the optimal price.



