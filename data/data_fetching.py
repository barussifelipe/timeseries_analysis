import sqlite3
import numpy as np 
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt
import os 
import duckdb 


db_filename = "data/src/stock_data.db"

conn = sqlite3.connect(db_filename)

cursor = conn.cursor()

def fetch_index_stock(file_path, type: str) -> list: 
    """
    Fetches stock index data from a CSV file and stores it in a SQLite database.

    Args:
        file_path (str): The path to the CSV file containing stock index data.

    Returns:
        list: A list of unique stock symbols.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=[0])

    if type == "NASDAQ":
        symbols = df.iloc[:, 0].astype(str).str[:4].unique().tolist() 

    elif type == "NYSE":
        index_string = df.iloc[:, 0].astype(str)
        symbols = index_string[index_string.str[3] != "^"]
        symbols = [symbols.drop(symbol) for symbol in symbols if symbol[-2:] == "/B"]
        


    
    print(symbols)

    return symbols





