import sqlite3
import numpy as np 
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt
import os 
from filtering_stock import * 


db_filename = "data/src/stock_data.db"

conn = sqlite3.connect(db_filename)

cursor = conn.cursor()



