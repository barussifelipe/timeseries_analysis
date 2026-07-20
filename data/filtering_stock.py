import re 
import pandas as pd 

def get_base_ticker(symbol):
    return re.sub(r'[/.\^-][A-Za-z0-9]+$', '', symbol)
       
def select_primary_ticker(group, base_symbol):
            
    plain_ticker = group[group['Symbol'] == base_symbol]
    if not plain_ticker.empty:
        return plain_ticker.iloc[0]


    class_a = group[group['Symbol'].str.endswith('/A', na=False)]
    if not class_a.empty:
        return class_a.iloc[0]

    print(f"Fallback: Selecting the row with the largest Market Cap or first available for symbol {base_symbol}.")
    return group.sort_values(by='Market Cap', ascending=False).iloc[0]

def fetch_index_stock(file_path, type: str) -> list: 
    """
    Fetches stock index data from a CSV file and stores it in a SQLite database.

    Args:
        file_path (str): The path to the CSV file containing stock index data.

    Returns:
        list: A list of unique stock symbols.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=['Symbol', 'Name', 'Market Cap'])
    

    if type == "NASDAQ":
        symbols = df.iloc[:, 0].astype(str).str[:4].unique().tolist() 

    elif type == "NYSE":
        
        df["Symbol"] = df.iloc[:, 0].astype(str) 
        df["Name"] = df.iloc[:, 1].astype(str)

        ignore_keywords = [
        'Fund', 'Trust', 'Notes', 'Warrant', 'Preferred', 
        'Debentures', 'Senior Floating Rate', 'Term'
        ]   
        pattern = '|'.join(ignore_keywords)

        df = df[~df['Symbol'].str.contains('^', regex=False)]

        df['Base_Symbol'] = df['Symbol'].apply(get_base_ticker)

        print(df['Base_Symbol'])

        clean_df = df.groupby('Base_Symbol', group_keys=False).apply(lambda g: select_primary_ticker(g, g.name)).reset_index()

        clean_df = clean_df[~clean_df['Name'].str.contains(pattern, case=False, na=False, regex=True)].replace('/', '-', regex=False)

        symbols = clean_df['Symbol'].unique().tolist() 


    return symbols