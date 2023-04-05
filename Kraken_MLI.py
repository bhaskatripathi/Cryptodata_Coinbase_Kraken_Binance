# DataSource: KRAKEN EXCHANGE
# https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data
# Each ZIP file contains the relevant CSV files for 1, 5, 15, 60, 720 and 1440 minute intervals, 
# which can be viewed in a text editor, used in code, converted into other formats (such as JSON, XML, etc.) or imported 
# into a graphical charting application.
# Get all files with _1 (daily data) and move them to a new directory

# In[1]======= Move daily data files to a new folder =======================

import os
import shutil
source_dir = r'E:\Bhaskar\Personal\Personal\Phd\Research\Research Proposal\19 Dec Literature Review\Objective4 - Liquidity of Bitcoin & Stock exchanges\data\Kraken_OHLCVT'
destination = r'E:\Bhaskar\Personal\Personal\Phd\Research\Research Proposal\19 Dec Literature Review\Objective4 - Liquidity of Bitcoin & Stock exchanges\data\Kraken_OHLCVT\to analyze'
for file_name in os.listdir(source_dir):
    # Get daily data 1440 minute intervals (daily data)
    if file_name.endswith('_1440.csv'):
        file_path = os.path.join(source_dir, file_name)
        shutil.move(file_path, destination)

# In[2]======= Add Header column OHLCVT =======================

#Add header row to the files
import os
import pandas as pd
destination_dir = r"E:\Bhaskar\Personal\Personal\Phd\Research\Research Proposal\19 Dec Literature Review\Objective4 - Liquidity of Bitcoin & Stock exchanges\data\Kraken_OHLCVT\to analyze"
# Get a list of all the files in the destination directory
files = os.listdir(destination_dir)
# Add header row to files ending with _1440.csv 
for file in files:
    # Data with 1440 minute intervals (daily data)
    if file.endswith("_1440.csv"):
        file_path = os.path.join(destination_dir, file)
        df = pd.read_csv(file_path)
        if not df.empty and 'Open' not in df.columns:
            df.columns = ['TimeStamp','Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
            df.to_csv(file_path, index=False)           
# Show the names and count of the files
print("Files in destination directory:")
for file in files:
    print(file)
print("Total count: ", len(files))

# In[2]======= Calculate MLI and save file to a destination folder=======================

import os
import pandas as pd

destination_dir = r"E:\Bhaskar\Personal\Personal\Phd\Research\Research Proposal\19 Dec Literature Review\Objective4 - Liquidity of Bitcoin & Stock exchanges\data\Kraken_OHLCVT\to analyze\calculated_MLI"
overall_mli_file = r"E:\Bhaskar\Personal\Personal\Phd\Research\Research Proposal\19 Dec Literature Review\Objective4 - Liquidity of Bitcoin & Stock exchanges\data\Kraken_OHLCVT\to analyze\calculated_MLI\calculated_MLIkraken_overall_mli.csv"

# Get a list of all the files in the destination directory
files = os.listdir(destination_dir)

# Calculate MLI for each currency pair and store the results in a list
all_ml_values = []
for file in files:
    # Data with 1440 minute intervals (daily data)
    if file.endswith("_1440.csv"):
        file_path = os.path.join(destination_dir, file)
        currency_pair = file[:-9]
        df = pd.read_csv(file_path)
        if 'Open' not in df.columns:
            df.columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']

        # Calculate MLI for currency pair
        mlis = []
        for i in range(1, len(df)):
            pt = df.iloc[i]['Close']
            pt_minus_1 = df.iloc[i-1]['Close']
            vt = df.iloc[i]['Volume']
            mli = ((pt - pt_minus_1)**2) / vt
            mlis.append(mli)
        mli_value = sum(mlis)
        volume = df['Volume'].sum()
        all_ml_values.append({'Currency Pair': currency_pair, 'MLI': mli_value, 'Volume': volume})
        print(f"MLI for {currency_pair}: {mli_value}")

# Save the updated file with header row in the destination directory
for file in files:
    if file.endswith("_1.csv"):
        file_path = os.path.join(destination_dir, file)
        df = pd.read_csv(file_path)
        if 'Open' not in df.columns:
            df.columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
            df.to_csv(os.path.join(destination_dir, file), index=False)

# Calculate overall exchange MLI weighted with traded volume
df_all = pd.DataFrame(all_ml_values)
total_volume = df_all.groupby(['Currency Pair']).sum().reset_index()
total_volume['Weightage'] = total_volume['Volume'] / total_volume['Volume'].sum()
total_volume['Weighted_MLI'] = total_volume['MLI'] * total_volume['Weightage']
overall_mli = total_volume['Weighted_MLI'].sum()
print(f"Overall MLI for Kraken Exchange: {overall_mli}")

# Save overall MLI to a CSV file
total_volume.to_csv(overall_mli_file, index=False)
print("Saved Overall MLI to CSV")

# Save individual MLI values to a CSV file
pd.DataFrame(all_ml_values).to_csv(os.path.join(destination_dir, "individual_mli.csv"), index=False)
print("Saved individual MLI values to a CSV file")
