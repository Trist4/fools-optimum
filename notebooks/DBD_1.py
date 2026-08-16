import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#file_path_1 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/95.csv"
#file_path_2 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/USDZAR.csv"
#file_path_3 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/DCOILBRENTEU.csv"
#file_path_4 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/GPR.csv"
#file_path_5 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/GSCPI.csv"
#file_path_6 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/GECON.csv"
#file_path_7 = "/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/INDPRO.csv"



def import_breakdown(file_path):

    df = pd.read_csv(file_path)

    # Remove non-breaking spaces and whitespace from every string cell
    df = df.replace({r"\xa0": "", r"¬†": "", r" ": "" }, regex=True)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace("&", "and", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    # Rename columns
    df.rename(columns={
        "Fuel_tax": "Fuel_Tax",
        "Customs_andexcise": "Customs_Excise",
        "Equalization_fund_levy": "Equalization_Fund_Levy",
        "Road_accident_fund": "Road_Accident_Fund",
        "Transport_cost": "Transport_Cost",
        "Petroleum_Products_levy": "Petroleum_Products_Levy",
        "Wholesale_margin": "Wholesale_Margin",
        "Secondary_distribution": "Secondary_Distribution",
        "Retail_margin": "Retail_Margin",
        "Slate_levy": "Slate_Levy",
        "Delivery_cost": "Delivery_Cost",
        "Incremental_inland_transposrt_recovery_cost":
            "Incremental_Inland_Transport_Recovery_Cost"
    }, inplace=True)

    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"])

    # Convert all columns except Date to numeric
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort newest first
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    # Add a total cost column
    df["Total_Fuel_Price"] = df.drop(columns=["Date"]).sum(axis=1)

    # Have a separate df for total price
    return df, df[["Date", "Total_Fuel_Price", "BFP"]]


def import_USD_ZAR(file_path):

    df = pd.read_csv(file_path)

    df.columns = ["Date", "USD_ZAR"]

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    return df



def load_jodi_production(data_dir="../data/indpro", pattern="*.csv",
                          countries=None, product="CRUDEOIL",
                          flow="INDPROD", unit="KBD", total_only=True):
    """
    Reads and compiles JODI-Oil yearly CSV files (one file per year) into a
    single clean monthly production dataframe.

    Parameters
    ----------
    data_dir  : folder containing the yearly JODI csv files
    pattern   : glob pattern matching the yearly files, e.g. '2011.csv'
    countries : optional list of REF_AREA codes to include (e.g. ["ZA", "US"])
                if None, includes all reporting countries
    product   : ENERGY_PRODUCT to filter to (default 'CRUDEOIL')
    flow      : FLOW_BREAKDOWN to filter to (default 'INDPROD' = indigenous production)
    unit      : UNIT_MEASURE to filter to (default 'KBD' = thousand barrels/day)
    total_only: if True (default), returns a single 'Total_Production' column
                summed across countries; if False, returns one column per country

    Returns
    -------
    DataFrame indexed by Date (month start).
    """

    files = sorted(glob.glob(f"{data_dir}/{pattern}"))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]

        mask = (
            (df["ENERGY_PRODUCT"] == product) &
            (df["FLOW_BREAKDOWN"] == flow) &
            (df["UNIT_MEASURE"] == unit)
        )
        df = df[mask].copy()

        if countries is not None:
            df = df[df["REF_AREA"].isin(countries)]

        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df["Date"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m")

        frames.append(df[["Date", "REF_AREA", "OBS_VALUE"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "REF_AREA"])

    if total_only:
        total = combined.groupby("Date")["OBS_VALUE"].sum(min_count=1)
        result = total.to_frame("Total_Production").sort_index()
    else:
        result = combined.pivot(index="Date", columns="REF_AREA", values="OBS_VALUE")
        result = result.sort_index()

    return result


def import_brentcrude(file_path):

    df = pd.read_csv(file_path)

    df.columns = ["Date", "Brent_Crude"]

    df["Date"] = pd.to_datetime(df["Date"])

    # Keep only Jan 2011 onwards
    df = df[df["Date"] >= "2011-01-01"]

    # Newest first
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    return df


def import_gpr(file_path):
    
    df = pd.read_csv(file_path)

    df.columns = ["Date", "GPR"]

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

    # Sort newest first
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    return df


def import_gscpi(file_path):

    df = pd.read_csv(file_path)

    df.columns = ["Date", "GSCPI"]

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")

    # Sort newest first
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    return df


def import_gecon(file_path):

    df = pd.read_csv(file_path)

    # Keep only the columns we need
    df = df[["Date", "Standardized GECON"]]

    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")

    # Rename for convenience
    df = df.rename(columns={
        "Standardized GECON": "GECON"
    })

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def import_indpro(file_path):

    df = pd.read_csv(file_path)

    # Keep only the required columns
    df = df[["Date", "INDPRO"]]

    # Convert dates (e.g. 6/1/26 -> 2026-06-01)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    return df




#       import files
#=============================================================
#       add features

def add_gpr_feature(master_df, gpr_df):

    master_df = master_df.copy()

    # Convert dates
    master_df["Date"] = pd.to_datetime(master_df["Date"])
    gpr_df["Date"] = pd.to_datetime(gpr_df["Date"], format="%m/%d/%y")

    # Rename the column (optional but keeps naming consistent)
    gpr_df = gpr_df.rename(columns={"GPR": "GPR"})

    # Merge
    master_df = master_df.merge(
        gpr_df,
        on="Date",
        how="left"
    )

    return master_df


def add_gscpi_feature(master_df, gscpi_df):

    master_df = master_df.copy()

    # Convert dates
    master_df["Date"] = pd.to_datetime(master_df["Date"])
    gscpi_df["Date"] = pd.to_datetime(gscpi_df["Date"], format="%d-%b-%Y")

    # Move all dates to the first day of the same month
    gscpi_df["Date"] = gscpi_df["Date"].dt.to_period("M").dt.to_timestamp()

    # Merge
    master_df = master_df.merge(
        gscpi_df[["Date", "GSCPI"]],
        on="Date",
        how="left"
    )

    return master_df


def add_usd_features(petrol_breakdown_df, USD_ZAR_df):

    usd = USD_ZAR_df.copy()

    usd = usd.sort_values("Date")

    # Rolling features (past information only)
    usd["USDZAR_Mean"] = (
        usd["USD_ZAR"]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    usd["USDZAR_Std"] = (
        usd["USD_ZAR"]
        .rolling(window=30, min_periods=2)
        .std()
    )

    usd["USDZAR_LastWeekMean"] = (
        usd["USD_ZAR"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    # Take the LAST available trading day of each month
    monthly_usd = (
        usd
        .groupby(usd["Date"].dt.to_period("M"))
        .last()
        .reset_index(drop=True)
    )

    # Convert to first day of month so merge still works
    monthly_usd["Date"] = (
        monthly_usd["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_usd = monthly_usd[
        [
            "Date",
            "USDZAR_Mean",
            "USDZAR_Std",
            "USDZAR_LastWeekMean"
        ]
    ]

    master_df = petrol_breakdown_df.merge(
        monthly_usd,
        on="Date",
        how="left"
    )

    return master_df


def add_brent_features(master_df, brent_df):

    brent = brent_df.copy()

    brent = brent.sort_values("Date")

    # Rolling features (past information only)
    brent["Brent_MonthMean"] = (
        brent["Brent_Crude"]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    brent["Brent_MonthStd"] = (
        brent["Brent_Crude"]
        .rolling(window=30, min_periods=2)
        .std()
    )

    brent["Brent_LastWeekMean"] = (
        brent["Brent_Crude"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    brent["Brent_LastWeekStd"] = (
        brent["Brent_Crude"]
        .rolling(window=7, min_periods=2)
        .std()
    )

    # Keep only the final trading day of each month
    monthly_brent = (
        brent
        .groupby(brent["Date"].dt.to_period("M"))
        .last()
        .reset_index(drop=True)
    )

    monthly_brent["Date"] = (
        monthly_brent["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_brent = monthly_brent[
        [
            "Date",
            "Brent_MonthMean",
            "Brent_MonthStd",
            "Brent_LastWeekMean",
            "Brent_LastWeekStd"
        ]
    ]

    master_df = master_df.merge(
        monthly_brent,
        on="Date",
        how="left"
    )

    return master_df


def add_gecon_feature(master_df, gecon_df):

    master_df = master_df.copy()

    master_df["Date"] = pd.to_datetime(master_df["Date"])

    master_df = master_df.merge(
        gecon_df,
        on="Date",
        how="left"
    )

    return master_df


def add_indpro_feature(master_df, indpro_df):

    master_df = master_df.copy()

    master_df["Date"] = pd.to_datetime(master_df["Date"])

    master_df = master_df.merge(
        indpro_df,
        on="Date",
        how="left"
    )

    return master_df


#       add features
#=============================================================
#       add lags




def add_bfp_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    master_df["BFP_Lag1"] = master_df["BFP"].shift(1)
    master_df["BFP_Lag2"] = master_df["BFP"].shift(2)
    master_df["BFP_Lag3"] = master_df["BFP"].shift(3)
    master_df["BFP_Lag6"] = master_df["BFP"].shift(6)
    master_df["BFP_Lag12"] = master_df["BFP"].shift(12)

    return master_df


def add_usdzar_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    cols = [
        "USDZAR_Mean",
        #"USDZAR_Std",
        "USDZAR_LastWeekMean"
    ]

    for col in cols:
        master_df[f"{col}_Lag1"] = master_df[col].shift(1)
        master_df[f"{col}_Lag2"] = master_df[col].shift(2)
        master_df[f"{col}_Lag3"] = master_df[col].shift(3)

    return master_df


def add_brent_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    cols = [
        "Brent_MonthMean",
        #"Brent_MonthStd",
        "Brent_LastWeekMean",
        "Brent_LastWeekStd"
    ]

    for col in cols:
        master_df[f"{col}_Lag1"] = master_df[col].shift(1)
        master_df[f"{col}_Lag2"] = master_df[col].shift(2)
        master_df[f"{col}_Lag3"] = master_df[col].shift(3)

    return master_df


def add_gpr_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    master_df["GPR_Lag1"] = master_df["GPR"].shift(1)
    master_df["GPR_Lag2"] = master_df["GPR"].shift(2)
    master_df["GPR_Lag3"] = master_df["GPR"].shift(3)
    master_df["GPR_Lag6"] = master_df["GPR"].shift(6)
    master_df["GPR_Lag12"] = master_df["GPR"].shift(12)

    return master_df


def add_gscpi_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    master_df["GSCPI_Lag1"] = master_df["GSCPI"].shift(1)
    master_df["GSCPI_Lag2"] = master_df["GSCPI"].shift(2)
    master_df["GSCPI_Lag3"] = master_df["GSCPI"].shift(3)
    master_df["GSCPI_Lag6"] = master_df["GSCPI"].shift(6)
    master_df["GSCPI_Lag12"] = master_df["GSCPI"].shift(12)

    return master_df


def add_gecon_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    master_df["GECON_Lag1"] = master_df["GECON"].shift(1)
    master_df["GECON_Lag2"] = master_df["GECON"].shift(2)
    master_df["GECON_Lag3"] = master_df["GECON"].shift(3)
    #master_df["GECON_Lag6"] = master_df["GECON"].shift(6)
    #master_df["GECON_Lag12"] = master_df["GECON"].shift(12)

    return master_df


def add_indpro_lags(master_df):

    master_df = master_df.sort_values("Date").copy()

    master_df["INDPRO_Lag1"] = master_df["INDPRO"].shift(1)
    master_df["INDPRO_Lag2"] = master_df["INDPRO"].shift(2)
    master_df["INDPRO_Lag3"] = master_df["INDPRO"].shift(3)
    master_df["INDPRO_Lag6"] = master_df["INDPRO"].shift(6)
    master_df["INDPRO_Lag12"] = master_df["INDPRO"].shift(12)

    return master_df



#       add lags
#=============================================================
#       add deta features


def add_bfp_delta_features(master_df):

    master_df = master_df.sort_values("Date").copy()

    # Month-to-month change in BFP
    master_df["BFP_Delta"] = master_df["BFP"].diff()

    # Lagged delta features
    master_df["BFP_Delta_Lag1"] = master_df["BFP_Delta"].shift(1)
    master_df["BFP_Delta_Lag2"] = master_df["BFP_Delta"].shift(2)
    master_df["BFP_Delta_Lag3"] = master_df["BFP_Delta"].shift(3)
    master_df["BFP_Delta_Lag4"] = master_df["BFP_Delta"].shift(4)
    master_df["BFP_Delta_Lag5"] = master_df["BFP_Delta"].shift(5)
    master_df["BFP_Delta_Lag6"] = master_df["BFP_Delta"].shift(6)

    # Drop the current month's delta to prevent leakage
    master_df = master_df.drop(columns=["BFP_Delta"])

    return master_df


def add_usdzar_delta_features(master_df):

    master_df = master_df.sort_values("Date").copy()

    # Month-to-month change
    master_df["USDZAR_Mean_Delta"] = master_df["USDZAR_Mean"].diff()

    # Lagged delta features
    master_df["USDZAR_Delta_Lag1"] = master_df["USDZAR_Mean_Delta"].shift(1)
    master_df["USDZAR_Delta_Lag2"] = master_df["USDZAR_Mean_Delta"].shift(2)
    #master_df["USDZAR_Delta_Lag3"] = master_df["USDZAR_Mean_Delta"].shift(3)
    #master_df["USDZAR_Delta_Lag4"] = master_df["USDZAR_Mean_Delta"].shift(4)
    #master_df["USDZAR_Delta_Lag5"] = master_df["USDZAR_Mean_Delta"].shift(5)
    #master_df["USDZAR_Delta_Lag6"] = master_df["USDZAR_Mean_Delta"].shift(6)

    # Remove current month's delta (prevents leakage)
    master_df = master_df.drop(columns=["USDZAR_Mean_Delta"])

    return master_df

# Functions for understanding features
def print_feature_dist(df):
    with PdfPages('histograms_of_features.pdf') as pdf:
        # Generate histograms for continuous features
        for col in df:
            plt.figure(figsize=(8, 5))
            series = pd.to_numeric(df[col], errors='coerce')
            plt.hist(series.dropna(), bins=20, color='blue', alpha=0.7)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            pdf.savefig()
            plt.close()

def print_feature_vs_y_line_graphs(df):
    """
    Plots each feature in the dataframe compared to 'y', normalizing both to start at the same value.
    Saves the plots as a PDF.

    Parameters:
    df (pd.DataFrame): DataFrame containing features and a column labeled 'y'.

    Raises:
    ValueError: If 'y' is not in the dataframe.
    """
    with PdfPages('line_graphs_features_vs_y.pdf') as pdf:
        # Ensure 'y' exists in the dataframe
        if 'y' not in df.columns:
            raise ValueError("The dataframe must contain a 'y' column for comparison.")

        # Normalize 'y' to start at the same value as each feature
        y_series = pd.to_numeric(df['y'], errors='coerce')
        y_start = y_series.iloc[0]

        # Generate line graphs for each feature compared to 'y'
        for col in df:
            if col == 'y':  # Skip 'y' itself
                continue

            plt.figure(figsize=(10, 6))
            series = pd.to_numeric(df[col], errors='coerce')
            feature_start = series.iloc[0]

            # Normalize both series to start at the same value
            normalized_feature = series / feature_start if feature_start != 0 else series
            normalized_y = y_series / y_start if y_start != 0 else y_series

            plt.plot(normalized_feature.index, normalized_feature, label=col, color='blue', alpha=0.7)
            plt.plot(normalized_y.index, normalized_y, label='y', color='red', linestyle='--', alpha=0.7)

            plt.title(f'{col} vs y (Normalized)')
            plt.xlabel('Index')
            plt.ylabel('Normalized Value')
            plt.legend()
            plt.grid(axis='both', linestyle='--', alpha=0.7)
            pdf.savefig()
            plt.close()

def load_petrol():
    petrol_breakdown_df, master_petrol_df = import_breakdown("../data/95.csv")
    USD_ZAR_df = import_USD_ZAR("../data/USDZAR.csv")
    brent_crude_df = import_brentcrude("../data/DCOILBRENTEU.csv")
    GPR_df = import_gpr("../data/GPR.csv")
    GSCPI_df = import_gscpi("../data/GSCPI.csv")
    gecon_df = import_gecon("../data/GECON.csv")
    indpro_df = import_indpro("../data/INDPRO.csv")
    
    
    
    master_petrol_df = add_usd_features(master_petrol_df, USD_ZAR_df)
    master_petrol_df = add_brent_features(master_petrol_df, brent_crude_df)
    
    master_petrol_df = add_gpr_feature(master_petrol_df, GPR_df)
    master_petrol_df = add_gscpi_feature(master_petrol_df, GSCPI_df)
    master_petrol_df = add_gecon_feature(master_petrol_df, gecon_df)
    master_petrol_df = add_indpro_feature(master_petrol_df, indpro_df)
    
        
    
    master_petrol_df = add_gpr_lags(master_petrol_df)
    master_petrol_df = add_bfp_lags(master_petrol_df)
    master_petrol_df = add_usdzar_lags(master_petrol_df)
    master_petrol_df = add_brent_lags(master_petrol_df)
    master_petrol_df = add_bfp_delta_features(master_petrol_df)
    master_petrol_df = add_usdzar_delta_features(master_petrol_df)
    master_petrol_df = add_gscpi_lags(master_petrol_df)
    master_petrol_df = add_gecon_lags(master_petrol_df)
    master_petrol_df = add_indpro_lags(master_petrol_df)
    
    total_production = load_jodi_production()
    total_production_reset = total_production.reset_index()  # columns: Date, Total_Production

    # Merge explicitly on the Date column
    master_petrol_df = pd.merge(master_petrol_df, total_production_reset, on="Date", how="outer")

    master_petrol_df = master_petrol_df.sort_values("Date", ascending=True).reset_index(drop=True)

    return master_petrol_df

def load_diesel():
    diesel_breakdown_df, master_diesel_df = import_breakdown("../data/50ppm.csv")
    USD_ZAR_df = import_USD_ZAR("../data/USDZAR.csv")
    brent_crude_df = import_brentcrude("../data/DCOILBRENTEU.csv")
    GPR_df = import_gpr("../data/GPR.csv")
    GSCPI_df = import_gscpi("../data/GSCPI.csv")
    gecon_df = import_gecon("../data/GECON.csv")
    indpro_df = import_indpro("../data/INDPRO.csv")
    
    
    
    master_diesel_df = add_usd_features(master_diesel_df, USD_ZAR_df)
    master_diesel_df = add_brent_features(master_diesel_df, brent_crude_df)
    
    master_diesel_df = add_gpr_feature(master_diesel_df, GPR_df)
    master_diesel_df = add_gscpi_feature(master_diesel_df, GSCPI_df)
    master_diesel_df = add_gecon_feature(master_diesel_df, gecon_df)
    master_diesel_df = add_indpro_feature(master_diesel_df, indpro_df)
    
        
    
    master_diesel_df = add_gpr_lags(master_diesel_df)
    master_diesel_df = add_bfp_lags(master_diesel_df)
    master_diesel_df = add_usdzar_lags(master_diesel_df)
    master_diesel_df = add_brent_lags(master_diesel_df)
    master_diesel_df = add_bfp_delta_features(master_diesel_df)
    master_diesel_df = add_usdzar_delta_features(master_diesel_df)
    master_diesel_df = add_gscpi_lags(master_diesel_df)
    master_diesel_df = add_gecon_lags(master_diesel_df)
    master_diesel_df = add_indpro_lags(master_diesel_df)
    
    total_production = load_jodi_production()
    total_production_reset = total_production.reset_index()  # columns: Date, Total_Production

    # Merge explicitly on the Date column
    master_diesel_df = pd.merge(master_diesel_df, total_production_reset, on="Date", how="outer")

    master_diesel_df = master_diesel_df.sort_values("Date", ascending=True).reset_index(drop=True)

    return master_diesel_df









#           
#===========================================================================

"""
petrol_breakdown_df = import_breakdown(file_path_1)
USD_ZAR_df = import_USD_ZAR(file_path_2)
brent_crude_df = import_brentcrude(file_path_3)
GPR_df = import_gpr(file_path_4)
GSCPI_df = import_gscpi(file_path_5)
gecon_df = import_gecon(file_path_6)
indpro_df = import_indpro(file_path_7)



master_df = add_usd_features(petrol_breakdown_df, USD_ZAR_df)
master_df = add_brent_features(master_df, brent_crude_df)

#master_df = add_gpr_feature(master_df, GPR_df)
#master_df = add_gscpi_feature(master_df, GSCPI_df)
master_df = add_gecon_feature(master_df, gecon_df)
#master_df = add_indpro_feature(master_df, indpro_df)

    

#master_df = add_gpr_lags(master_df)
master_df = add_bfp_lags(master_df)
master_df = add_usdzar_lags(master_df)
master_df = add_brent_lags(master_df)
master_df = add_bfp_delta_features(master_df)
#master_df = add_usdzar_delta_features(master_df)
#master_df = add_gscpi_lags(master_df)
#master_df = add_gecon_lags(master_df)
#master_df = add_indpro_lags(master_df)

master_df = master_df.sort_values("Date", ascending=False).reset_index(drop=True)





#print(master_df.head())


master_df.to_csv("/Users/aidanpenfold/Desktop/Honours/Methods/aramex/Data/csv_files/master_df.csv", index=False)


print("\nMaster dataframe created and saved to CSV.\n")
"""