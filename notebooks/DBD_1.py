import pandas as pd


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