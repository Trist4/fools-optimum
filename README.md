Aramex Fuel Price Forecasting — Project 2

Methods of Operations Research | 20 July – 4 September 2026

Overview

Predictive modelling project for Aramex South Africa's Business Intelligence Executive (BIE) team. Goal: forecast South African fuel prices (1-month, 3-month, and 12-month horizons) using publicly available data, to support proactive client pricing rather than reactive responses to price shocks.

Repository structure
.
├── README.md
├── data/               # all raw and processed data (see convention below)
├── notebooks/          # exploratory analysis
├── src/                # modelling and pipeline code
├── report/             # technical report drafts (max 15 pages)
└── slides/             # presentation materials
Data storage convention

All collected data lives in the data/ folder as CSV files, one entry per month.

Granularity: each row = one month.
File naming: data/<source>_<indicator>.csv (e.g. data/cef_basic_fuel_price.csv, data/imf_gscpi.csv, data/sarb_usd_zar.csv).
Required columns (minimum):
column	format	notes
date	YYYY-MM-01	first of month
value	numeric	the indicator's value for that month - give the label of the data ie fuel or gold 



Key dates
Presentation: Friday, 4 September 2026
Report: Sunday, 6 September 2026
