import os
import shutil

# get paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "cleaned"))
FUND_PATH = BASE_DIR

# tech tickers
tech_stocks = [
    "aapl",
    "msft",
    "nvda",
    "amd",
    "intc",
    "csco",
    "adbe",
    "crm",
    "orcl",
    "ibm",
    "txn",
    "avgo",
    "qcom",
    "meta",
    "goog",
    "googl",
    "tsla",
    "snow",
    "pltr",
    "net",
    "ddog",
    "okta",
    "twlo",
    "team",
    "zm",
    "docu",
    "mdb",
    "crwd",
    "zs",
    "panw",
    "ftnt",
    "now",
    "wdc",
    "stx",
    "hpq",
    "dell",
    "roku",
    "snap",
]

copied_count = 0

for ticker in tech_stocks:

    filename = f"{ticker}.us_cleaned.csv"
    src = os.path.join(CLEANED_PATH, filename)
    dst = os.path.join(FUND_PATH, filename)

    if os.path.exists(src):

        shutil.copy(src, dst)
        copied_count += 1
        print(f"Copied: {filename}")

    else:
        print(f"Missing: {filename}")

print(f"\nTech fund dataset created with {copied_count} stocks.")
