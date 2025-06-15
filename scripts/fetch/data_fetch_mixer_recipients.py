import pandas as pd
import requests
import time
from pathlib import Path

# --- Config ---
API_KEY = "WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH"  # Replace with your Etherscan API key
MIXER_TX_FILE = Path("datasource/raw/mixer_recipient_wallets.csv")
OUTPUT_DIR = Path("datasource/raw/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper ---
def fetch_transactions(address):
    url = (
        f"https://api.etherscan.io/api?module=account&action=txlist&address={address}"
        f"&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    if data["status"] == "1":
        return pd.DataFrame(data["result"])
    else:
        print(f"[!] Failed to fetch {address}: {data['message']}")
        return pd.DataFrame()

# --- Load Mixer Recipient Wallets ---
df_mixer = pd.read_csv(MIXER_TX_FILE)
recipient_wallets = df_mixer["to_address"].str.lower().dropna().unique()

print(f"🔍 Found {len(recipient_wallets)} unique recipient wallets from mixer outflows.")

# --- Fetch Transactions ---
all_txs = []
for i, wallet in enumerate(recipient_wallets):
    print(f"[{i+1}/{len(recipient_wallets)}] Fetching transactions for {wallet}...")
    txs = fetch_transactions(wallet)
    if not txs.empty:
        txs["wallet_address"] = wallet
        txs["label"] = "suspicious"  # Tag all mixer recipients as suspicious
        all_txs.append(txs)
    time.sleep(0.25)  # Be kind to API

# --- Save Merged Dataset ---
if all_txs:
    df_all = pd.concat(all_txs, ignore_index=True)
    out_path = OUTPUT_DIR / "mixer_recipients_transactions.csv"
    df_all.to_csv(out_path, index=False)
    print(f"✅ Saved fetched transactions to {out_path}")
else:
    print("⚠️ No transactions fetched.")