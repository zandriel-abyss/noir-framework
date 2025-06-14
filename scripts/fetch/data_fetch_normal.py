import requests
import json
import time
import pandas as pd
from pathlib import Path

ETHERSCAN_API_KEY = 'WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH'
BASE_URL = 'https://api.etherscan.io/api'

# Load OFAC and NK hacker addresses
with open('datasource/raw/ofac-eth-addresses.txt') as f:
    ofac_addresses = {line.strip().lower() for line in f if line.strip()}

with open('datasource/raw/nk-hackers.txt') as f:
    nk_addresses = {line.strip().lower() for line in f if line.strip()}

flagged_addresses = ofac_addresses.union(nk_addresses)

# Load “normal” user addresses (one per line)
with open('datasource/raw/normal-users.txt') as f:
    normal_list = [line.strip().lower() for line in f if line.strip()]

# Filter out any that appear in the flagged lists
normal_addresses = [addr for addr in normal_list if addr not in flagged_addresses]
print(f"Found {len(normal_addresses)} normal addresses (not flagged).")

def fetch_transactions(address, startblock=0, endblock=99999999, sort='asc'):
    """
    Call Etherscan txlist API to fetch all transactions for `address`.
    Returns an empty list if none or on error.
    """
    url = (
        f"{BASE_URL}"
        f"?module=account&action=txlist&address={address}"
        f"&startblock={startblock}&endblock={endblock}&sort={sort}"
        f"&apikey={ETHERSCAN_API_KEY}"
    )
    try:
        response = requests.get(url)
    except Exception as e:
        print(f"Request error for {address}: {e}")
        return []
    time.sleep(0.2)  # rate‐limit pause

    if response.status_code != 200:
        print(f"HTTP {response.status_code} fetching {address}")
        return []

    data = response.json()
    if data.get('status') == '1' and 'result' in data:
        return data['result']
    else:
        # status '0' means no transactions, or an error message returned
        return []

def main():
    all_txs = []
    output_dir = Path('datasource/raw')
    output_dir.mkdir(exist_ok=True)

    # How many normal addresses to process (adjust if needed)
    SAMPLE_SIZE = min(50, len(normal_addresses))
    print(f"Processing {SAMPLE_SIZE} normal addresses...")

    for idx, addr in enumerate(normal_addresses[:SAMPLE_SIZE], start=1):
        print(f"[{idx}/{SAMPLE_SIZE}] Fetching transactions for: {addr}")
        txs = fetch_transactions(addr)
        if not txs:
            # No transactions or error, skip
            continue

        for tx in txs:
            all_txs.append({
                'wallet_address': addr,
                'hash':          tx.get('hash'),
                'from':          tx.get('from'),
                'to':            tx.get('to'),
                'value':         tx.get('value'),
                'blockNumber':   tx.get('blockNumber'),
                'timeStamp':     tx.get('timeStamp'),
                'gasUsed':       tx.get('gasUsed'),
                'isError':       tx.get('isError'),
                'contractAddress': tx.get('contractAddress'),
                'input':         tx.get('input')
            })

    # Convert to DataFrame and save CSV
    if all_txs:
        df = pd.DataFrame(all_txs)
        output_path = output_dir / 'normal_transactions.csv'
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} total transaction rows to {output_path}")
    else:
        print("No transactions collected; normal_transactions.csv not created.")

if __name__ == '__main__':
    main()