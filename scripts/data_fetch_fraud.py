import requests
import time
import json
import pandas as pd
from pathlib import Path

# Etherscan API key
ETHERSCAN_API_KEY = 'WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH'
BASE_URL = 'https://api.etherscan.io/api'

# Load OFAC and NK hacker addresses
with open('datasource/raw/ofac-eth-addresses.txt') as f:
    ofac_addresses = [line.strip() for line in f.readlines() if line.strip()]

with open('datasource/raw/nk-hackers.txt') as f:
    nk_addresses = [line.strip() for line in f.readlines() if line.strip()]

fraud_addresses = list(set(ofac_addresses + nk_addresses))

def fetch_transactions(address, startblock=0, endblock=99999999, sort='asc'):
    url = (f"{BASE_URL}?module=account&action=txlist&address={address}"
           f"&startblock={startblock}&endblock={endblock}&sort={sort}&apikey={ETHERSCAN_API_KEY}")
    response = requests.get(url)
    time.sleep(0.2)  # Avoid hitting rate limits
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':  # Status 1 means transactions found
            return data['result']
        else:
            print(f"No transactions found for {address}.")
            return []
    else:
        print(f"Error fetching transactions for {address}: {response.status_code}")
        return []

def main():
    all_txs = []

    for idx, addr in enumerate(fraud_addresses):
        print(f"[{idx+1}/{len(fraud_addresses)}] Fetching for: {addr}")
        txs = fetch_transactions(addr)
        for tx in txs:
            # Flatten transaction data, include address label
            tx_record = {
                'wallet_address': addr,
                'hash': tx.get('hash'),
                'from': tx.get('from'),
                'to': tx.get('to'),
                'value': tx.get('value'),
                'blockNumber': tx.get('blockNumber'),
                'timeStamp': tx.get('timeStamp'),
                'gasUsed': tx.get('gasUsed'),
                'isError': tx.get('isError'),
                'contractAddress': tx.get('contractAddress'),
                'input': tx.get('input')
            }
            all_txs.append(tx_record)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_txs)
    output_path = Path('datasource/raw/fraud_transactions.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved combined fraud transactions to {output_path}")

if __name__ == '__main__':
    main()