import requests
import json
import time
import csv
from pathlib import Path

ETHERSCAN_API_KEY = 'WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH'
BASE_URL = 'https://api.etherscan.io/api'

# Example mixer/bridge addresses
with open('datasource/raw/mixer-users.txt') as f:
    mixer_addresses = [line.strip() for line in f.readlines() if line.strip()]

def fetch_transactions(address, startblock=0, endblock=99999999, sort='asc'):
    url = (f"{BASE_URL}?module=account&action=txlist&address={address}"
           f"&startblock={startblock}&endblock={endblock}&sort={sort}&apikey={ETHERSCAN_API_KEY}")
    response = requests.get(url)
    time.sleep(0.2)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching transactions for {address}: {response.status_code}")
        return None

def main():
    output_file = Path(__file__).resolve().parent.parent.parent / 'datasource' / 'raw' / 'mixer_interactions.csv'
    all_transactions = []

    for idx, addr in enumerate(mixer_addresses):
        print(f"[{idx+1}/{len(mixer_addresses)}] Fetching for: {addr}")
        tx_data = fetch_transactions(addr)
        if tx_data:
            tx_list = tx_data.get("result", [])
            if isinstance(tx_list, list) and tx_list:
                for tx in tx_list:
                    tx['wallet_address'] = addr
                all_transactions.extend(tx_list)
            else:
                print(f"No transactions found for: {addr}")
        else:
            print(f"Failed to fetch: {addr}")

    if all_transactions:
        keys = ['wallet_address', 'hash', 'nonce', 'blockHash', 'blockNumber', 'transactionIndex', 'from',
                'to', 'value', 'gas', 'gasPrice', 'isError', 'txreceipt_status', 'input', 'contractAddress',
                'cumulativeGasUsed', 'gasUsed', 'confirmations', 'methodId', 'functionName', 'timeStamp']
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_transactions)
        print(f"Saved all mixer transactions to {output_file}")
    else:
        print("No transactions found for any mixer addresses.")

if __name__ == '__main__':
    main()