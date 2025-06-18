import requests
import json
import time
from pathlib import Path
import csv

ETHERSCAN_API_KEY = 'WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH'
BASE_URL = 'https://api.etherscan.io/api'

# Load known mixer contract addresses (e.g., Tornado Cash contracts)
with open('datasource/raw/mixer-users.txt') as f:
    mixer_addresses = [line.strip().lower() for line in f.readlines() if line.strip()]

# Output CSV
OUTPUT_FILE = Path('datasource/raw/mixer_recipient_wallets.csv')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Write header if file doesn't exist
if not OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['mixer_contract', 'tx_hash', 'to_address', 'value', 'timestamp'])

def fetch_outgoing_transactions(address, startblock=0, endblock=99999999, sort='asc'):
    url = (
        f"{BASE_URL}?module=account&action=txlist&address={address}"
        f"&startblock={startblock}&endblock={endblock}&sort={sort}&apikey={ETHERSCAN_API_KEY}"
    )
    response = requests.get(url)
    time.sleep(0.25)  # Rate limit
    if response.status_code == 200:
        return response.json().get('result', [])
    else:
        print(f"Failed to fetch txs for {address}")
        return []

def main():
    with open(OUTPUT_FILE, 'a') as f:
        writer = csv.writer(f)

        for idx, mixer in enumerate(mixer_addresses):
            print(f"[{idx+1}/{len(mixer_addresses)}] Scanning mixer: {mixer}")
            txs = fetch_outgoing_transactions(mixer)
            for tx in txs:
                if tx['from'].lower() == mixer and int(tx['value']) > 0:
                    writer.writerow([
                        mixer,
                        tx['hash'],
                        tx['to'],
                        tx['value'],
                        tx['timeStamp']
                    ])
    print(f"Saved recipient list to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()