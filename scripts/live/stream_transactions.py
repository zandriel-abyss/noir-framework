import asyncio
import websockets
import json
from web3 import Web3
from web3.exceptions import TransactionNotFound
import time
from pathlib import Path

# --- Config ---
# Your Alchemy WebSocket URL for Ethereum Mainnet
ALCHEMY_WSS_URL_ETH = "wss://eth-mainnet.g.alchemy.com/v2/CrTsyvhAZiKhQmqP7hz1E"

# You'll also need a corresponding HTTP endpoint for fetching full transaction details
ALCHEMY_HTTP_URL_ETH = "https://eth-mainnet.g.alchemy.com/v2/CrTsyvhAZiKhQmqP7hz1E"

# Initialize Web3 provider for HTTP requests
w3 = Web3(Web3.HTTPProvider(ALCHEMY_HTTP_URL_ETH))

# --- Helper to connect and stream ---
async def stream_live_transactions():
    print(f"Connecting to Ethereum Mainnet WebSocket: {ALCHEMY_WSS_URL_ETH}")
    
    retry_delay = 1  # Initial delay in seconds
    max_retry_delay = 60 # Maximum delay in seconds

    while True:
        try:
            async with websockets.connect(ALCHEMY_WSS_URL_ETH, ping_interval=20, ping_timeout=20) as ws:
                print("WebSocket connection established.")
                # Reset delay on successful connection
                retry_delay = 1

                # Subscribe to new pending transactions
                # This gives us transaction hashes
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newPendingTransactions"]
                }))
                
                subscription_response = json.loads(await ws.recv())
                print(f"Subscription response: {subscription_response}")

                print("Waiting for new pending transactions...")
                while True:
                    try:
                        message = json.loads(await ws.recv())
                        if "params" in message and "result" in message["params"]:
                            tx_hash = message["params"]["result"]
                            # For pending transactions, sometimes the full details aren't immediately available
                            # We'll try to fetch it and handle potential not found errors
                            try:
                                full_tx = w3.eth.get_transaction(tx_hash)
                                
                                # IMPORTANT: Add a check for 'from' being None before using it
                                if full_tx is None or full_tx['from'] is None:
                                    print(f"Skipping transaction {tx_hash}: 'from' address is None.")
                                    continue # Skip to the next message

                                # Yield the transaction as a dictionary
                                yield {
                                    'hash': full_tx.hash.hex(),
                                    'from': full_tx['from'],
                                    'to': full_tx['to'],
                                    'value': w3.from_wei(full_tx.value, 'ether'), # Convert value from Wei to Ether
                                    'gasPrice': w3.from_wei(full_tx.gasPrice, 'gwei'), # Convert gasPrice from Wei to Gwei
                                    'blockNumber': full_tx.blockNumber,
                                    'timeStamp': int(time.time()), # Approximate timestamp for pending tx
                                    # Add other relevant fields your pipeline expects
                                    'isError': 0, # Assume no error until confirmed post-mine
                                    'gasUsed': 0, # Not available for pending tx
                                    'transactionIndex': 0, # Not available for pending tx
                                    'nonce': full_tx.nonce,
                                    # Placeholder for 'label', which would be determined by our model
                                    'label': 'unknown',
                                    'wallet_address': full_tx['from'] # Assuming the sender is the wallet of interest
                                }
                                print(f"Streamed live transaction: {full_tx.hash.hex()}")
                            except TransactionNotFound:
                                print(f"Transaction {tx_hash} not found yet, retrying or skipping.")
                            except Exception as e:
                                print(f"Error fetching transaction {tx_hash}: {e}")
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"WebSocket connection closed unexpectedly: {e}. Attempting to reconnect...")
                        break # Break inner loop to trigger reconnection
                    except Exception as e:
                        print(f"An error occurred within the transaction streaming loop: {e}")
                        # This specific error is within the message processing, not the connection itself
                        await asyncio.sleep(1) # Small delay to prevent tight loop on persistent message errors

        except (websockets.exceptions.WebSocketException, OSError) as e:
            print(f"WebSocket connection failed: {e}. Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay) # Exponential backoff
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay) # Exponential backoff


# --- Main execution block ---
async def main_async():
    print("Running stream_transactions.py to stream live Ethereum Mainnet transactions.")
    try:
        async for tx in stream_live_transactions():
            print(f"Processed live transaction: From={tx.get('from')}, To={tx.get('to')}, Value={tx.get('value')} ETH, Reason={tx.get('label')}")
            # Here, you would typically pass 'tx' to your realtime_inference_service
            # For this demonstration, we'll just print it.
            # To integrate: you would instantiate the realtime_inference_service and call its inference function here.
            # Example (conceptual):
            # inference_service.perform_realtime_inference(tx, wallet_history_df) 

    except KeyboardInterrupt:
        print("Live transaction streaming stopped by user.")
    except Exception as e:
        print(f"Error during main_async execution: {e}")

if __name__ == "__main__":
    asyncio.run(main_async()) 