import time
from pathlib import Path
from pybit.unified_trading import HTTP

# --- Configuration ---
OUTPUT_FILENAME = "master_symbol_list.txt"
CATEGORY = "linear"

def fetch_symbols_by_status(session: HTTP, status: str) -> set:
    """Fetches all symbols for a given status, handling pagination."""
    symbols = set()
    cursor = ""
    while True:
        try:
            response = session.get_instruments_info(
                category=CATEGORY,
                status=status,
                limit=1000,
                cursor=cursor
            )

            if response.get("retCode") == 0:
                result_list = response["result"]["list"]
                for item in result_list:
                    if item.get("quoteCoin") == "USDT":
                        symbols.add(item["symbol"])
                cursor = response["result"].get("nextPageCursor", "")
                if not cursor:
                    break
            else:
                print(f"API Error while fetching status '{status}': {response.get('retMsg')}")
                break

            time.sleep(0.1)  # Respect rate limits

        except Exception as e:
            print(f"An exception occurred: {e}")
            break

    return symbols

def main():
    print("Connecting to Bybit API...")
    session = HTTP()  # Public session for instrument info

    # 1) Fetch active trading symbols
    print("Fetching 'Trading' symbols...")
    trading_symbols = fetch_symbols_by_status(session, "Trading")
    print(f"Found {len(trading_symbols)} currently trading symbols.")

    # 2) (Optional) Fetch closed/delisted symbols
    print("\nFetching 'Closed' (delisted) symbols...")
    closed_symbols = fetch_symbols_by_status(session, "Closed")
    print(f"Found {len(closed_symbols)} closed/delisted symbols.")

    # 3) Subtract closed from trading to ensure none of the delisted get through
    valid_symbols = sorted(trading_symbols - closed_symbols)
    print(f"\nTotal unique non‑delisted symbols: {len(valid_symbols)}")

    # 4) Write out only the valid symbols
    output_path = Path(OUTPUT_FILENAME)
    with output_path.open('w') as f:
        for symbol in valid_symbols:
            f.write(f"{symbol}\n")

    print(f"\nSuccessfully saved non‑delisted symbols to '{output_path.resolve()}'")

if __name__ == "__main__":
    main()
