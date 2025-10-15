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
            # The API endpoint for instrument info supports filtering by status. [1]
            response = session.get_instruments_info(
                category=CATEGORY,
                status=status,
                limit=1000,  # Max limit per page
                cursor=cursor
            )

            if response.get("retCode") == 0:
                result_list = response.get("result", {}).get("list", [])
                for item in result_list:
                    # We are only interested in USDT perpetuals
                    if item.get("quoteCoin") == "USDT":
                        symbols.add(item["symbol"])

                cursor = response.get("result", {}).get("nextPageCursor", "")
                if not cursor:
                    break  # No more pages
            else:
                print(f"API Error while fetching status '{status}': {response.get('retMsg')}")
                break
            
            # Respect rate limits
            time.sleep(0.1) 

        except Exception as e:
            print(f"An exception occurred: {e}")
            break
            
    return symbols

def main():
    """Main function to generate the master symbol list."""
    print("Connecting to Bybit API...")
    # Public session, no API key needed for this endpoint. [2]
    session = HTTP()
    
    print("Fetching 'Trading' symbols...")
    trading_symbols = fetch_symbols_by_status(session, "Trading")
    print(f"Found {len(trading_symbols)} currently trading symbols.")

    print("\nFetching 'Closed' (delisted) symbols...")
    closed_symbols = fetch_symbols_by_status(session, "Closed")
    print(f"Found {len(closed_symbols)} closed/delisted symbols.")

    # Combine the sets, which automatically handles any overlaps
    all_symbols = sorted(list(trading_symbols.union(closed_symbols)))
    
    print(f"\nTotal unique symbols found: {len(all_symbols)}")

    # Save to file
    output_path = Path(OUTPUT_FILENAME)
    with output_path.open('w') as f:
        for symbol in all_symbols:
            f.write(f"{symbol}\n")

    print(f"\nSuccessfully saved all symbol names to '{output_path.resolve()}'")


if __name__ == "__main__":
    main()
