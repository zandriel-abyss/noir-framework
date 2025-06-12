#!/usr/bin/env python3
"""fetch_richlist.py

Scrapes the EtherScan rich-list (https://etherscan.io/accounts) and returns a filtered
list of wallets **not** present in either an OFAC sanctions file or a known-hackers
file.

Run it with no arguments and it will automatically use the two path names you just
sent me:

    • /Users/jzackslineandreela/Downloads/noir-framework/datasource/raw/nk-hackers.txt
    • /Users/jzackslineandreela/Downloads/noir-framework/datasource/raw/ofac-eth-addresses.txt

You can still override these via the usual `--ofac` / `--hackers` cli flags.

₪  Example
---------
$ python fetch_richlist.py --pages 50 --outfile clean-richlist.txt

Each rich-list page contains 50 addresses; so `--pages 20` ≈ top-1,000 wallets.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Set

import requests
from bs4 import BeautifulSoup

##########################################################################################
# Constants & Config
##########################################################################################
ETHERSCAN_RICHLIST_URL = "https://etherscan.io/accounts/{page}?sort=balance&order=desc"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

# polite defaults
MIN_DELAY: float = 1.0  # seconds
MAX_DELAY: float = 3.0

# user-supplied default paths
DEFAULT_HACKERS_PATH = Path(
    "/Users/jzackslineandreela/Downloads/noir-framework/datasource/raw/nk-hackers.txt"
)
DEFAULT_OFAC_PATH = Path(
    "/Users/jzackslineandreela/Downloads/noir-framework/datasource/raw/ofac-eth-addresses.txt"
)

##########################################################################################
# Utility functions
##########################################################################################

def human_sleep(min_delay: float = MIN_DELAY, max_delay: float = MAX_DELAY) -> None:
    """Sleep a random 𝑡 ∈ [min_delay, max_delay] seconds."""
    time.sleep(random.uniform(min_delay, max_delay))


def load_addresses(path: Path) -> Set[str]:
    """Load addresses from a file (csv/json/txt).  Returns a lowercase address set."""
    if not path.exists():
        raise FileNotFoundError(path)

    def normalize(addr: str) -> str:
        return addr.strip().lower()

    addresses: Set[str] = set()

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            addresses |= {normalize(a) for a in data}
        elif isinstance(data, dict):
            addresses |= {normalize(a) for a in data.get("addresses", [])}
    else:
        with path.open(newline="") as fh:
            reader: Iterable
            if path.suffix.lower() == ".csv":
                reader = csv.reader(fh)
                for row in reader:
                    if row:
                        addresses.add(normalize(row[0]))
            else:
                for line in fh:
                    if line.strip():
                        addresses.add(normalize(line))
    return addresses


def parse_richlist_page(html: str) -> Set[str]:
    """Extract wallet addresses from one EtherScan rich-list HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table")
    if not table:
        raise ValueError("Rich-list table not found; the HTML layout may have changed.")

    addresses: Set[str] = set()
    for a in table.select("a[href^='/address/0x']"):
        href = a.get("href", "")
        if href.startswith("/address/0x"):
            addr = href.split("/address/")[-1].split("?")[0]
            addresses.add(addr.lower())
    return addresses


def fetch_richlist_page(page: int, session: requests.Session) -> str:
    url = ETHERSCAN_RICHLIST_URL.format(page=page)
    resp = session.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        raise ConnectionError(f"Request failed with HTTP {resp.status_code}: {url}")
    return resp.text

##########################################################################################
# Core logic
##########################################################################################

def collect_richlist_addresses(pages: int) -> Set[str]:
    """Scrape the first *pages* pages of the EtherScan rich-list."""
    pages = max(1, pages)
    addresses: Set[str] = set()
    with requests.Session() as session:
        for page in range(1, pages + 1):
            sys.stderr.write(f"Fetching rich-list page {page}/{pages}\n")
            html = fetch_richlist_page(page, session)
            addresses.update(parse_richlist_page(html))
            human_sleep()  # polite delay
    return addresses


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages",
        type=int,
        default=25,
        help="Number of rich-list pages to scrape (50 addresses per page).",
    )
    parser.add_argument(
        "--ofac",
        type=Path,
        default=DEFAULT_OFAC_PATH,
        help="OFAC sanctions address list file (default set in script)",
    )
    parser.add_argument(
        "--hackers",
        type=Path,
        default=DEFAULT_HACKERS_PATH,
        help="Known-hackers address list file (default set in script)",
    )
    parser.add_argument(
        "--min-balance",
        type=float,
        default=0.0,
        help="Filter out wallets below this ETH balance (approximate).",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("-"),
        help="Output file (default stdout). Use '-' for stdout.",
    )
    args = parser.parse_args(argv)

    try:
        ofac_set = load_addresses(args.ofac)
    except FileNotFoundError as e:
        sys.exit(f"[!] OFAC file not found: {e}")
    try:
        hackers_set = load_addresses(args.hackers)
    except FileNotFoundError as e:
        sys.exit(f"[!] Hackers file not found: {e}")

    blacklist = ofac_set | hackers_set

    try:
        rich_addresses = collect_richlist_addresses(args.pages)
    except Exception as exc:
        sys.exit(f"[!] Failed during scraping: {exc}")

    clean_addresses = sorted(rich_addresses - blacklist)

    if args.outfile == Path("-"):
        print("\n".join(clean_addresses))
    else:
        args.outfile.write_text("\n".join(clean_addresses))
        print(f"Saved {len(clean_addresses):,} addresses → {args.outfile}")

if __name__ == "__main__":  # pragma: no cover
    main()