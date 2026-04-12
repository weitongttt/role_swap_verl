"""
Entry point for running TcpExchangeServer.
"""

import argparse
import asyncio

from verl.experimental.fully_async_policy.tcp_exchange import TcpExchangeServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()

    server = TcpExchangeServer(host=args.host, port=args.port)
    print(f"[TCP_EXCHANGE] listening on {args.host}:{args.port} (hash-grouped, expected_per_hash=2)", flush=True)
    asyncio.run(server.run_forever())


if __name__ == "__main__":
    main()
