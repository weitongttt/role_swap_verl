"""
Entry point for running TcpExchangeServer.
"""

import argparse
import asyncio

from verl.experimental.fully_async_policy.tcp_exchange import TcpExchangeServer


def main():
    ap = argparse.ArgumentParser()
    # 默认监听所有接口，便于多机/多容器训练进程连入；仅需本机时传 --host 127.0.0.1
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=28080)
    ap.add_argument("--max-queue-size", type=int, default=20000)
    args = ap.parse_args()

    server = TcpExchangeServer(host=args.host, port=args.port, max_queue_size=args.max_queue_size)
    print(f"[TCP_EXCHANGE] listening on {args.host}:{args.port} max_queue_size={args.max_queue_size}", flush=True)
    asyncio.run(server.run_forever())


if __name__ == "__main__":
    main()

