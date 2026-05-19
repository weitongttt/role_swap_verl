"""
Quick test: verify N-site TCP exchange server groups correctly for N=2, 3, 4.
"""
import asyncio
import pickle
import sys
import os
import importlib.util

# Direct import to avoid pulling in ray via verl.__init__
_tcp_exchange_path = os.path.join(
    os.path.dirname(__file__),
    "verl", "verl", "experimental", "fully_async_policy", "tcp_exchange.py",
)
_spec = importlib.util.spec_from_file_location("tcp_exchange", _tcp_exchange_path)
_tcp_exchange = importlib.util.module_from_spec(_spec)
sys.modules["tcp_exchange"] = _tcp_exchange  # Required for Python 3.14 dataclass resolution
_spec.loader.exec_module(_tcp_exchange)
TcpExchangeServer = _tcp_exchange.TcpExchangeServer
TcpExchangeClient = _tcp_exchange.TcpExchangeClient


async def test_n_site_exchange(n_sites: int, port: int):
    """Test that N sites can push samples and all receive grouped results."""
    server = TcpExchangeServer(host="127.0.0.1", port=port, expected_per_hash=n_sites)
    srv = await asyncio.start_server(server.handle, "127.0.0.1", port)

    run_id = "test_run"
    prompt_hash = "abc123"

    clients = [
        TcpExchangeClient(host="127.0.0.1", port=port, run_id=run_id, site_id=str(i))
        for i in range(n_sites)
    ]

    # Push one sample from each site
    for i, client in enumerate(clients):
        payload = pickle.dumps(f"sample_from_site_{i}")
        result = await client.push_grouped_async(prompt_hash, payload)
        assert result, f"push from site {i} failed"
        print(f"  Site {i} pushed OK")

    # Each site should now be able to pull a group of N samples
    # Use thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    for i, client in enumerate(clients):
        group, remaining = await loop.run_in_executor(None, client.pull_grouped_sync)
        assert len(group) == n_sites, f"Site {i} got group of {len(group)}, expected {n_sites}"
        samples = [pickle.loads(g) for g in group]
        print(f"  Site {i} pulled group: {samples}")
        for j in range(n_sites):
            assert f"sample_from_site_{j}" in samples, f"Site {i} missing sample from site {j}"

    # Check stats
    stats = await loop.run_in_executor(None, clients[0].get_statistics_sync)
    assert stats["expected_per_hash"] == n_sites
    assert len(stats["registered_sites"]) == n_sites
    print(f"  Stats: registered_sites={stats['registered_sites']} expected_per_hash={stats['expected_per_hash']}")

    srv.close()
    await srv.wait_closed()
    print(f"  ✓ {n_sites}-site exchange test PASSED\n")


async def test_partial_push(port: int):
    """Test that with 3 sites, pushing only 2 samples does NOT form a group."""
    n_sites = 3
    server = TcpExchangeServer(host="127.0.0.1", port=port, expected_per_hash=n_sites)
    srv = await asyncio.start_server(server.handle, "127.0.0.1", port)

    run_id = "test_partial"
    prompt_hash = "partial_hash"
    loop = asyncio.get_event_loop()

    clients = [
        TcpExchangeClient(host="127.0.0.1", port=port, run_id=run_id, site_id=str(i))
        for i in range(n_sites)
    ]

    # Push from only 2 of 3 sites
    for i in range(2):
        payload = pickle.dumps(f"partial_sample_{i}")
        await clients[i].push_grouped_async(prompt_hash, payload)

    # Check stats — ready should be 0 for all sites
    stats = await loop.run_in_executor(None, clients[0].get_statistics_sync)
    assert stats["my_ready"] == 0, f"Expected no ready groups, got {stats['my_ready']}"
    print(f"  Stats after partial push: my_ready={stats['my_ready']} (expected 0)")

    # Now push the 3rd sample — should complete the group
    payload = pickle.dumps("partial_sample_2")
    await clients[2].push_grouped_async(prompt_hash, payload)

    # Now all sites should have a ready group
    stats = await loop.run_in_executor(None, clients[0].get_statistics_sync)
    assert stats["my_ready"] == 1, f"Expected 1 ready group, got {stats['my_ready']}"
    print(f"  Stats after complete push: my_ready={stats['my_ready']} (expected 1)")

    srv.close()
    await srv.wait_closed()
    print(f"  ✓ Partial push test PASSED\n")


async def test_backward_compat_2site(port: int):
    """Test backward compatibility: 2-site with side='A' and side='B'."""
    server = TcpExchangeServer(host="127.0.0.1", port=port, expected_per_hash=2)
    srv = await asyncio.start_server(server.handle, "127.0.0.1", port)

    run_id = "compat_run"
    prompt_hash = "compat_hash"
    loop = asyncio.get_event_loop()

    client_a = TcpExchangeClient(host="127.0.0.1", port=port, run_id=run_id, site_id="A")
    client_b = TcpExchangeClient(host="127.0.0.1", port=port, run_id=run_id, site_id="B")

    await client_a.push_grouped_async(prompt_hash, pickle.dumps("from_A"))
    await client_b.push_grouped_async(prompt_hash, pickle.dumps("from_B"))

    group_a, _ = await loop.run_in_executor(None, client_a.pull_grouped_sync)
    group_b, _ = await loop.run_in_executor(None, client_b.pull_grouped_sync)

    assert len(group_a) == 2
    assert len(group_b) == 2
    samples_a = set(pickle.loads(g) for g in group_a)
    samples_b = set(pickle.loads(g) for g in group_b)
    assert samples_a == {"from_A", "from_B"}
    assert samples_b == {"from_A", "from_B"}

    srv.close()
    await srv.wait_closed()
    print("  ✓ Backward compat (A/B) test PASSED\n")


async def test_multiple_hashes(port: int):
    """Test that multiple prompt hashes work correctly with 3 sites."""
    n_sites = 3
    server = TcpExchangeServer(host="127.0.0.1", port=port, expected_per_hash=n_sites)
    srv = await asyncio.start_server(server.handle, "127.0.0.1", port)

    run_id = "test_multi_hash"
    loop = asyncio.get_event_loop()

    clients = [
        TcpExchangeClient(host="127.0.0.1", port=port, run_id=run_id, site_id=str(i))
        for i in range(n_sites)
    ]

    # Push for two different hashes
    for hash_id in ["hash_A", "hash_B"]:
        for i, client in enumerate(clients):
            payload = pickle.dumps(f"site_{i}_{hash_id}")
            await client.push_grouped_async(hash_id, payload)

    # Each site should have 2 ready groups
    for i, client in enumerate(clients):
        for _ in range(2):
            group, _ = await loop.run_in_executor(None, client.pull_grouped_sync)
            assert len(group) == n_sites, f"Site {i} got group of {len(group)}"
            samples = [pickle.loads(g) for g in group]
            print(f"  Site {i} pulled: {samples}")

    srv.close()
    await srv.wait_closed()
    print("  ✓ Multiple hashes test PASSED\n")


async def main():
    print("=" * 60)
    print("N-Site TCP Exchange Server Tests")
    print("=" * 60)

    base_port = 19100

    print("\n[Test 1] 2-site exchange")
    await test_n_site_exchange(2, base_port)

    print("[Test 2] 3-site exchange")
    await test_n_site_exchange(3, base_port + 1)

    print("[Test 3] 4-site exchange")
    await test_n_site_exchange(4, base_port + 2)

    print("[Test 4] Partial push (3-site, only 2 push)")
    await test_partial_push(base_port + 3)

    print("[Test 5] Backward compat (A/B string IDs)")
    await test_backward_compat_2site(base_port + 4)

    print("[Test 6] Multiple hashes (3-site)")
    await test_multiple_hashes(base_port + 5)

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
