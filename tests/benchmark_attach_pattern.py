# do not commit
# claude coded to test attach/close pattern vs persistent attachment

import time
from multiprocessing import shared_memory

import torch

"""
do not commit, for PR review

=== Pattern 1: Persistent Attachment ===
(Attach once, copy multiple times, close once)
  Attach: 3.0ms
  FIRST copy (warmup): 2426.6ms (4.4 GB/s) <- page faults here!
  Iter 1: copy=149.4ms (71.9 GB/s)
  Iter 2: copy=142.4ms (75.4 GB/s)
  Iter 3: copy=142.4ms (75.4 GB/s)
  Iter 4: copy=141.3ms (76.0 GB/s)
  Iter 5: copy=142.1ms (75.5 GB/s)

=== Pattern 2: Attach-per-Copy ===
(Attach, copy, close on every operation)
  Iter 1: attach=0.1ms, view=0.0ms, copy=452.6ms, close=723.5ms, total=1176.3ms (9.1 GB/s)
  Iter 2: attach=3.2ms, view=0.0ms, copy=526.5ms, close=287.1ms, total=816.8ms (13.1 GB/s)
  Iter 3: attach=1.6ms, view=0.0ms, copy=519.3ms, close=259.3ms, total=780.2ms (13.8 GB/s)
  Iter 4: attach=0.7ms, view=0.0ms, copy=521.6ms, close=268.5ms, total=790.9ms (13.6 GB/s)
  Iter 5: attach=3.1ms, view=0.0ms, copy=518.3ms, close=270.6ms, total=792.1ms (13.6 GB/s)
"""


def test_attach_patterns(size_gb: float = 10.0, iterations: int = 5):
    """Compare persistent vs attach-per-operation."""
    size_bytes = int(size_gb * 1024 * 1024 * 1024)
    numel = size_bytes // 4
    name = "pattern_test_shm"

    # Cleanup
    try:
        existing = shared_memory.SharedMemory(name=name)
        existing.close()
        existing.unlink()
    except FileNotFoundError:
        pass

    # Create shared memory (simulates storage volume)
    shm_owner = shared_memory.SharedMemory(name=name, create=True, size=size_bytes)

    # Create and warm up source tensor
    src = torch.randn(numel, dtype=torch.float32)

    print("\n=== Pattern 1: Persistent Attachment ===")
    print("(Attach once, copy multiple times, close once)")
    # Attach once
    t_attach = time.perf_counter()
    shm_client = shared_memory.SharedMemory(name=name)
    shm_tensor = torch.frombuffer(shm_client.buf, dtype=torch.float32, count=numel)
    attach_time = (time.perf_counter() - t_attach) * 1000
    print(f"  Attach: {attach_time:.1f}ms")

    # First copy (warmup) - should be slow due to page faults
    t_warmup = time.perf_counter()
    shm_tensor.copy_(src)
    warmup_time = time.perf_counter() - t_warmup
    warmup_throughput = (size_bytes / 1e9) / warmup_time
    print(
        f"  FIRST copy (warmup): {warmup_time*1000:.1f}ms ({warmup_throughput:.1f} GB/s) <- page faults here!"
    )

    for i in range(iterations):
        src.normal_()
        t0 = time.perf_counter()
        shm_tensor.copy_(src)
        elapsed = time.perf_counter() - t0
        throughput = (size_bytes / 1e9) / elapsed
        print(f"  Iter {i+1}: copy={elapsed*1000:.1f}ms ({throughput:.1f} GB/s)")

    del shm_tensor
    shm_client.close()

    print("\n=== Pattern 2: Attach-per-Copy (like TorchStore) ===")
    print("(Attach, copy, close on every operation)")

    for i in range(iterations):
        src.normal_()
        t_start = time.perf_counter()

        # Attach
        t0 = time.perf_counter()
        shm_client = shared_memory.SharedMemory(name=name)
        t1 = time.perf_counter()
        attach_time = (t1 - t0) * 1000

        # Create tensor view
        t2 = time.perf_counter()
        shm_tensor = torch.frombuffer(shm_client.buf, dtype=torch.float32, count=numel)
        t3 = time.perf_counter()
        view_time = (t3 - t2) * 1000

        # Copy
        t4 = time.perf_counter()
        shm_tensor.copy_(src)
        t5 = time.perf_counter()
        copy_time = (t5 - t4) * 1000

        # Close
        t6 = time.perf_counter()
        del shm_tensor
        shm_client.close()
        t7 = time.perf_counter()
        close_time = (t7 - t6) * 1000

        total = time.perf_counter() - t_start
        throughput = (size_bytes / 1e9) / total
        print(
            f"  Iter {i+1}: attach={attach_time:.1f}ms, view={view_time:.1f}ms, "
            f"copy={copy_time:.1f}ms, close={close_time:.1f}ms, "
            f"total={total*1000:.1f}ms ({throughput:.1f} GB/s)"
        )

    print("\n=== Pattern 3: Attach-per-Copy without close ===")
    print("(Attach, copy on every operation, but don't close)")

    clients = []  # Keep references to prevent GC
    for i in range(iterations):
        src.normal_()
        t_start = time.perf_counter()

        # Attach
        t0 = time.perf_counter()
        shm_client = shared_memory.SharedMemory(name=name)
        t1 = time.perf_counter()
        attach_time = (t1 - t0) * 1000

        # Create tensor view
        t2 = time.perf_counter()
        shm_tensor = torch.frombuffer(shm_client.buf, dtype=torch.float32, count=numel)
        t3 = time.perf_counter()
        view_time = (t3 - t2) * 1000

        # Copy
        t4 = time.perf_counter()
        shm_tensor.copy_(src)
        t5 = time.perf_counter()
        copy_time = (t5 - t4) * 1000

        total = time.perf_counter() - t_start
        throughput = (size_bytes / 1e9) / total
        print(
            f"  Iter {i+1}: attach={attach_time:.1f}ms, view={view_time:.1f}ms, "
            f"copy={copy_time:.1f}ms, total={total*1000:.1f}ms ({throughput:.1f} GB/s)"
        )
        clients.append((shm_client, shm_tensor))

    # Cleanup
    for shm_client, shm_tensor in clients:
        del shm_tensor
        shm_client.close()
    del src
    shm_owner.close()
    shm_owner.unlink()


if __name__ == "__main__":
    test_attach_patterns(10.0)
