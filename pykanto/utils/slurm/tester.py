"""Test launch ray processes on a slurm scheduler."""

from collections import Counter
import os
import sys
import time
import ray
import psutil


@ray.remote
def f():
    print("hello")
    time.sleep(60)
    return ray._private.services.get_node_ip_address()


if __name__ == "__main__":

    redis_password = sys.argv[1]
    ray.init(address=os.environ["ip_head"], _redis_password=redis_password)

    print(f"Nodes in the Ray cluster: {ray.nodes()}")
    print(f"ncpus(logical=False): {psutil.cpu_count(logical=False)}")
    print(f"ncpus(logical=True): {psutil.cpu_count(logical=True)}")

    print(ray.cluster_resources())
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(100)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)
