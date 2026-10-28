import sys
import time
import numpy as np
from llm_sdk import Small_LLM_Model


def run_benchmark(iterations: int = 10) -> None:
    """Measures model forward pass latency and calculates generation budgets.

    Executes a series of forward passes using a standard test prompt to compute
    average execution latency and standard deviation. Incorporates an initial
    warmup pass to discard cold-start overhead, then outputs a performance
    budget report estimating total affordable token generation within a
    5-minute execution window.

    Args:
        iterations: The number of benchmark passes to execute. Defaults to 10.

    Raises:
        SystemExit: If model initialization fails during setup.
    """
    print("Initializing model for latency benchmarking...")
    try:
        model = Small_LLM_Model()
    except Exception as err:
        print(f"Failed to initialize model: {err}")
        sys.exit(1)
    prompt = "Generate a JSON object describing a user profile."
    input_ids: list[int] = model.encode(prompt).tolist()[0]
    _ = model.get_logits_from_input_ids(input_ids)
    print(f"Benchmarking {iterations} forward passes...")
    latencies: list[float] = []
    for step in range(iterations):
        start_time = time.perf_counter()
        _ = model.get_logits_from_input_ids(input_ids)
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)
        print(f" Pass {step + 1}/{iterations}: {elapsed * 1000:.2f} ms")
    avg_lat = float(np.mean(latencies))
    std_dev = float(np.std(latencies))
    max_tokens_5min = int(300 / avg_lat)
    print("\n--- Performance Budget Report ---")
    print(f"Average Latency : {avg_lat * 1000:.2f} ms ({avg_lat:.4f} s)")
    print(f"Std Deviation   : {std_dev * 1000:.2f} ms")
    print(f"5-Min Cap (300s): ~{max_tokens_5min} total generated tokens")


if __name__ == "__main__":
    run_benchmark()
