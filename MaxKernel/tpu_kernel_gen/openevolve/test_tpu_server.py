import sys

import requests

# Server configuration
SERVER_URL = "http://10.130.0.129:5463"

# Sample kernel code for testing
SAMPLE_CODE = """
import jax
import jax.numpy as jnp
import time
import argparse

def simple_kernel(x, y):
    return jnp.add(x, y)

def optimized_kernel(x, y):
    return jnp.add(x, y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compilation", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    
    if args.compilation:
        try:
            result = simple_kernel(x, y)
            print("Compilation successful")
        except Exception as e:
            print(f"Compilation failed: {e}")
            sys.exit(1)
    
    elif args.correctness:
        result1 = simple_kernel(x, y)
        result2 = optimized_kernel(x, y)
        if jnp.allclose(result1, result2):
            print("Correctness test passed")
        else:
            print("Correctness test failed")
            sys.exit(1)
    
    elif args.performance:
        # Simple kernel timing
        start = time.time()
        for _ in range(100):
            simple_kernel(x, y)
        simple_time = (time.time() - start) / 100
        
        # Optimized kernel timing
        start = time.time()
        for _ in range(100):
            optimized_kernel(x, y)
        optimized_time = (time.time() - start) / 100
        
        print(f"Simple compute average time: {simple_time} s")
        print(f"Optimized compute average time: {optimized_time} s")

if __name__ == "__main__":
    main()
"""


def test_health():
  """Test the health check endpoint"""
  print("Testing /health endpoint...")
  try:
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200
  except Exception as e:
    print(f"Health check failed: {e}")
    return False


def test_compilation():
  """Test the compilation endpoint"""
  print("\nTesting /test_compilation endpoint...")
  try:
    payload = {"code": SAMPLE_CODE, "timeout": 30}
    response = requests.post(f"{SERVER_URL}/test_compilation", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Exit code: {result.get('exit_code', 'N/A')}")
    print(f"Output: {result.get('output', '')}")
    if result.get("error"):
      print(f"Error: {result.get('error')}")
    return response.status_code == 200
  except Exception as e:
    print(f"Compilation test failed: {e}")
    return False


def test_correctness():
  """Test the correctness endpoint"""
  print("\nTesting /test_correctness endpoint...")
  try:
    payload = {"code": SAMPLE_CODE, "timeout": 30}
    response = requests.post(f"{SERVER_URL}/test_correctness", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Exit code: {result.get('exit_code', 'N/A')}")
    print(f"Output: {result.get('output', '')}")
    if result.get("error"):
      print(f"Error: {result.get('error')}")
    return response.status_code == 200
  except Exception as e:
    print(f"Correctness test failed: {e}")
    return False


def test_performance():
  """Test the performance endpoint"""
  print("\nTesting /test_performance endpoint...")
  try:
    payload = {"code": SAMPLE_CODE, "timeout": 30}
    response = requests.post(f"{SERVER_URL}/test_performance", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Exit code: {result.get('exit_code', 'N/A')}")
    print(f"Output: {result.get('output', '')}")
    if result.get("simple_time") is not None:
      print(f"Simple time: {result.get('simple_time')} s")
    if result.get("optimized_time") is not None:
      print(f"Optimized time: {result.get('optimized_time')} s")
    if result.get("error"):
      print(f"Error: {result.get('error')}")
    return response.status_code == 200
  except Exception as e:
    print(f"Performance test failed: {e}")
    return False


def main():
  """Run all tests"""
  print("Starting TPU Server Tests...")
  print("=" * 50)

  # Test all endpoints
  tests = [
    ("Health Check", test_health),
    ("Compilation Test", test_compilation),
    ("Correctness Test", test_correctness),
    ("Performance Test", test_performance),
  ]

  results = []
  for test_name, test_func in tests:
    success = test_func()
    results.append((test_name, success))

  # Print summary
  print("\n" + "=" * 50)
  print("Test Summary:")
  for test_name, success in results:
    status = "PASS" if success else "FAIL"
    print(f"{test_name}: {status}")

  all_passed = all(success for _, success in results)
  print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
  return 0 if all_passed else 1


if __name__ == "__main__":
  sys.exit(main())
