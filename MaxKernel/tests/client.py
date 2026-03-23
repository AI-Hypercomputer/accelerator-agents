import numpy as np
import requests

# IMPORTANT: Replace with your server's actual internal IP address
SERVER_IP = "10.130.0.155"
SERVER_URL = f"http://{SERVER_IP}:5000/receive_data"


def send_data_to_server():
  """Creates a NumPy array and sends it to the server."""

  # 1. Create some data to send
  my_array = np.arange(12, dtype=np.float32).reshape(3, 4)
  print("📦 Data to be sent from client:")
  print(my_array)

  # 2. Prepare the data for transfer
  # We convert the NumPy array to a standard Python list to serialize it into JSON.
  payload = {"array_data": my_array.tolist()}

  # 3. Send the HTTP POST request
  try:
    print(f"\n🚀 Sending data to {SERVER_URL}...")
    response = requests.post(SERVER_URL, json=payload, timeout=10)

    # Check if the request was successful
    response.raise_for_status()

    print("✅ Server responded successfully!")
    print(f"Response JSON: {response.json()}")

  except requests.exceptions.RequestException as e:
    print(f"\n❌ Failed to connect to server: {e}")


if __name__ == "__main__":
  send_data_to_server()
