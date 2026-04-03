import numpy as np
from flask import Flask, jsonify, request

# Initialize the Flask application
app = Flask(__name__)


@app.route("/receive_data", methods=["POST"])
def receive_data():
  """An endpoint that listens for POST requests with JSON data."""
  print("🚀 Request received!")

  # Get the JSON data sent from the client
  payload = request.get_json()

  if not payload or "array_data" not in payload:
    return jsonify({"status": "error", "message": "Missing 'array_data' in payload"}), 400

  # For this example, we'll convert the received list back into a NumPy array
  try:
    # The data arrives as a standard Python list
    received_list = payload["array_data"]
    # Convert it back to a NumPy array on the server side
    data_as_np_array = np.array(received_list)

    print("✅ Successfully received and converted data to NumPy array:")
    print(data_as_np_array)
    print(f"Shape: {data_as_np_array.shape}, Dtype: {data_as_np_array.dtype}")

    # You could now load this array onto the server's TPU if needed
    # e.g., with jax.device_put(data_as_np_array, jax.devices('tpu')[0])

    return jsonify({"status": "success", "message": "Data received"}), 200

  except Exception as e:
    print(f"Error processing data: {e}")
    return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
  # Run the app. '0.0.0.0' makes it accessible from other computers on the network.
  app.run(host="0.0.0.0", port=5000)
