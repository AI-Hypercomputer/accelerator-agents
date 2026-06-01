import asyncio
import logging
import time

import aiohttp


async def call_eval_server_async(
  session: aiohttp.ClientSession,
  eval_server_url: str,
  payload: dict,
  poll_interval: int = 10,
  client_wait_timeout: int = 3600 * 3,  # Default to 3 hours
) -> dict:
  """Calls the evaluation server asynchronously and polls for status.

  Args:
      session: aiohttp.ClientSession to use.
      eval_server_url: Base URL of the eval server (e.g.,
        "http://localhost:1245").
      payload: The request payload.
      poll_interval: Seconds to wait between polls.
      client_wait_timeout: Max seconds to wait for task completion.

  Returns:
      The result from the evaluation server.
  """
  # 1. Submit the task
  submit_url = f"{eval_server_url}/evaluate"
  logging.info(f"Submitting async task to {submit_url}")

  payload = payload.copy()
  payload["client_wait_timeout"] = client_wait_timeout

  async with session.post(submit_url, json=payload) as response:
    if response.status != 202:
      error_text = await response.text()
      raise Exception(
        f"Failed to submit task. Status: {response.status}, Error: {error_text}"
      )
    resp_data = await response.json()
    task_id = resp_data["task_id"]
    logging.info(f"Task submitted successfully. ID: {task_id}")

  # 2. Poll for status
  start_time = time.time()
  status_url = f"{eval_server_url}/status/{task_id}"

  while True:
    if time.time() - start_time > client_wait_timeout:
      raise Exception(
        f"Client timed out waiting for task {task_id} after {client_wait_timeout} seconds"
      )

    async with session.get(status_url) as response:
      if response.status != 200:
        error_text = await response.text()
        raise Exception(
          f"Failed to get task status. Status: {response.status}, Error: {error_text}"
        )

      status_data = await response.json()
      status = status_data["status"]

      if status == "success":
        logging.info(f"Task {task_id} completed successfully.")
        return status_data["result"]
      elif status in ["failed", "timeout"]:
        raise Exception(
          f"Task {task_id} ended with status {status}: {status_data.get('error')}"
        )

      logging.info(
        f"Task {task_id} status: {status}. Waiting {poll_interval}s..."
      )
      await asyncio.sleep(poll_interval)
