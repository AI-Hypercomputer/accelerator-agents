import sys
import unittest
from pathlib import Path

server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

from fastapi.testclient import TestClient
from tpu_server import (
  app,
)

client = TestClient(app)


class TestTPUServer(unittest.TestCase):
  def test_health(self):
    response = client.get("/health")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(
      response.json(), {"status": "healthy", "service": "maxkernel-tpu-server"}
    )

  def test_job_submission_and_polling(self):
    payload = {
      "action": "compilation_test",
      "code_request": {"code": "print('hello world')", "timeout": 10},
    }
    sub_resp = client.post("/submit", json=payload)
    self.assertEqual(sub_resp.status_code, 200)
    data = sub_resp.json()
    self.assertIn("job_id", data)
    job_id = data["job_id"]
    self.assertEqual(data["status"], "queued")
    self.assertEqual(data["action"], "compilation_test")

    status_resp = client.get(f"/job/{job_id}")
    self.assertEqual(status_resp.status_code, 200)
    job_data = status_resp.json()
    self.assertEqual(job_data["job_id"], job_id)
    self.assertEqual(job_data["action"], "compilation_test")


if __name__ == "__main__":
  unittest.main()
