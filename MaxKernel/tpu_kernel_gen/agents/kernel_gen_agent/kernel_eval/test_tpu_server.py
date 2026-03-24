import pytest
from fastapi.testclient import TestClient
from tpu_server import app

client = TestClient(app)


class TestTPUServer:
  def test_health_check(self):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

  def test_simple_python_execution(self):
    """Test basic Python code execution"""
    code_request = {"code": "print('Hello, World!')", "language": "python"}
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert result["output"] == "Hello, World!\n"
    assert result["error"] is None
    assert result["exit_code"] == 0

  def test_python_with_calculations(self):
    """Test Python code with calculations"""
    code_request = {
      "code": "result = 2 + 3\nprint(f'Result: {result}')",
      "language": "python",
    }
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert "Result: 5" in result["output"]
    assert result["exit_code"] == 0

  def test_python_with_imports(self):
    """Test Python code with standard library imports"""
    code_request = {
      "code": "import math\nprint(f'Pi: {math.pi:.2f}')",
      "language": "python",
    }
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert "Pi: 3.14" in result["output"]
    assert result["exit_code"] == 0

  def test_python_with_error(self):
    """Test Python code that raises an exception"""
    code_request = {
      "code": "print('Before error')\nraise ValueError('Test error')\nprint('After error')",
      "language": "python",
    }
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert "Before error" in result["output"]
    assert "ValueError: Test error" in result["error"]
    assert result["exit_code"] != 0

  def test_python_syntax_error(self):
    """Test Python code with syntax error"""
    code_request = {
      "code": "print('Hello'\nprint('Missing parenthesis')",
      "language": "python",
    }
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]
    assert result["exit_code"] != 0

  def test_unsupported_language(self):
    """Test execution with unsupported language"""
    code_request = {"code": "console.log('Hello');", "language": "javascript"}
    response = client.post("/execute", json=code_request)
    assert response.status_code == 400
    assert "Only Python code execution is supported" in response.json()["detail"]

  def test_empty_code(self):
    """Test execution with empty code"""
    code_request = {"code": "", "language": "python"}
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert result["output"] == ""
    assert result["exit_code"] == 0

  def test_multiline_output(self):
    """Test code that produces multiline output"""
    code_request = {
      "code": "for i in range(3):\n    print(f'Line {i}')",
      "language": "python",
    }
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    lines = result["output"].strip().split("\n")
    assert len(lines) == 3
    assert "Line 0" in lines[0]
    assert "Line 1" in lines[1]
    assert "Line 2" in lines[2]

  def test_code_with_stderr_and_stdout(self):
    """Test code that writes to both stdout and stderr"""
    code_request = {
      "code": "import sys\nprint('stdout message')\nsys.stderr.write('stderr message\\n')",
      "language": "python",
    }
    response = client.post("/execute", json=code_request)
    assert response.status_code == 200

    result = response.json()
    assert "stdout message" in result["output"]
    assert "stderr message" in result["error"]
    assert result["exit_code"] == 0


if __name__ == "__main__":
  pytest.main([__file__])
