from setuptools import find_packages, setup

setup(
  name="hitl-agent",
  version="0.1.0",
  description="human-in-the-loop agent for TPU kernel generation, orchestrating the entire process and integrating GPU to JAX conversion capabilities.",
  author="Your Name",
  author_email="your.email@example.com",
  packages=find_packages(),
  python_requires=">=3.7",
  install_requires=[],
)
