import yaml

from tpu_kernel_gen.openevolve.prompts import pallas_apis, pallas_docs, prompt_template


def prep_config():
  # Read base configuration
  with open("base_config.yaml", "r") as f:
    config = yaml.safe_load(f)

  prompt = prompt_template.PROMPT.format(
    pallas_docs=pallas_docs.PROMPT,
    pallas_apis=pallas_apis.PROMPT,
  )

  # Set the system message
  config["prompt"]["system_message"] = prompt

  # Write to config.yaml
  with open("config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

  print("Generated config.yaml")


if __name__ == "__main__":
  prep_config()
