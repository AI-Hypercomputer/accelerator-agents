import urllib.request
from html.parser import HTMLParser


class SphinxDocParser(HTMLParser):
  def __init__(self):
    super().__init__()
    self.in_body = False
    self.body_depth = 0
    self.depth = 0
    self.content = []
    self.in_code = False
    self.tag_count = 0

  def handle_starttag(self, tag, attrs):
    self.depth += 1
    self.tag_count += 1
    attrs_dict = dict(attrs)

    if self.tag_count < 500:
      print(
        f"Tag: {tag}, class: {attrs_dict.get('class')}, role: {attrs_dict.get('role')}"
      )

    classes = attrs_dict.get("class", "").split()
    if (
      "body" in classes
      or "bd-article" in classes
      or attrs_dict.get("role") == "main"
    ):
      if not self.in_body:
        self.in_body = True
        self.body_depth = self.depth
        print(f"Found body at depth {self.depth} with tag {tag}")

    if not self.in_body:
      return

    if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
      level = int(tag[1])
      self.content.append("\n" + "#" * level + " ")
    elif tag == "p":
      self.content.append("\n")
    elif tag == "pre":
      self.in_code = True
      self.content.append("\n```python\n")
    elif tag == "br":
      self.content.append("\n")

  def handle_endtag(self, tag):
    if self.in_body and tag == "div" and self.depth == self.body_depth:
      self.in_body = False

    if not self.in_body:
      self.depth -= 1
      return

    if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
      self.content.append("\n")
    elif tag == "p":
      self.content.append("\n")
    elif tag == "pre":
      self.in_code = False
      self.content.append("```\n")

    self.depth -= 1

  def handle_data(self, data):
    if not self.in_body:
      return

    if self.in_code:
      self.content.append(data)
    else:
      cleaned_data = data.strip()
      if cleaned_data:
        self.content.append(cleaned_data + " ")

  def get_markdown(self):
    return "".join(self.content)


def fetch_and_clean(url):
  print(f"Fetching {url}...")
  try:
    req = urllib.request.Request(
      url,
      headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      },
    )
    with urllib.request.urlopen(req) as response:
      html = response.read().decode("utf-8")
      parser = SphinxDocParser()
      parser.feed(html)
      return parser.get_markdown()
  except Exception as e:
    print(f"Error fetching {url}: {e}")
    return ""


def main():
  urls = [
    "https://docs.jax.dev/en/latest/pallas/quickstart.html",
    "https://docs.jax.dev/en/latest/pallas/pipelining.html",
    "https://docs.jax.dev/en/latest/pallas/grid_blockspec.html",
    "https://docs.jax.dev/en/latest/pallas/tpu/details.html",
    "https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html",
    "https://docs.jax.dev/en/latest/pallas/tpu/matmul.html",
    "https://docs.jax.dev/en/latest/pallas/tpu/sparse.html",
  ]

  combined_content = ""
  for url in urls:
    content = fetch_and_clean(url)
    combined_content += content + "\n\n"

  target_file = "/usr/local/google/home/shangkunwang/kernel_agent/accelerator-agents/MaxKernel/auto_agent/knowledge_base/pallas_docs.py"

  print(f"Updating {target_file}...")
  try:
    with open(target_file, "w") as f:
      f.write("PROMPT = r'''\n")
      f.write(combined_content)
      f.write("'''\n")
    print("Successfully updated file.")
  except Exception as e:
    print(f"Error writing to file: {e}")


if __name__ == "__main__":
  main()
