"""
Document parser for JAX Pallas documentation.
Fetches and chunks documentation for RAG database.
"""

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
  """Represents a chunk of documentation."""

  content: str
  title: str
  url: str
  section: str
  subsection: Optional[str] = None
  chunk_id: str = ""
  word_count: int = 0
  char_count: int = 0
  metadata: Dict = None

  def __post_init__(self):
    if not self.chunk_id:
      # Create more unique chunk ID including content hash
      content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
      url_hash = hashlib.md5(self.url.encode()).hexdigest()[:8]
      section_hash = hashlib.md5(self.section.encode()).hexdigest()[:8]
      subsection_part = self.subsection if self.subsection else "none"
      subsection_hash = hashlib.md5(subsection_part.encode()).hexdigest()[:8]

      self.chunk_id = f"{url_hash}_{section_hash}_{subsection_hash}_{content_hash}"

    if not self.word_count:
      self.word_count = len(self.content.split())
    if not self.char_count:
      self.char_count = len(self.content)
    if self.metadata is None:
      self.metadata = {}


class PallasDocParser:
  """Parser for JAX Pallas documentation."""

  def __init__(
    self,
    base_url: str = "https://docs.jax.dev/en/latest/pallas/index.html",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    output_dir: str = "parsed_docs",
  ):
    self.base_url = base_url
    self.base_domain = "https://docs.jax.dev"
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True)

    # Initialize text splitter
    self.text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    # Track visited URLs and content hashes to avoid duplicates
    self.visited_urls: Set[str] = set()
    self.content_hashes: Set[str] = set()
    self.chunk_ids: Set[str] = set()

    self.session = requests.Session()
    self.session.headers.update(
      {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
      }
    )

  def normalize_url(self, url: str) -> str:
    """Normalize URL to avoid duplicates from different formats."""
    # Remove fragment identifiers and trailing slashes
    url = url.split("#")[0].rstrip("/")

    # Parse and reconstruct to normalize
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    if parsed.query:
      normalized += f"?{parsed.query}"

    return normalized

  def is_duplicate_content(self, content: str) -> bool:
    """Check if content is a duplicate based on hash."""
    content_hash = hashlib.md5(content.strip().encode()).hexdigest()
    if content_hash in self.content_hashes:
      return True
    self.content_hashes.add(content_hash)
    return False

  def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
    """Fetch and parse a single page."""
    try:
      print(f"Fetching: {url}")
      response = self.session.get(url, timeout=30)
      response.raise_for_status()

      # Parse with BeautifulSoup
      soup = BeautifulSoup(response.content, "html.parser")
      return soup

    except Exception as e:
      print(f"Error fetching {url}: {e}")
      return None

  def extract_pallas_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
    """Extract links to other Pallas documentation pages."""
    links = []

    # Find navigation links and content links
    for link in soup.find_all("a", href=True):
      href = link["href"]

      # Convert relative URLs to absolute
      if href.startswith("/"):
        full_url = urljoin(self.base_domain, href)
      elif href.startswith("http"):
        full_url = href
      else:
        full_url = urljoin(current_url, href)

      # Normalize URL
      normalized_url = self.normalize_url(full_url)

      # Only include Pallas-related pages
      if (
        "pallas" in normalized_url.lower()
        and normalized_url.startswith(self.base_domain)
        and normalized_url not in self.visited_urls
      ):
        links.append(normalized_url)

    return list(set(links))  # Remove duplicates

  def clean_text(self, text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove code block markers that might interfere
    text = re.sub(r"```\w*\n", "```\n", text)

    # Clean up common HTML artifacts
    text = text.replace("\xa0", " ")  # Non-breaking spaces
    text = text.replace("\u200b", "")  # Zero-width spaces

    return text.strip()

  def extract_content_sections(self, soup: BeautifulSoup, url: str) -> List[Dict]:
    """Extract structured content from a documentation page."""
    sections = []

    # Get page title
    title_elem = soup.find("title")
    page_title = title_elem.get_text().strip() if title_elem else "Pallas Documentation"

    # Find main content area
    main_content = (
      soup.find("main")
      or soup.find("div", class_="rst-content")
      or soup.find("div", class_="document")
      or soup.find("body")
    )

    if not main_content:
      return sections

    # Extract sections based on headers
    current_section = {"title": page_title, "content": "", "subsections": []}
    current_subsection = None

    for element in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "pre", "code", "ul", "ol"]):
      if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        # Save previous section if it has content
        if current_section["content"].strip():
          sections.append(current_section.copy())

        # Start new section
        header_text = self.clean_text(element.get_text())
        if element.name in ["h1", "h2"]:
          current_section = {"title": header_text, "content": "", "subsections": [], "level": element.name}
          current_subsection = None
        else:
          # Subsection
          if current_subsection:
            current_section["subsections"].append(current_subsection)
          current_subsection = {"title": header_text, "content": "", "level": element.name}

      elif element.name in ["p", "div", "pre", "ul", "ol"]:
        # Extract text content, preserving code blocks
        if element.name == "pre":
          # Preserve code formatting
          code_content = element.get_text()
          content = f"\n```\n{code_content}\n```\n"
        else:
          content = self.clean_text(element.get_text())

        if content.strip():
          if current_subsection:
            current_subsection["content"] += content + "\n\n"
          else:
            current_section["content"] += content + "\n\n"

    # Add final subsection and section
    if current_subsection:
      current_section["subsections"].append(current_subsection)
    if current_section["content"].strip() or current_section["subsections"]:
      sections.append(current_section)

    return sections

  def chunk_section(self, section: Dict, url: str, page_title: str) -> List[DocumentChunk]:
    """Chunk a documentation section into smaller pieces."""
    chunks = []
    section_title = section.get("title", "")
    section_content = section.get("content", "")

    # Chunk main section content
    if section_content.strip():
      # Skip if content is duplicate
      if self.is_duplicate_content(section_content):
        print(f"Skipping duplicate content in section: {section_title}")
        return chunks

      text_chunks = self.text_splitter.split_text(section_content)
      for i, chunk_text in enumerate(text_chunks):
        # Skip empty or very short chunks
        if len(chunk_text.strip()) < 50:
          continue

        chunk = DocumentChunk(
          content=chunk_text,
          title=page_title,
          url=url,
          section=section_title,
          metadata={
            "chunk_index": i,
            "total_chunks": len(text_chunks),
            "section_level": section.get("level", "unknown"),
            "timestamp": datetime.now().isoformat(),
          },
        )

        # Check for duplicate chunk IDs
        if chunk.chunk_id not in self.chunk_ids:
          self.chunk_ids.add(chunk.chunk_id)
          chunks.append(chunk)
        else:
          print(f"Skipping duplicate chunk ID: {chunk.chunk_id}")

    # Chunk subsections
    for subsection in section.get("subsections", []):
      subsection_content = subsection.get("content", "")
      if subsection_content.strip():
        # Skip if content is duplicate
        if self.is_duplicate_content(subsection_content):
          print(f"Skipping duplicate subsection content: {subsection.get('title', '')}")
          continue

        text_chunks = self.text_splitter.split_text(subsection_content)
        for i, chunk_text in enumerate(text_chunks):
          # Skip empty or very short chunks
          if len(chunk_text.strip()) < 50:
            continue

          chunk = DocumentChunk(
            content=chunk_text,
            title=page_title,
            url=url,
            section=section_title,
            subsection=subsection.get("title", ""),
            metadata={
              "chunk_index": i,
              "total_chunks": len(text_chunks),
              "section_level": section.get("level", "unknown"),
              "subsection_level": subsection.get("level", "unknown"),
              "timestamp": datetime.now().isoformat(),
            },
          )

          # Check for duplicate chunk IDs
          if chunk.chunk_id not in self.chunk_ids:
            self.chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)
          else:
            print(f"Skipping duplicate chunk ID: {chunk.chunk_id}")

    return chunks

  def parse_page(self, url: str) -> List[DocumentChunk]:
    """Parse a single documentation page and return chunks."""
    normalized_url = self.normalize_url(url)

    if normalized_url in self.visited_urls:
      return []

    self.visited_urls.add(normalized_url)
    soup = self.fetch_page(normalized_url)

    if not soup:
      return []

    # Extract page title
    title_elem = soup.find("title")
    page_title = title_elem.get_text().strip() if title_elem else "Pallas Documentation"

    # Extract structured content
    sections = self.extract_content_sections(soup, normalized_url)

    # Chunk all sections
    all_chunks = []
    for section in sections:
      chunks = self.chunk_section(section, normalized_url, page_title)
      all_chunks.extend(chunks)

    print(f"Extracted {len(all_chunks)} unique chunks from {normalized_url}")
    return all_chunks

  def crawl_documentation(self, max_pages: int = 50) -> List[DocumentChunk]:
    """Crawl Pallas documentation starting from the index page."""
    all_chunks = []
    urls_to_visit = [self.normalize_url(self.base_url)]
    pages_visited = 0

    while urls_to_visit and pages_visited < max_pages:
      current_url = urls_to_visit.pop(0)

      if current_url in self.visited_urls:
        continue

      # Parse current page
      chunks = self.parse_page(current_url)
      all_chunks.extend(chunks)
      pages_visited += 1

      # Find more pages to visit
      soup = self.fetch_page(current_url)
      if soup:
        new_links = self.extract_pallas_links(soup, current_url)
        # Filter out already visited URLs
        new_links = [link for link in new_links if link not in self.visited_urls]
        urls_to_visit.extend(new_links)

      # Be respectful - add delay between requests
      time.sleep(1)

    print(f"Crawled {pages_visited} pages, extracted {len(all_chunks)} unique chunks")
    print(f"Skipped {len(self.content_hashes) - len(all_chunks)} duplicate content pieces")
    return all_chunks

  def save_chunks(self, chunks: List[DocumentChunk], filename: str = "pallas_chunks.json"):
    """Save chunks to JSON file."""
    output_path = self.output_dir / filename

    # Convert chunks to dictionaries
    chunks_data = [asdict(chunk) for chunk in chunks]

    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_path}")

  def save_chunks_csv(self, chunks: List[DocumentChunk], filename: str = "pallas_chunks.csv"):
    """Save chunks to CSV file for easy import into databases."""
    import csv

    output_path = self.output_dir / filename

    with open(output_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)

      # Write header
      writer.writerow(
        ["chunk_id", "content", "title", "url", "section", "subsection", "word_count", "char_count", "metadata"]
      )

      # Write chunks
      for chunk in chunks:
        writer.writerow(
          [
            chunk.chunk_id,
            chunk.content,
            chunk.title,
            chunk.url,
            chunk.section,
            chunk.subsection or "",
            chunk.word_count,
            chunk.char_count,
            json.dumps(chunk.metadata),
          ]
        )

    print(f"Saved {len(chunks)} chunks to {output_path}")

  def get_stats(self, chunks: List[DocumentChunk]) -> Dict:
    """Get statistics about the parsed chunks."""
    if not chunks:
      return {}

    total_words = sum(chunk.word_count for chunk in chunks)
    total_chars = sum(chunk.char_count for chunk in chunks)
    unique_urls = len(set(chunk.url for chunk in chunks))
    unique_sections = len(set(chunk.section for chunk in chunks))
    unique_chunk_ids = len(set(chunk.chunk_id for chunk in chunks))

    return {
      "total_chunks": len(chunks),
      "unique_chunk_ids": unique_chunk_ids,
      "total_words": total_words,
      "total_characters": total_chars,
      "average_chunk_words": total_words / len(chunks),
      "average_chunk_chars": total_chars / len(chunks),
      "unique_pages": unique_urls,
      "unique_sections": unique_sections,
      "duplicate_detection_stats": {
        "content_hashes_tracked": len(self.content_hashes),
        "chunk_ids_tracked": len(self.chunk_ids),
        "urls_visited": len(self.visited_urls),
      },
      "urls": list(set(chunk.url for chunk in chunks)),
    }


def main():
  """Main function to run the document parser."""
  parser = PallasDocParser(
    chunk_size=800,  # Smaller chunks for better RAG retrieval
    chunk_overlap=100,
    output_dir="pallas_docs_parsed",
  )

  print("Starting Pallas documentation parsing...")

  # Crawl and parse documentation
  chunks = parser.crawl_documentation(max_pages=20)

  if chunks:
    # Save in multiple formats
    parser.save_chunks(chunks, "pallas_chunks.json")
    parser.save_chunks_csv(chunks, "pallas_chunks.csv")

    # Print statistics
    stats = parser.get_stats(chunks)
    print("\n=== Parsing Statistics ===")
    for key, value in stats.items():
      if key != "urls":
        print(f"{key}: {value}")

    print("\nParsed pages:")
    for url in stats.get("urls", []):
      print(f"  - {url}")

  else:
    print("No chunks were extracted. Check the URL and network connection.")


if __name__ == "__main__":
  main()
