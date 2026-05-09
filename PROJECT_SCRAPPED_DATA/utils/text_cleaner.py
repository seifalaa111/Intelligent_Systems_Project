"""
MIDAN Data Pipeline — Text Cleaner
Converts raw HTML to clean text and normalizes content.
"""

import re
import html
from bs4 import BeautifulSoup, Comment


def html_to_text(raw_html: str) -> str:
    """
    Convert raw HTML to clean, readable text.
    Removes scripts, styles, comments, and excessive whitespace.
    """
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "lxml")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "noscript", "iframe",
                               "svg", "canvas", "nav", "footer", "header"]):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Get text
    text = soup.get_text(separator=" ", strip=True)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def extract_meta(raw_html: str) -> dict:
    """
    Extract meta tags from HTML — description, keywords, og:tags.
    Returns a dict of meta values.
    """
    if not raw_html:
        return {}

    soup = BeautifulSoup(raw_html, "lxml")
    meta = {}

    # Title
    title_tag = soup.find("title")
    if title_tag:
        meta["title"] = title_tag.get_text(strip=True)

    # Meta description
    desc_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
    if desc_tag and desc_tag.get("content"):
        meta["description"] = desc_tag["content"].strip()

    # Meta keywords
    kw_tag = soup.find("meta", attrs={"name": re.compile(r"keywords", re.I)})
    if kw_tag and kw_tag.get("content"):
        meta["keywords"] = kw_tag["content"].strip()

    # Open Graph tags
    for og_tag in soup.find_all("meta", attrs={"property": re.compile(r"^og:")}):
        prop = og_tag.get("property", "").replace("og:", "og_")
        content = og_tag.get("content", "").strip()
        if content:
            meta[prop] = content

    # JSON-LD structured data
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            import json
            ld_data = json.loads(script.string)
            if isinstance(ld_data, dict):
                meta["jsonld"] = ld_data
            elif isinstance(ld_data, list) and ld_data:
                meta["jsonld"] = ld_data[0]
        except (json.JSONDecodeError, TypeError):
            pass

    return meta


def extract_headings(raw_html: str) -> list[str]:
    """Extract all h1-h3 headings from HTML."""
    if not raw_html:
        return []

    soup = BeautifulSoup(raw_html, "lxml")
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text and len(text) > 3:
            headings.append(text)
    return headings


def clean_text(text: str) -> str:
    """Normalize and clean extracted text."""
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove excessively repeated characters
    text = re.sub(r"(.)\1{4,}", r"\1\1", text)

    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, ending at a word boundary."""
    if not text or len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated.strip() + "..."
