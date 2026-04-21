"""
MIDAN Data Pipeline -- Product Hunt Collector
Scrapes public Product Hunt product pages for positioning and category data.
Uses public pages (no API key required for basic scraping).
"""

import re
from collectors.base_collector import BaseCollector
from utils.http_client import fetch_url
from utils.text_cleaner import html_to_text, extract_meta, extract_headings
from utils.logger import log_stage, log_success, log_error
from config.settings import PRODUCTHUNT_DELAY
from bs4 import BeautifulSoup


class ProductHuntCollector(BaseCollector):
    """
    Collects product positioning data from Product Hunt.
    Scrapes public product pages to extract tagline, description,
    topics, and upvote signals.
    """

    PH_BASE_URL = "https://www.producthunt.com"

    def __init__(self):
        super().__init__("producthunt")

    def collect(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Collect Product Hunt data for target startups.
        Searches for each startup on Product Hunt by name.
        """
        if not targets:
            self.logger.warning("No targets provided for Product Hunt collector")
            return []

        log_stage(self.logger, "PRODUCTHUNT COLLECTOR", f"Processing {len(targets)} targets")
        results = []

        for target in targets:
            name = target.get("name", "")
            if not name:
                continue

            try:
                entry = self._collect_product(name)
                if entry:
                    results.append(entry)
                    self.save_raw(entry)
                    self.collected_count += 1
                    log_success(self.logger, f"Collected PH data: {name}")
                else:
                    self.logger.info(f"No Product Hunt page found for: {name}")
            except Exception as e:
                log_error(self.logger, f"Failed to collect PH data for {name}: {e}")
                self.error_count += 1

        log_stage(self.logger, "PRODUCTHUNT COLLECTOR",
                  f"Done -- {self.collected_count} collected, {self.error_count} errors")
        return results

    def _collect_product(self, name: str) -> dict | None:
        """Try to scrape a Product Hunt product page by name."""
        # Product Hunt URLs are typically /posts/product-name
        slug = name.lower().replace(" ", "-").replace(".", "-")
        url = f"{self.PH_BASE_URL}/posts/{slug}"

        response = fetch_url(url, delay=PRODUCTHUNT_DELAY)
        if response is None:
            # Try search URL pattern
            url = f"{self.PH_BASE_URL}/products/{slug}"
            response = fetch_url(url, delay=PRODUCTHUNT_DELAY)
            if response is None:
                return None

        html_content = response.text
        soup = BeautifulSoup(html_content, "lxml")

        # Extract text content
        text = html_to_text(html_content)
        if not text or len(text) < 50:
            return None

        # Extract meta tags for structured data
        meta = extract_meta(html_content)

        # Extract tagline (usually in og:description or first h2)
        tagline = ""
        if meta.get("og_description"):
            tagline = meta["og_description"]
        elif meta.get("description"):
            tagline = meta["description"]

        # Extract topics/categories from page
        topics = self._extract_topics(soup)

        return self._make_raw_entry(
            startup_name=name,
            source_url=url,
            raw_content=text[:5000],
            metadata={
                "tagline": tagline,
                "topics": topics,
                "meta_tags": meta,
                "data_source": "producthunt_scrape",
            },
        )

    def _extract_topics(self, soup: BeautifulSoup) -> list[str]:
        """Extract topic/category tags from a Product Hunt page."""
        topics = []

        # PH uses various class patterns for topics
        for tag in soup.find_all("a", href=True):
            href = tag.get("href", "")
            text = tag.get_text(strip=True)
            # Topic links usually point to /topics/something
            if "/topics/" in href and text and len(text) < 50:
                topics.append(text)

        # Also check for structured topic data
        for tag in soup.find_all(attrs={"data-test": re.compile(r"topic", re.I)}):
            text = tag.get_text(strip=True)
            if text and text not in topics:
                topics.append(text)

        return list(set(topics))[:10]
