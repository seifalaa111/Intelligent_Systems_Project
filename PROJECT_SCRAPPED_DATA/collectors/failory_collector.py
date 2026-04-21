"""
MIDAN Data Pipeline — Failory Collector
Scrapes startup failure/success case studies from Failory.com.
"""

import re
from collectors.base_collector import BaseCollector
from utils.http_client import fetch_url
from utils.text_cleaner import html_to_text, extract_headings
from utils.logger import log_stage, log_success, log_error
from config.settings import FAILORY_BASE_URL, FAILORY_INDEX_URLS, FAILORY_MAX_PAGES, FAILORY_DELAY
from bs4 import BeautifulSoup


class FailoryCollector(BaseCollector):
    """
    Collects startup failure/success case studies from Failory.
    Crawls the cemetery index and individual case study pages.
    """

    def __init__(self):
        super().__init__("failory")

    def collect(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Collect case studies from Failory.
        If targets are provided, tries to find specific startups.
        Otherwise, crawls the cemetery index for all available entries.
        """
        log_stage(self.logger, "FAILORY COLLECTOR", "Starting collection")
        results = []

        # Step 1: Get article URLs from index pages
        article_urls = self._get_article_urls()
        self.logger.info(f"Found {len(article_urls)} article URLs from index")

        if not article_urls:
            # Fallback: try known Failory URL patterns
            article_urls = self._get_fallback_urls()
            self.logger.info(f"Using {len(article_urls)} fallback URLs")

        # Step 2: Fetch and parse each article
        for url in article_urls:
            try:
                entry = self._collect_article(url)
                if entry:
                    results.append(entry)
                    self.save_raw(entry)
                    self.collected_count += 1
                    log_success(self.logger, f"Collected: {entry['startup_name']}")
            except Exception as e:
                log_error(self.logger, f"Failed to collect {url}: {e}")
                self.error_count += 1

        log_stage(self.logger, "FAILORY COLLECTOR",
                  f"Done — {self.collected_count} collected, {self.error_count} errors")
        return results

    def _get_article_urls(self) -> list[str]:
        """Crawl Failory index pages to discover individual article URLs."""
        urls = set()

        for index_url in FAILORY_INDEX_URLS:
            self.logger.info(f"Fetching index: {index_url}")
            response = fetch_url(index_url, delay=FAILORY_DELAY)
            if response is None:
                continue

            soup = BeautifulSoup(response.text, "lxml")

            # Find links to individual startup case study pages
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Failory article URLs typically look like /cemetery/startup-name
                # or /blog/startup-name or /interview/startup-name
                if self._is_article_url(href):
                    full_url = href if href.startswith("http") else f"{FAILORY_BASE_URL}{href}"
                    urls.add(full_url)

            # Check for pagination
            for page_num in range(2, FAILORY_MAX_PAGES + 1):
                paginated_url = f"{index_url}?page={page_num}"
                response = fetch_url(paginated_url, delay=FAILORY_DELAY)
                if response is None:
                    break

                soup = BeautifulSoup(response.text, "lxml")
                new_links = False
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if self._is_article_url(href):
                        full_url = href if href.startswith("http") else f"{FAILORY_BASE_URL}{href}"
                        if full_url not in urls:
                            urls.add(full_url)
                            new_links = True

                if not new_links:
                    break  # No new articles on this page

        return list(urls)

    def _is_article_url(self, href: str) -> bool:
        """Check if a URL looks like a Failory article / case study."""
        # Skip generic navigation links
        skip_patterns = [
            "/blog$", "/cemetery$", "/interviews$",
            "/newsletter", "/about", "/contact", "/privacy",
            "/terms", "/advertise", "#", "javascript:",
            "/tag/", "/category/", "/author/",
        ]
        for pat in skip_patterns:
            if pat in href:
                return False

        # Must be a Failory path with a slug
        article_patterns = [
            r"/cemetery/[\w-]+",
            r"/blog/[\w-]+-startup",
            r"/blog/[\w-]+-failed",
            r"/blog/[\w-]+-success",
            r"/interview/[\w-]+",
        ]
        for pat in article_patterns:
            if re.search(pat, href):
                return True

        return False

    def _get_fallback_urls(self) -> list[str]:
        """Return known Failory article URL patterns as fallback."""
        known_slugs = [
            "quibi", "theranos", "vine", "yik-yak", "jawbone",
            "juicero", "rdio", "homejoy", "beepi", "exec",
            "secret", "zirtual", "sprig", "washio", "luxe",
            "shyp", "meerkat", "quirky", "fab", "solyndra",
        ]
        urls = []
        for slug in known_slugs:
            urls.append(f"{FAILORY_BASE_URL}/cemetery/{slug}")
        return urls

    def _collect_article(self, url: str) -> dict | None:
        """Fetch and parse a single Failory article."""
        response = fetch_url(url, delay=FAILORY_DELAY)
        if response is None:
            return None

        html_content = response.text
        soup = BeautifulSoup(html_content, "lxml")

        # Extract startup name from title or h1
        startup_name = self._extract_startup_name(soup, url)
        if not startup_name:
            return None

        # Extract article body text
        article_text = self._extract_article_text(soup)
        if not article_text or len(article_text) < 100:
            self.logger.warning(f"Insufficient content for {url}")
            return None

        # Extract headings for structure
        headings = extract_headings(html_content)

        return self._make_raw_entry(
            startup_name=startup_name,
            source_url=url,
            raw_content=article_text,
            metadata={
                "headings": headings,
                "article_length": len(article_text),
                "content_type": "case_study",
            },
        )

    def _extract_startup_name(self, soup: BeautifulSoup, url: str) -> str | None:
        """Extract startup name from the page."""
        # Try h1
        h1 = soup.find("h1")
        if h1:
            text = h1.get_text(strip=True)
            # Clean common prefixes like "Why X Failed" or "How X Succeeded"
            name = re.sub(r"^(Why|How|The Rise and Fall of|What Happened to)\s+", "", text, flags=re.I)
            name = re.sub(r"\s+(Failed|Succeeded|Shut Down|Closed|Story).*$", "", name, flags=re.I)
            if name and len(name) < 80:
                return name.strip()

        # Fall back to URL slug
        slug = url.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").title()

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        """Extract the main article body text from a Failory page."""
        # Try to find the main content area
        article = (
            soup.find("article") or
            soup.find("div", class_=re.compile(r"(content|article|post|entry|blog)", re.I)) or
            soup.find("main")
        )

        if article:
            # Remove non-content elements within article
            for tag in article.find_all(["script", "style", "nav", "footer",
                                          "aside", "form", "iframe"]):
                tag.decompose()
            return article.get_text(separator="\n", strip=True)

        # Fallback: get all paragraph text
        paragraphs = soup.find_all("p")
        if paragraphs:
            return "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)

        return ""
