"""
MIDAN Data Pipeline — Website Collector
Scrapes startup websites (homepage, about, pricing) for business model extraction.
"""

from collectors.base_collector import BaseCollector
from utils.http_client import fetch_url
from utils.text_cleaner import html_to_text, extract_meta, extract_headings
from utils.logger import log_stage, log_success, log_error
from config.settings import WEBSITE_PAGES_TO_CHECK, WEBSITE_DELAY


class WebsiteCollector(BaseCollector):
    """
    Collects content from startup websites.
    Fetches homepage + key pages (about, pricing, product).
    Extracts visible text, meta tags, headings, and structured data.
    """

    def __init__(self):
        super().__init__("websites")

    def collect(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Collect website data for target startups.

        targets: List of dicts with {"name": "...", "url": "https://..."}
        """
        if not targets:
            self.logger.warning("No targets provided for website collector")
            return []

        log_stage(self.logger, "WEBSITE COLLECTOR", f"Processing {len(targets)} targets")
        results = []

        for target in targets:
            name = target.get("name", "unknown")
            base_url = target.get("url", "").rstrip("/")

            if not base_url:
                self.logger.warning(f"No URL for {name}, skipping")
                self.error_count += 1
                continue

            try:
                entry = self._collect_single(name, base_url)
                if entry:
                    results.append(entry)
                    self.save_raw(entry)
                    self.collected_count += 1
                    log_success(self.logger, f"Collected website data: {name}")
                else:
                    self.error_count += 1
            except Exception as e:
                log_error(self.logger, f"Failed to collect {name}: {e}")
                self.error_count += 1

        log_stage(self.logger, "WEBSITE COLLECTOR",
                  f"Done — {self.collected_count} collected, {self.error_count} errors")
        return results

    def _collect_single(self, name: str, base_url: str) -> dict | None:
        """Collect data from a single startup website."""
        all_text_parts = []
        all_headings = []
        all_meta = {}
        source_urls = []

        for page_path in WEBSITE_PAGES_TO_CHECK:
            url = f"{base_url}{page_path}"
            response = fetch_url(url, delay=WEBSITE_DELAY)

            if response is None:
                continue

            html_content = response.text
            source_urls.append(url)

            # Extract text
            text = html_to_text(html_content)
            if text:
                all_text_parts.append(f"[PAGE: {page_path or '/'}]\n{text}")

            # Extract meta tags (prioritize homepage meta)
            meta = extract_meta(html_content)
            if not all_meta:
                all_meta = meta
            else:
                # Merge without overwriting existing keys
                for k, v in meta.items():
                    if k not in all_meta:
                        all_meta[k] = v

            # Extract headings
            headings = extract_headings(html_content)
            all_headings.extend(headings)

        if not all_text_parts:
            self.logger.warning(f"No content retrieved for {name} ({base_url})")
            return None

        combined_text = "\n\n".join(all_text_parts)

        return self._make_raw_entry(
            startup_name=name,
            source_url=base_url,
            raw_content=combined_text,
            metadata={
                "meta_tags": all_meta,
                "headings": all_headings[:30],  # cap headings
                "pages_fetched": source_urls,
                "total_text_length": len(combined_text),
            },
        )
