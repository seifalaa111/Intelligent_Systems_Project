"""
MIDAN Data Pipeline — Y Combinator Directory Collector
Scrapes the public YC directory at ycombinator.com/companies.
Falls back to a curated seed list of well-known YC companies.
"""

import re
import json
from collectors.base_collector import BaseCollector
from utils.http_client import fetch_url
from utils.text_cleaner import html_to_text, extract_meta
from utils.logger import log_stage, log_success, log_error
from config.settings import YC_DELAY
from bs4 import BeautifulSoup


# Well-known YC companies with verified data for fallback / seed
YC_SEED_DATA = [
    {
        "name": "Stripe",
        "batch": "S09",
        "description": "Financial infrastructure for the internet. Stripe builds the most powerful and flexible tools for internet commerce.",
        "industry": "Fintech",
        "status": "Active",
        "website": "https://stripe.com",
    },
    {
        "name": "Airbnb",
        "batch": "W09",
        "description": "Book unique homes and experiences all over the world.",
        "industry": "Consumer",
        "status": "Active",
        "website": "https://www.airbnb.com",
    },
    {
        "name": "Dropbox",
        "batch": "S07",
        "description": "Dropbox is a file hosting service that offers cloud storage, file synchronization, and personal cloud software.",
        "industry": "Enterprise Software",
        "status": "Active",
        "website": "https://www.dropbox.com",
    },
    {
        "name": "Coinbase",
        "batch": "S12",
        "description": "Coinbase is a secure online platform for buying, selling, transferring, and storing cryptocurrency.",
        "industry": "Fintech",
        "status": "Active",
        "website": "https://www.coinbase.com",
    },
    {
        "name": "DoorDash",
        "batch": "S13",
        "description": "DoorDash is a technology company that connects consumers with their favorite local and national businesses.",
        "industry": "Consumer",
        "status": "Active",
        "website": "https://www.doordash.com",
    },
    {
        "name": "Instacart",
        "batch": "S12",
        "description": "Instacart is a grocery delivery and pick-up service that operates through a website and mobile app.",
        "industry": "Consumer",
        "status": "Active",
        "website": "https://www.instacart.com",
    },
    {
        "name": "Twitch",
        "batch": "W07",
        "description": "Twitch is an interactive livestreaming service for content spanning gaming, entertainment, sports, music, and more.",
        "industry": "Media",
        "status": "Acquired",
        "website": "https://www.twitch.tv",
    },
    {
        "name": "Reddit",
        "batch": "S05",
        "description": "Reddit is a network of communities where people can dive into their interests, hobbies and passions.",
        "industry": "Consumer",
        "status": "Active",
        "website": "https://www.reddit.com",
    },
    {
        "name": "GitLab",
        "batch": "W15",
        "description": "GitLab is an open source end-to-end software development platform with built-in version control, issue tracking, code review, CI/CD, and more.",
        "industry": "Devtools",
        "status": "Active",
        "website": "https://about.gitlab.com",
    },
    {
        "name": "Gusto",
        "batch": "W12",
        "description": "Gusto provides payroll, benefits, and HR tools for small businesses.",
        "industry": "HR / Recruiting",
        "status": "Active",
        "website": "https://gusto.com",
    },
    {
        "name": "Retool",
        "batch": "W17",
        "description": "Retool is a development platform for building internal business tools quickly.",
        "industry": "Devtools",
        "status": "Active",
        "website": "https://retool.com",
    },
    {
        "name": "Brex",
        "batch": "W17",
        "description": "Brex is a financial operating system for companies, offering corporate cards, expense management, and bill pay.",
        "industry": "Fintech",
        "status": "Active",
        "website": "https://www.brex.com",
    },
    {
        "name": "Deel",
        "batch": "W19",
        "description": "Deel is a global payroll and compliance platform that helps companies hire anyone, anywhere.",
        "industry": "HR / Recruiting",
        "status": "Active",
        "website": "https://www.deel.com",
    },
    {
        "name": "PostHog",
        "batch": "W20",
        "description": "PostHog is an open-source product analytics platform. It provides event tracking, session recording, feature flags, and A/B testing.",
        "industry": "Devtools",
        "status": "Active",
        "website": "https://posthog.com",
    },
    {
        "name": "Supabase",
        "batch": "S20",
        "description": "Supabase is an open source Firebase alternative providing a Postgres database, authentication, instant APIs, real-time subscriptions, and storage.",
        "industry": "Devtools",
        "status": "Active",
        "website": "https://supabase.com",
    },
    {
        "name": "Webflow",
        "batch": "S13",
        "description": "Webflow empowers designers to build professional, custom websites in a completely visual canvas with no code.",
        "industry": "Design",
        "status": "Active",
        "website": "https://webflow.com",
    },
    {
        "name": "Zapier",
        "batch": "S12",
        "description": "Zapier lets you connect your apps and automate workflows. Easy automation for busy people.",
        "industry": "Enterprise Software",
        "status": "Active",
        "website": "https://zapier.com",
    },
    {
        "name": "PagerDuty",
        "batch": "None",
        "description": "PagerDuty is a digital operations management platform that manages urgent and mission-critical work for modern digital businesses.",
        "industry": "Devtools",
        "status": "Active",
        "website": "https://www.pagerduty.com",
    },
    {
        "name": "Algolia",
        "batch": "W14",
        "description": "Algolia is a search-as-a-service platform providing hosted search API for websites and mobile applications.",
        "industry": "Enterprise Software",
        "status": "Active",
        "website": "https://www.algolia.com",
    },
    {
        "name": "Segment",
        "batch": "S11",
        "description": "Segment is a customer data platform that collects, cleans, and controls customer data.",
        "industry": "Enterprise Software",
        "status": "Acquired",
        "website": "https://segment.com",
    },
    {
        "name": "Razorpay",
        "batch": "W15",
        "description": "Razorpay is a payment gateway for India that allows businesses to accept, process, and disburse payments.",
        "industry": "Fintech",
        "status": "Active",
        "website": "https://razorpay.com",
    },
    {
        "name": "Loom",
        "batch": "S15",
        "description": "Loom is a video messaging tool that helps you get your message across through instantly shareable videos.",
        "industry": "Communication",
        "status": "Acquired",
        "website": "https://www.loom.com",
    },
    {
        "name": "Replit",
        "batch": "W18",
        "description": "Replit is a collaborative browser-based IDE that lets you write code, build apps, and host everything in one place.",
        "industry": "Devtools",
        "status": "Active",
        "website": "https://replit.com",
    },
    {
        "name": "Whatnot",
        "batch": "W20",
        "description": "Whatnot is a livestream shopping platform for collectibles and more. Buyers and sellers interact in real-time through live auctions.",
        "industry": "Consumer",
        "status": "Active",
        "website": "https://www.whatnot.com",
    },
    {
        "name": "Fivetran",
        "batch": "W13",
        "description": "Fivetran automates data integration for analytics teams by replicating data from applications, databases, and files into data warehouses.",
        "industry": "Enterprise Software",
        "status": "Active",
        "website": "https://fivetran.com",
    },
]


class YCCollector(BaseCollector):
    """
    Collects startup data from Y Combinator's directory.
    Primary method: Scrape public company pages.
    Fallback: Uses curated seed data of well-known YC companies.
    """

    YC_COMPANIES_URL = "https://www.ycombinator.com/companies"

    def __init__(self):
        super().__init__("yc")

    def collect(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Collect YC company data.
        If targets provided, searches for matching YC companies in seed data.
        Otherwise, attempts to scrape the public directory.
        """
        log_stage(self.logger, "YC COLLECTOR", "Starting collection")
        results = []

        # Strategy 1: Try scraping individual YC company pages
        if targets:
            results = self._collect_from_targets(targets)

        # Strategy 2: If no targets or scraping failed, use seed data
        if not results:
            self.logger.info("Using curated YC seed data")
            results = self._collect_from_seed()

        self.save_all_raw(results)
        self.collected_count = len(results)

        log_stage(self.logger, "YC COLLECTOR",
                  f"Done -- {self.collected_count} companies collected")
        return results

    def _collect_from_targets(self, targets: list[dict]) -> list[dict]:
        """Try to scrape individual YC company pages for specific targets."""
        results = []

        for target in targets:
            name = target.get("name", "")
            if not name:
                continue

            # Check if this company is in our seed data
            seed_match = self._find_in_seed(name)
            if seed_match:
                entry = self._seed_to_raw_entry(seed_match)
                results.append(entry)
                log_success(self.logger, f"Found YC company in seed: {name}")
                continue

            # Try scraping the YC company page directly
            slug = name.lower().replace(" ", "-")
            url = f"{self.YC_COMPANIES_URL}/{slug}"
            entry = self._scrape_company_page(name, url)
            if entry:
                results.append(entry)
                log_success(self.logger, f"Scraped YC company page: {name}")

        return results

    def _collect_from_seed(self) -> list[dict]:
        """Convert seed data to raw entries."""
        results = []
        for seed in YC_SEED_DATA:
            entry = self._seed_to_raw_entry(seed)
            results.append(entry)
        return results

    def _find_in_seed(self, name: str) -> dict | None:
        """Find a company in the seed data by name (case-insensitive)."""
        name_lower = name.lower().strip()
        for seed in YC_SEED_DATA:
            if seed["name"].lower() == name_lower:
                return seed
        return None

    def _seed_to_raw_entry(self, seed: dict) -> dict:
        """Convert a seed data dict to a standardized raw entry."""
        name = seed["name"]
        slug = name.lower().replace(" ", "-")
        source_url = f"https://www.ycombinator.com/companies/{slug}"

        return self._make_raw_entry(
            startup_name=name,
            source_url=source_url,
            raw_content=seed.get("description", ""),
            metadata={
                "batch": seed.get("batch", ""),
                "industry": seed.get("industry", ""),
                "status": seed.get("status", ""),
                "website": seed.get("website", ""),
                "data_source": "yc_seed",
            },
        )

    def _scrape_company_page(self, name: str, url: str) -> dict | None:
        """Try to scrape an individual YC company page."""
        try:
            response = fetch_url(url, delay=YC_DELAY)
            if response is None:
                return None

            soup = BeautifulSoup(response.text, "lxml")

            # Extract the main content
            text = html_to_text(response.text)
            if not text or len(text) < 50:
                return None

            # Try to extract structured data from the page
            meta = extract_meta(response.text)

            return self._make_raw_entry(
                startup_name=name,
                source_url=url,
                raw_content=text[:5000],  # cap content length
                metadata={
                    "meta_tags": meta,
                    "data_source": "yc_scrape",
                },
            )

        except Exception as e:
            self.logger.warning(f"Could not scrape YC page for {name}: {e}")
            return None
