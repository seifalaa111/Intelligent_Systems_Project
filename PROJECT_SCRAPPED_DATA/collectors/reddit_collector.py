"""
MIDAN Data Pipeline -- Reddit Collector
Uses PRAW to collect startup-related discussions from targeted subreddits.
Requires Reddit API credentials in .env file.
"""

import time
from collectors.base_collector import BaseCollector
from utils.logger import log_stage, log_success, log_error
from config.settings import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS, REDDIT_MAX_POSTS_PER_SEARCH, REDDIT_DELAY,
)


class RedditCollector(BaseCollector):
    """
    Collects startup-related discussions from Reddit.
    Searches targeted subreddits for startup names and extracts
    posts + top comments for sentiment and pain point extraction.
    """

    def __init__(self):
        super().__init__("reddit")
        self.reddit = None
        self._init_reddit()

    def _init_reddit(self):
        """Initialize PRAW Reddit instance."""
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            self.logger.warning(
                "Reddit API credentials not configured. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env"
            )
            return

        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            self.logger.info("Reddit API initialized successfully")
        except ImportError:
            self.logger.warning("PRAW not installed. Run: pip install praw")
        except Exception as e:
            log_error(self.logger, f"Failed to initialize Reddit API: {e}")

    def collect(self, targets: list[dict] | None = None) -> list[dict]:
        """
        Collect Reddit discussions about target startups.
        Searches across configured subreddits for each target.
        """
        if not self.reddit:
            self.logger.warning("Reddit collector not available (no API credentials)")
            return []

        if not targets:
            self.logger.warning("No targets provided for Reddit collector")
            return []

        log_stage(self.logger, "REDDIT COLLECTOR", f"Searching for {len(targets)} startups")
        results = []

        for target in targets:
            name = target.get("name", "")
            if not name:
                continue

            try:
                entries = self._search_startup(name)
                results.extend(entries)
                if entries:
                    self.collected_count += len(entries)
                    log_success(self.logger, f"Found {len(entries)} Reddit discussions for: {name}")
                else:
                    self.logger.info(f"No Reddit discussions found for: {name}")
            except Exception as e:
                log_error(self.logger, f"Reddit search failed for '{name}': {e}")
                self.error_count += 1

        self.save_all_raw(results)
        log_stage(self.logger, "REDDIT COLLECTOR",
                  f"Done -- {self.collected_count} posts collected")
        return results

    def _search_startup(self, startup_name: str) -> list[dict]:
        """Search multiple subreddits for a startup name."""
        all_entries = []

        for subreddit_name in REDDIT_SUBREDDITS:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                # Search for the startup name
                search_results = subreddit.search(
                    startup_name,
                    limit=REDDIT_MAX_POSTS_PER_SEARCH,
                    sort="relevance",
                    time_filter="all",
                )

                for submission in search_results:
                    entry = self._parse_submission(submission, startup_name, subreddit_name)
                    if entry:
                        all_entries.append(entry)

                time.sleep(REDDIT_DELAY)

            except Exception as e:
                self.logger.warning(f"Failed to search r/{subreddit_name} for '{startup_name}': {e}")

        return all_entries

    def _parse_submission(self, submission, startup_name: str,
                          subreddit_name: str) -> dict | None:
        """Parse a Reddit submission into a raw entry."""
        # Build content from post + top comments
        content_parts = [
            f"[TITLE] {submission.title}",
            f"[BODY] {submission.selftext}" if submission.selftext else "",
        ]

        # Get top-level comments (limit to top 10)
        try:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments[:10]:
                if hasattr(comment, "body") and len(comment.body) > 20:
                    score_tag = f"[score:{comment.score}]" if hasattr(comment, "score") else ""
                    content_parts.append(f"[COMMENT {score_tag}] {comment.body[:500]}")
        except Exception:
            pass

        raw_content = "\n\n".join(part for part in content_parts if part)

        if len(raw_content) < 50:
            return None

        source_url = f"https://www.reddit.com{submission.permalink}"

        return self._make_raw_entry(
            startup_name=startup_name,
            source_url=source_url,
            raw_content=raw_content,
            metadata={
                "subreddit": subreddit_name,
                "post_title": submission.title,
                "post_score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
                "post_id": submission.id,
            },
        )
