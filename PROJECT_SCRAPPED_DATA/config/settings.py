"""
MIDAN Data Pipeline -- Configuration (Phase 2)
Centralized settings for the data acquisition system.
Includes strengthened classification logic and new L3 signals.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STRUCTURED_DIR = DATA_DIR / "structured"
LOGS_DIR = DATA_DIR / "logs"
TARGETS_DIR = BASE_DIR / "targets"

# Ensure directories exist
for d in [RAW_DIR, STRUCTURED_DIR, LOGS_DIR,
          RAW_DIR / "websites", RAW_DIR / "failory", RAW_DIR / "yc",
          RAW_DIR / "reddit", RAW_DIR / "producthunt"]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# HTTP CLIENT
# ──────────────────────────────────────────────
HTTP_TIMEOUT = 15  # seconds
HTTP_MAX_RETRIES = 3
HTTP_RETRY_DELAY = 2  # seconds between retries
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Rate limiting -- minimum delay between requests per domain
RATE_LIMIT_DELAY = 2.0  # seconds

# ──────────────────────────────────────────────
# SOURCE TYPES (enum-like)
# ──────────────────────────────────────────────
SOURCE_TYPES = {
    "website": "website",
    "insight": "insight",
    "directory": "directory",
    "review": "review",
    "community": "community",
}

# ──────────────────────────────────────────────
# API KEYS (Phase 2)
# ──────────────────────────────────────────────
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MIDAN:v2.0 (by /u/midan_bot)")
PRODUCTHUNT_API_TOKEN = os.getenv("PRODUCTHUNT_API_TOKEN", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ──────────────────────────────────────────────
# COLLECTOR SETTINGS
# ──────────────────────────────────────────────

# Failory
FAILORY_BASE_URL = "https://www.failory.com"
FAILORY_INDEX_URLS = [
    "https://www.failory.com/cemetery",
]
FAILORY_MAX_PAGES = 10
FAILORY_DELAY = 3.0  # seconds between requests

# YC Directory
YC_DELAY = 1.0

# Website collector
WEBSITE_PAGES_TO_CHECK = ["", "/about", "/pricing", "/product"]
WEBSITE_DELAY = 2.0

# Reddit collector
REDDIT_SUBREDDITS = [
    "startups", "SaaS", "Entrepreneur", "smallbusiness",
    "venturecapital", "EntrepreneurRideAlong",
]
REDDIT_MAX_POSTS_PER_SEARCH = 25
REDDIT_DELAY = 1.0

# Product Hunt
PRODUCTHUNT_DELAY = 2.0

# ──────────────────────────────────────────────
# BUSINESS MODEL CLASSIFICATION (Phase 2 -- HARDENED)
# ──────────────────────────────────────────────
# Each model requires:
#   - primary_keywords: strong indicators (high weight)
#   - secondary_keywords: supporting indicators (low weight)
#   - negative_keywords: disqualifiers that reduce score
#   - min_score: minimum total score to classify
#   - priority: lower = higher priority (resolves ties)
# ──────────────────────────────────────────────

BUSINESS_MODEL_RULES = {
    "SaaS": {
        "primary": ["saas", "software as a service", "subscription software",
                     "per seat", "per user pricing", "recurring revenue",
                     "annual subscription", "monthly subscription"],
        "secondary": ["subscription", "monthly plan", "annual plan",
                      "cloud-based", "cloud platform", "hosted solution",
                      "free trial", "upgrade to pro"],
        "negative": ["physical product", "hardware", "device", "mortgage",
                     "real estate", "restaurant"],
        "min_score": 2,
        "priority": 3,
    },
    "Marketplace": {
        "primary": ["marketplace", "two-sided platform", "buyers and sellers",
                     "listing fee", "transaction fee", "platform connecting",
                     "commission-based"],
        "secondary": ["commission", "buyer", "seller", "listing",
                      "peer-to-peer", "p2p", "matching"],
        "negative": ["saas", "subscription software", "developer tools"],
        "min_score": 2,
        "priority": 4,
    },
    "E-Commerce": {
        "primary": ["e-commerce", "ecommerce", "online store", "online retail",
                     "direct to consumer", "d2c", "dtc", "shopify store"],
        "secondary": ["shop", "retail", "checkout", "cart", "buy now",
                      "shipping", "warehouse", "inventory"],
        "negative": ["marketplace", "two-sided", "saas", "api"],
        "min_score": 2,
        "priority": 5,
    },
    "Fintech": {
        "primary": ["fintech", "neobank", "digital bank", "payment processing",
                     "lending platform", "insurance tech", "robo-advisor",
                     "cryptocurrency exchange", "payment gateway"],
        "secondary": ["financial", "banking", "payment", "lending",
                      "investment", "wallet", "trading", "credit"],
        "negative": ["design tool", "project management", "cms", "video",
                     "education", "developer tools"],
        "min_score": 3,
        "priority": 2,
    },
    "Edtech": {
        "primary": ["edtech", "education technology", "e-learning platform",
                     "online courses", "learning management system", "lms"],
        "secondary": ["education", "learning", "course", "tutorial",
                      "training", "student", "teacher", "classroom"],
        "negative": ["design", "code editor", "deployment", "payment",
                     "financial", "marketing"],
        "min_score": 3,
        "priority": 6,
    },
    "Healthtech": {
        "primary": ["healthtech", "health technology", "telemedicine",
                     "digital health", "patient portal", "ehr",
                     "electronic health record"],
        "secondary": ["health", "medical", "patient", "clinical",
                      "diagnosis", "wellness", "fitness", "therapy"],
        "negative": ["design", "marketing", "developer", "payment"],
        "min_score": 3,
        "priority": 6,
    },
    "Adtech": {
        "primary": ["ad tech", "adtech", "advertising technology",
                     "programmatic advertising", "ad network", "ad platform",
                     "demand-side platform", "supply-side platform"],
        "secondary": ["display ads", "impressions", "cpm", "cpc",
                      "ad exchange", "ad server"],
        "negative": ["design", "project management", "code", "developer",
                     "collaboration", "video messaging"],
        "min_score": 3,
        "priority": 7,
    },
    "Devtools": {
        "primary": ["developer tools", "devtools", "developer platform",
                     "code editor", "ide", "ci/cd", "continuous integration",
                     "deployment platform", "infrastructure as code"],
        "secondary": ["sdk", "debugging", "git", "open source",
                      "code review", "pull request", "repository"],
        "negative": ["advertising", "marketing", "health", "finance",
                     "education", "real estate", "food", "delivery",
                     "social", "consumer", "video app", "messaging app",
                     "dating", "gaming", "music", "photo", "travel",
                     "hotel", "grocery", "fashion", "retail"],
        "min_score": 4,
        "priority": 4,
    },
    "Freemium": {
        "primary": ["freemium model", "free forever plan", "free tier with"],
        "secondary": ["free plan", "freemium", "free tier", "upgrade",
                      "premium", "basic plan", "pro plan", "enterprise plan"],
        "negative": [],
        "min_score": 3,
        "priority": 10,  # lowest priority -- freemium is a pricing strategy, not a model
    },
    "Hardware": {
        "primary": ["hardware product", "physical device", "iot device",
                     "wearable device", "sensor hardware", "manufacturing"],
        "secondary": ["hardware", "device", "iot", "wearable", "sensor",
                      "physical product", "robotics"],
        "negative": ["software", "cloud", "saas", "api"],
        "min_score": 3,
        "priority": 5,
    },
}

# ──────────────────────────────────────────────
# INDUSTRY CLASSIFICATION (Phase 2 -- HARDENED)
# ──────────────────────────────────────────────

INDUSTRY_RULES = {
    "Fintech": {
        "primary": ["fintech", "neobank", "digital bank", "payment gateway",
                     "cryptocurrency", "blockchain", "defi"],
        "secondary": ["finance", "banking", "payment", "lending", "insurance",
                      "investment", "wallet", "trading", "crypto", "credit"],
        "negative": ["design tool", "project management", "code editor",
                     "video", "education", "developer tools", "collaboration tool"],
        "min_score": 3,
        "priority": 2,
    },
    "Healthtech": {
        "primary": ["healthtech", "telemedicine", "digital health",
                     "electronic health record", "patient portal"],
        "secondary": ["health", "medical", "patient", "clinical",
                      "wellness", "fitness", "pharmaceutical", "biotech"],
        "negative": ["design", "developer", "marketing", "finance"],
        "min_score": 3,
        "priority": 3,
    },
    "Edtech": {
        "primary": ["edtech", "education technology", "lms",
                     "e-learning", "online courses"],
        "secondary": ["education", "learning", "course", "training",
                      "student", "teacher", "academic", "tutoring"],
        "negative": ["design", "developer", "payment", "financial"],
        "min_score": 3,
        "priority": 3,
    },
    "Enterprise Software": {
        "primary": ["enterprise software", "b2b software", "workflow automation",
                     "crm platform", "erp system", "project management tool"],
        "secondary": ["enterprise", "b2b", "workflow", "automation",
                      "productivity", "collaboration", "crm", "erp",
                      "project management", "team"],
        "negative": ["consumer app", "social media", "gaming", "dating"],
        "min_score": 2,
        "priority": 5,
    },
    "Consumer": {
        "primary": ["consumer app", "b2c", "social network", "dating app",
                     "food delivery", "ride sharing"],
        "secondary": ["consumer", "social", "entertainment", "lifestyle",
                      "food", "travel", "dating", "gaming", "mobile app"],
        "negative": ["enterprise", "b2b", "api", "infrastructure"],
        "min_score": 3,
        "priority": 5,
    },
    "Devtools": {
        "primary": ["developer tools", "devtools", "developer platform",
                     "code hosting", "ci/cd platform"],
        "secondary": ["developer", "devops", "open source",
                      "code review", "repository"],
        "negative": ["advertising", "marketing", "health", "finance",
                     "education", "food", "social", "consumer",
                     "gaming", "music", "photo", "video app",
                     "travel", "hotel", "grocery", "fashion", "retail"],
        "min_score": 4,
        "priority": 3,
    },
    "Cybersecurity": {
        "primary": ["cybersecurity", "information security", "threat detection",
                     "endpoint protection", "siem"],
        "secondary": ["security", "encryption", "privacy", "authentication",
                      "firewall", "threat", "vulnerability"],
        "negative": ["design", "marketing", "social", "food"],
        "min_score": 3,
        "priority": 3,
    },
    "AI / ML": {
        "primary": ["artificial intelligence", "machine learning platform",
                     "ai startup", "deep learning", "llm", "ai company"],
        "secondary": ["machine learning", "nlp", "computer vision", "neural network",
                      "generative ai", "ai-powered", "ai platform"],
        "negative": ["social", "video app", "food", "retail",
                     "toy", "fashion", "music app", "dating",
                     "hotel", "grocery", "consumer"],
        "min_score": 4,
        "priority": 4,
    },
    "Logistics": {
        "primary": ["logistics platform", "supply chain management",
                     "fleet management", "last-mile delivery"],
        "secondary": ["logistics", "supply chain", "shipping", "delivery",
                      "warehouse", "fulfillment", "fleet", "transportation"],
        "negative": ["design", "developer", "marketing"],
        "min_score": 3,
        "priority": 4,
    },
    "Real Estate": {
        "primary": ["proptech", "real estate technology", "property management",
                     "mortgage platform"],
        "secondary": ["real estate", "property", "housing", "rental",
                      "mortgage", "tenant", "landlord"],
        "negative": ["saas", "developer", "design", "marketing"],
        "min_score": 3,
        "priority": 4,
    },
    "Media": {
        "primary": ["media company", "streaming service", "content platform",
                     "video platform", "podcast platform"],
        "secondary": ["media", "content", "publishing", "news", "video",
                      "streaming", "podcast", "creator economy", "social media"],
        "negative": ["developer", "api", "payment", "financial"],
        "min_score": 3,
        "priority": 6,
    },
    "HR / Recruiting": {
        "primary": ["hr tech", "recruiting platform", "payroll software",
                     "talent management", "hiring platform"],
        "secondary": ["hr", "recruiting", "hiring", "talent", "workforce",
                      "payroll", "employee", "human resources",
                      "applicant tracking"],
        "negative": ["design", "developer", "payment", "marketing"],
        "min_score": 3,
        "priority": 4,
    },
    "Marketing": {
        "primary": ["martech", "marketing technology", "marketing automation",
                     "seo platform", "email marketing"],
        "secondary": ["marketing", "seo", "analytics", "email marketing",
                      "social media marketing", "advertising", "growth"],
        "negative": ["developer", "api", "health", "finance", "education"],
        "min_score": 3,
        "priority": 5,
    },
    "Design": {
        "primary": ["design tool", "design platform", "ui/ux tool",
                     "prototyping tool", "graphic design"],
        "secondary": ["design", "prototype", "wireframe", "ui", "ux",
                      "creative", "illustration", "visual"],
        "negative": ["payment", "financial", "health", "education"],
        "min_score": 3,
        "priority": 4,
    },
    "Communication": {
        "primary": ["communication platform", "messaging platform",
                     "video conferencing", "team chat", "video messaging"],
        "secondary": ["messaging", "chat", "video call", "communication",
                      "conferencing", "calls", "meetings"],
        "negative": ["payment", "financial", "health", "education", "design"],
        "min_score": 3,
        "priority": 5,
    },
}

# ──────────────────────────────────────────────
# TARGET USER CLASSIFICATION (B2B / B2C)
# ──────────────────────────────────────────────
TARGET_SEGMENT_RULES = {
    "B2B": ["enterprise", "business", "team", "company", "companies",
            "organization", "corporate", "professional", "agency",
            "b2b", "workflow", "operations", "employees"],
    "B2C": ["consumer", "individual", "personal", "user", "people",
            "everyone", "anyone", "customer", "shopper", "b2c",
            "family", "home"],
}

# ──────────────────────────────────────────────
# FAILURE KEYWORDS (Enhanced for Phase 2)
# ──────────────────────────────────────────────

FAILURE_KEYWORDS = {
    "no_market_need": ["no market need", "nobody wanted", "no demand",
                       "no customers", "couldn't find users", "no traction",
                       "market didn't want", "solution looking for a problem"],
    "ran_out_of_cash": ["ran out of cash", "ran out of money", "burn rate",
                        "couldn't raise", "no investment", "funding dried up",
                        "couldn't secure funding", "cash flow"],
    "bad_team": ["co-founder conflict", "team fell apart", "wrong hires",
                 "founder disagreement", "team issues", "leadership problems"],
    "competition": ["outcompeted", "lost to competitor", "market leader dominated",
                    "crowded market", "commoditized", "couldn't differentiate"],
    "pricing_monetization": ["couldn't monetize", "pricing wrong",
                             "users wouldn't pay", "willingness to pay",
                             "unit economics", "too expensive", "free forever trap"],
    "product_issues": ["product wasn't good enough", "bug", "technical debt",
                       "scalability problems", "poor user experience",
                       "couldn't ship fast enough"],
    "timing": ["too early", "too late", "timing was off",
               "market wasn't ready", "ahead of its time"],
    "pivot_failure": ["failed pivot", "pivoted too late", "lost focus",
                      "feature creep", "identity crisis"],
    "regulation": ["regulatory issues", "legal problems", "compliance failure",
                   "government shutdown", "copyright issues"],
    "growth_retention": ["couldn't scale", "high churn", "low retention",
                         "acquisition cost too high", "couldn't grow",
                         "user growth stalled"],
    "market_timing": ["pandemic", "economic downturn", "recession",
                      "market crash", "bubble burst"],
}

# ──────────────────────────────────────────────
# FUNDING STAGE KEYWORDS (for Failory extraction)
# ──────────────────────────────────────────────
FUNDING_STAGE_KEYWORDS = {
    "pre_seed": ["pre-seed", "bootstrapped", "self-funded", "angel"],
    "seed": ["seed round", "seed funding", "seed stage"],
    "series_a": ["series a", "series-a"],
    "series_b": ["series b", "series-b"],
    "series_c_plus": ["series c", "series d", "series e", "late stage"],
    "ipo": ["ipo", "public offering", "went public"],
    "acquired": ["acquired", "acquisition", "bought by"],
}

# ──────────────────────────────────────────────
# PAIN POINT KEYWORDS
# ──────────────────────────────────────────────
PAIN_KEYWORDS = [
    "frustrating", "difficult", "slow", "expensive", "confusing",
    "complex", "painful", "broken", "unreliable", "annoying",
    "hard to use", "time-consuming", "manual", "tedious", "inefficient",
    "lack of", "missing", "no way to", "can't", "doesn't work",
    "poor", "limited", "outdated", "clunky", "overwhelming",
]

# ──────────────────────────────────────────────
# SUCCESS DRIVER KEYWORDS
# ──────────────────────────────────────────────
SUCCESS_KEYWORDS = [
    "product-market fit", "viral", "word of mouth", "organic growth",
    "strong team", "unique value", "first mover", "network effects",
    "retention", "engagement", "repeat usage", "revenue growth",
    "profitability", "scalable", "defensible", "moat",
    "community", "brand loyalty", "partnerships", "distribution",
]

# ──────────────────────────────────────────────
# L3 SIGNAL KEYWORDS (NEW Phase 2)
# ──────────────────────────────────────────────

RETENTION_KEYWORDS = {
    "high": ["daily active", "sticky", "habit", "repeat usage", "engagement",
             "retention rate", "weekly active", "power users", "active daily",
             "can't live without", "essential tool"],
    "medium": ["regular use", "monthly active", "returning users",
               "occasional use", "useful tool"],
    "low": ["churn", "abandoned", "stopped using", "one-time",
            "low engagement", "rarely used", "uninstalled"],
}

ONBOARDING_KEYWORDS = {
    "low_friction": ["instant setup", "no credit card", "get started in minutes",
                     "sign up free", "one-click", "no setup required",
                     "try it now", "start free"],
    "medium_friction": ["onboarding", "getting started guide", "setup wizard",
                        "tutorial", "14-day trial", "30-day trial",
                        "import your data"],
    "high_friction": ["implementation", "contact sales", "custom setup",
                      "professional services", "training required",
                      "deployment", "enterprise onboarding", "sso setup",
                      "dedicated account manager"],
}

MONETIZATION_KEYWORDS = {
    "strong": ["profitable", "revenue growing", "strong unit economics",
               "positive margins", "high arpu", "upsell", "expansion revenue",
               "net revenue retention", "paying customers"],
    "moderate": ["monetizing", "subscription revenue", "paid users",
                 "free trial conversion", "pricing tiers"],
    "weak": ["pre-revenue", "not monetized", "free only", "ad-supported only",
             "struggling to monetize", "low conversion"],
}

COMPETITION_KEYWORDS = {
    "high": ["crowded market", "many competitors", "saturated", "red ocean",
             "commoditized", "intense competition", "highly competitive",
             "dominated by", "market leader"],
    "moderate": ["some competitors", "competitive", "alternatives exist",
                 "growing market", "emerging space"],
    "low": ["no competition", "blue ocean", "first mover", "untapped market",
            "no one else", "unique market", "new category", "category creator"],
}

# ──────────────────────────────────────────────
# NORMALIZED ENUM VALUES
# ──────────────────────────────────────────────

# Canonical business model labels (display format)
BUSINESS_MODEL_CANONICAL = {
    "saas": "SaaS",
    "marketplace": "Marketplace",
    "e-commerce": "E-Commerce",
    "fintech": "Fintech",
    "edtech": "Edtech",
    "healthtech": "Healthtech",
    "adtech": "Adtech",
    "devtools": "Devtools",
    "freemium": "Freemium",
    "hardware": "Hardware",
}

# Canonical industry labels (display format)
INDUSTRY_CANONICAL = {
    "fintech": "Fintech",
    "healthtech": "Healthtech",
    "edtech": "Edtech",
    "enterprise_software": "Enterprise Software",
    "enterprise software": "Enterprise Software",
    "consumer": "Consumer",
    "devtools": "Devtools",
    "cybersecurity": "Cybersecurity",
    "ai_ml": "AI / ML",
    "ai / ml": "AI / ML",
    "logistics": "Logistics",
    "real_estate": "Real Estate",
    "real estate": "Real Estate",
    "media": "Media",
    "hr_recruiting": "HR / Recruiting",
    "hr / recruiting": "HR / Recruiting",
    "marketing": "Marketing",
    "design": "Design",
    "communication": "Communication",
    "sustainability": "Sustainability",
    "legal": "Legal",
}

# Switching cost canonical values
SWITCHING_COST_CANONICAL = {"high", "medium", "low"}

# ──────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────
OUTPUT_FILE = STRUCTURED_DIR / "startup_intelligence.json"
LOG_FILE = LOGS_DIR / "pipeline.log"
