from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import logging
from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from .config import MAX_CONTEXT_SCORE_FOR_SHORT, MIN_CONTEXT_SCORE_FOR_LONG
from .settings import load_runtime_settings


LOGGER = logging.getLogger(__name__)

COINTELEGRAPH_RSS = "https://cointelegraph.com/rss"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

POSITIVE_WORDS = {
    "approval", "approved", "launch", "partnership", "partnerships", "bullish", "surge", "gain",
    "gains", "rise", "rally", "buy", "bought", "adoption", "upgrade", "breakout", "strong",
}
NEGATIVE_WORDS = {
    "war", "attack", "sanction", "sanctions", "ban", "hack", "lawsuit", "bearish", "drop", "dump",
    "sell", "sold", "fraud", "liquidation", "inflation", "recession", "crash", "risk", "oil",
}


@dataclass
class MarketContext:
    enabled: bool
    score: float
    article_count: int
    headline: str
    reason: str


def _score_text(text: str) -> float:
    lowered = text.lower()
    positive = sum(1 for word in POSITIVE_WORDS if word in lowered)
    negative = sum(1 for word in NEGATIVE_WORDS if word in lowered)
    return float(positive - negative)


def _safe_text(element, tag: str) -> str:
    found = element.find(tag)
    return (found.text or "").strip() if found is not None and found.text else ""


@lru_cache(maxsize=256)
def _fetch_rss(url: str, rounded_key: str) -> list[dict]:
    with urlopen(url, timeout=15) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    items = []
    for item in root.findall(".//item")[:20]:
        items.append(
            {
                "title": _safe_text(item, "title"),
                "description": _safe_text(item, "description"),
                "pubDate": _safe_text(item, "pubDate"),
            }
        )
    return items


def _cache_key() -> str:
    now = datetime.now(timezone.utc)
    rounded_minute = (now.minute // 5) * 5
    rounded = now.replace(minute=rounded_minute, second=0, microsecond=0)
    return rounded.isoformat()


def get_market_context(symbol: str) -> MarketContext:
    settings = load_runtime_settings()
    news_settings = settings.get("news", {})
    if not news_settings.get("enabled"):
        return MarketContext(False, 0.0, 0, "", "news_disabled")

    provider = news_settings.get("provider", "rss")
    if provider != "rss":
        return MarketContext(False, 0.0, 0, "", "unsupported_news_provider")

    base_asset = symbol.replace("USDT", "")
    macro_query = news_settings.get("macro_query", "crypto OR bitcoin OR fed OR inflation OR war OR sanctions OR oil OR gas")
    symbol_query = quote_plus(f'{base_asset} crypto OR {base_asset} token OR {base_asset} binance')
    macro_query_url = quote_plus(macro_query)
    key = _cache_key()

    try:
        symbol_articles = _fetch_rss(GOOGLE_NEWS_RSS.format(query=symbol_query), key)
        macro_articles = _fetch_rss(GOOGLE_NEWS_RSS.format(query=macro_query_url), key)
        crypto_articles = _fetch_rss(COINTELEGRAPH_RSS, key)
    except Exception as exc:
        LOGGER.warning("Market context fetch failed for %s: %s", symbol, exc)
        return MarketContext(False, 0.0, 0, "", "rss_fetch_failed")

    combined = symbol_articles[:8] + macro_articles[:8] + crypto_articles[:8]
    if not combined:
        return MarketContext(True, 0.0, 0, "", "no_articles")

    total_score = 0.0
    for article in combined:
        total_score += _score_text(f"{article.get('title', '')} {article.get('description', '')}")

    normalized_score = max(-1.0, min(1.0, total_score / max(len(combined), 1)))
    headline = combined[0].get("title") or ""
    return MarketContext(True, normalized_score, len(combined), headline, "ok")


def context_blocks_trade(side: str, context: MarketContext) -> bool:
    if not context.enabled:
        return False
    if side == "BUY" and context.score < MIN_CONTEXT_SCORE_FOR_LONG:
        return True
    if side == "SELL" and context.score > MAX_CONTEXT_SCORE_FOR_SHORT:
        return True
    return False
