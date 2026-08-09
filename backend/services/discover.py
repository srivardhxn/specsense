"""
STAGE 1: Discover

Takes the minimal product input (part number, brand, description) and
finds candidate source material -- checking provided/ingested datasets
FIRST (via RAG), then falling back to live web search for anything not
covered locally. This mirrors how a real product team would work: use
what you already have before going out to the open web.

Live search uses SerpAPI (https://serpapi.com) -- free tier, no Google
Cloud project needed, fastest to set up for a hackathon.
"""
import os
import httpx
from models import ProductInput, SourceHit
from services.rag import rag_store

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"


async def discover_sources(product: ProductInput, max_results: int = 3) -> list[SourceHit]:
    """
    Retrieval order:
      1. Query the local RAG store (provided datasets/reference artifacts).
      2. If it returns confident hits, use those -- fast, free, no web dependency.
      3. Otherwise (or in addition, if RAG returns fewer than max_results),
         fall back to live web search to fill the gap.
    """
    query_text = f"{product.brand} {product.part_number} {product.short_description}"

    rag_hits = rag_store.query(query_text, top_k=max_results, part_number=product.part_number)

    remaining = max_results - len(rag_hits)
    web_hits: list[SourceHit] = []
    if remaining > 0:
        web_hits = await _web_search(product, max_results=remaining)

    return rag_hits + web_hits


async def _web_search(product: ProductInput, max_results: int) -> list[SourceHit]:
    if not SERPAPI_KEY:
        # No key configured -- degrade gracefully instead of crashing the
        # whole pipeline. The structuring stage handles zero sources fine
        # (everything just comes back low-confidence / needs_review).
        print("[discover] SERPAPI_KEY not set -- skipping live web search.")
        return []

    query = f"{product.brand} {product.part_number} datasheet specifications"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                SERPAPI_URL,
                params={"q": query, "api_key": SERPAPI_KEY, "num": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"[discover] web search failed: {e}")
        return []

    results = []
    for item in data.get("organic_results", [])[:max_results]:
        results.append(
            SourceHit(
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                origin="web",
            )
        )
    return results
