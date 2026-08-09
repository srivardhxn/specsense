"""
STAGE 3 + 4: Structure & Score Confidence (multi-source cross-validation,
single combined LLM call)

Sends ALL discovered sources to the LLM in ONE call (not one call per
source), and asks it to report, per field, what each individual source
says. This preserves genuine multi-source cross-validation -- our code
still checks whether sources agree, not the model -- while using ~3x
fewer API calls than calling the LLM once per source. On a rate-limited
free tier, this is the difference between a demo that survives and one
that doesn't.

Uses Google's Gemini API (free tier, no credit card). Model is
Flash-Lite, which sits in a separate/larger daily-quota bucket than
full Flash on the free tier.
"""
import os
import json
import time
import asyncio
from collections import defaultdict
import google.generativeai as genai
from groq import Groq
from models import ProductInput, SourceHit, StructuredProduct, FieldValue

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
GROQ_MODEL = "llama-3.3-70b-versatile"

SCHEMA_FIELDS = ["category", "description_long", "dimensions", "material", "weight", "certifications"]
FREEFORM_FIELDS = {"description_long"}  # exact-match agreement doesn't make sense for free text

SYSTEM_PROMPT = """You are a product data extraction engine for an industrial commerce catalog.
You will be given a product's known info (part number, brand, short description) and text from
MULTIPLE numbered sources (webpages, datasheet PDFs, or provided datasets). Each source may or
may not actually be about this exact product.

For EACH of the following fields, examine EVERY source independently and report what EACH source
says (not a merged answer -- report each source's own value separately):
- category (product category, e.g. "Hydraulic Fitting")
- description_long (a rich 1-2 sentence product description)
- dimensions (physical dimensions if stated)
- material (material composition if stated)
- weight (weight if stated)
- certifications (any listed certifications/standards, e.g. "ISO 9001", "UL Listed")

Only report a value for a source if that field is DIRECTLY stated in THAT source's text. Do not
guess, do not use outside knowledge, and do not copy a value from one source into another source's
entry. If a source doesn't mention a field, omit that source from that field's list entirely.

Respond ONLY with valid JSON in this exact shape, no other text, no markdown fences:
{
  "category": [ {"source_index": 0, "value": "..."}, {"source_index": 2, "value": "..."} ],
  "description_long": [ {"source_index": 1, "value": "..."} ],
  "dimensions": [],
  "material": [ {"source_index": 0, "value": "..."} ],
  "weight": [],
  "certifications": [ {"source_index": 0, "value": "..."} ]
}
Every field key must be present even if its array is empty.
"""


def _empty_extraction():
    return {f: [] for f in SCHEMA_FIELDS}


def _call_gemini(user_prompt: str) -> dict:
    response = gemini_model.generate_content(
        user_prompt,
        generation_config={
            "temperature": 0,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
        },
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return _parse_json_loosely(raw)


def _call_groq(user_prompt: str) -> dict:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY not set -- cannot use fallback provider.")
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return _parse_json_loosely(raw)


def _parse_json_loosely(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise


def _call_llm_with_fallback(user_prompt: str, max_attempts: int = 2) -> dict:
    """
    Tries Gemini first (with short backoff retries for transient rate
    limits). If Gemini fails outright -- especially a DAILY quota
    exhaustion, which retrying can never fix within the same day --
    automatically falls back to Groq's free tier instead of failing
    the whole product lookup. This is what keeps a demo alive even if
    one provider's free quota runs dry mid-presentation.
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return _call_gemini(user_prompt)
        except Exception as e:
            last_error = e
            msg = str(e)
            is_daily_cap = "PerDay" in msg
            is_rate_limit = "429" in msg or "quota" in msg.lower()
            if is_daily_cap:
                print(f"[structure] Gemini DAILY quota exhausted -- switching to Groq fallback.")
                break  # no point retrying Gemini, go straight to fallback below
            if is_rate_limit and attempt < max_attempts - 1:
                wait_seconds = 10 * (attempt + 1)
                print(f"[structure] Gemini rate limited, waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)
                continue
            break  # non-rate-limit error -- no point retrying Gemini, try fallback

    if groq_client:
        try:
            print("[structure] Falling back to Groq (Llama 3.3) for this request...")
            return _call_groq(user_prompt)
        except Exception as e:
            print(f"[structure] Groq fallback also failed: {e}")
            last_error = e
    else:
        print("[structure] No GROQ_API_KEY set -- no fallback available. Add one to .env for resilience.")

    raise last_error


def _extract_all_sources(product: ProductInput, sources: list[SourceHit]) -> dict:
    """One LLM call covering every source at once."""
    if not sources:
        return _empty_extraction()

    source_blocks = []
    for i, s in enumerate(sources):
        text = (s.raw_text or "")[:6000]  # bounded per source since we're now sending several at once
        source_blocks.append(f"--- SOURCE {i} ({s.origin}): {s.url} ---\n{text}")

    combined = "\n\n".join(source_blocks)

    user_prompt = f"""{SYSTEM_PROMPT}

KNOWN PRODUCT INFO:
Part Number: {product.part_number}
Brand: {product.brand}
Short Description: {product.short_description}

SOURCES:
{combined}
"""

    try:
        return _call_llm_with_fallback(user_prompt)
    except Exception as e:
        print(f"[structure] extraction failed on ALL providers: {e}")
        return _empty_extraction()


async def structure_product(product: ProductInput, sources: list[SourceHit]) -> StructuredProduct:
    usable_sources = [s for s in sources if s.raw_text]

    # _extract_all_sources makes BLOCKING network calls (the Gemini/Groq
    # SDKs are synchronous) and can even call time.sleep() during retry
    # backoff. Running that directly inside this async function would
    # freeze the entire event loop -- meaning asyncio.gather() in batch
    # mode couldn't actually run other products at the same time, even
    # though it looks like it should. asyncio.to_thread() pushes the
    # blocking work onto a separate thread so multiple products can
    # genuinely process concurrently in batch mode.
    parsed = await asyncio.to_thread(_extract_all_sources, product, usable_sources)

    fields = {}
    for field_name in SCHEMA_FIELDS:
        entries = parsed.get(field_name, []) or []
        # Resolve each entry's source_index back to the actual SourceHit.
        supporting = []
        for entry in entries:
            idx = entry.get("source_index")
            value = entry.get("value")
            if value and idx is not None and 0 <= idx < len(usable_sources):
                supporting.append((value, usable_sources[idx]))

        if not supporting:
            fields[field_name] = FieldValue(
                value=None, confidence=0.15, source_url=None,
                agreeing_sources=0, needs_review=True,
            )
            continue

        if field_name in FREEFORM_FIELDS:
            supporting.sort(key=lambda pair: len(pair[0]), reverse=True)
            chosen_value, chosen_source = supporting[0]
            fields[field_name] = FieldValue(
                value=chosen_value,
                confidence=0.75 if len(supporting) >= 2 else 0.6,
                source_url=(chosen_source.url or None),
                agreeing_sources=len(supporting),
                needs_review=False,
            )
            continue

        groups = defaultdict(list)
        for value, src in supporting:
            groups[value.strip().lower()].append((value, src))

        best_key = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_key]
        agreeing_count = len(best_group)
        chosen_value, chosen_source = best_group[0]
        conflicting = len(groups) > 1

        if agreeing_count >= 2 and not conflicting:
            confidence = 0.95
            needs_review = False
        elif agreeing_count >= 2 and conflicting:
            confidence = 0.7
            needs_review = True
        elif conflicting:
            confidence = 0.35
            needs_review = True
        else:
            confidence = 0.6
            needs_review = False

        fields[field_name] = FieldValue(
            value=chosen_value,
            confidence=confidence,
            source_url=(chosen_source.url or None),
            agreeing_sources=agreeing_count,
            needs_review=needs_review,
        )

    return StructuredProduct(
        part_number=product.part_number,
        brand=product.brand,
        category=fields["category"],
        description_long=fields["description_long"],
        dimensions=fields["dimensions"],
        material=fields["material"],
        weight=fields["weight"],
        certifications=fields["certifications"],
        sources_used=[s.url for s in usable_sources if s.url],
    )
