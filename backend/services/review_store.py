"""
Human-in-the-loop review store.

Day 1 simplicity: in-memory dict, keyed by part_number, holding the
latest StructuredProduct plus a log of corrections. Good enough for a
hackathon demo session. Swap for a real DB (SQLite/Postgres) if you
want corrections to survive a server restart -- the interface below
(get_flagged, submit_review) stays the same either way, so upgrading
later is a small change contained to this one file.
"""
from models import StructuredProduct, ReviewSubmission

_products: dict[str, StructuredProduct] = {}
_correction_log: list[dict] = []


def save_product(product: StructuredProduct) -> None:
    _products[product.part_number] = product


def get_product(part_number: str) -> StructuredProduct | None:
    return _products.get(part_number)


def get_all_products() -> list[StructuredProduct]:
    return list(_products.values())


def get_flagged_fields() -> list[dict]:
    """Returns every field across every processed product that needs review."""
    flagged = []
    for product in _products.values():
        for field_name in ["category", "description_long", "dimensions", "material", "weight", "certifications"]:
            field = getattr(product, field_name)
            if field.needs_review and field.review_status == "pending":
                flagged.append({
                    "part_number": product.part_number,
                    "brand": product.brand,
                    "field_name": field_name,
                    "current_value": field.value,
                    "confidence": field.confidence,
                    "agreeing_sources": field.agreeing_sources,
                })
    return flagged


def submit_review(review: ReviewSubmission) -> StructuredProduct | None:
    """Applies a human correction and marks the field resolved."""
    product = _products.get(review.part_number)
    if not product:
        return None

    field = getattr(product, review.field_name, None)
    if field is None:
        return None

    field.value = review.corrected_value
    field.confidence = 1.0
    field.needs_review = False
    field.review_status = "corrected"

    _correction_log.append({
        "part_number": review.part_number,
        "field_name": review.field_name,
        "corrected_value": review.corrected_value,
    })
    save_product(product)
    return product


def get_correction_log() -> list[dict]:
    return _correction_log
