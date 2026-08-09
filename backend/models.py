"""
Data models shared across the pipeline.
Keeping these in one place means the frontend, extractor, and structurer
all agree on the exact same shape of a "product".
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ProductInput(BaseModel):
    part_number: str
    brand: str
    short_description: str


class SourceHit(BaseModel):
    """One discovered source (a webpage, PDF, or ingested dataset chunk) for a product."""
    url: str
    title: str
    snippet: Optional[str] = None
    raw_text: Optional[str] = None  # filled in during extraction
    origin: str = "web"  # "web" (live search) or "rag" (provided dataset)


class FieldValue(BaseModel):
    """
    A single structured attribute, always paired with where it came from
    and how confident we are in it. This is what makes the output
    explainable instead of a black box.
    """
    value: Optional[str] = None
    confidence: float = 0.0  # 0.0 - 1.0
    source_url: Optional[str] = None
    agreeing_sources: int = 0  # how many independent sources supported this value
    needs_review: bool = False
    review_status: str = "pending"  # "pending" | "approved" | "corrected"


class StructuredProduct(BaseModel):
    part_number: str
    brand: str
    category: FieldValue
    description_long: FieldValue
    dimensions: FieldValue
    material: FieldValue
    weight: FieldValue
    certifications: FieldValue
    sources_used: List[str] = Field(default_factory=list)


class ReviewSubmission(BaseModel):
    part_number: str
    field_name: str
    corrected_value: str


class BatchRequest(BaseModel):
    products: List[ProductInput]


class BatchResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    elapsed_seconds: float
    results: List[StructuredProduct]
