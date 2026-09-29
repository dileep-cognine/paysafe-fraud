"""Tests for deterministic and optional LLM merchant-category normalization."""

import pandas as pd
import pytest

from fraud_scoring.features import (
    MerchantCategoryNormalizer,
    build_serving_features,
    normalize_merchant_category,
)


class StubClassifier:
    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0

    def classify(self, normalized_category: str, allowed_categories: frozenset[str]) -> str | None:
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("electronics", "electronics"),
        ("  Phone Shop ", "electronics"),
        ("mobile-store", "electronics"),
        ("restaurant", "dining"),
        ("Food Delivery", "dining"),
    ],
)
def test_known_categories_use_explicit_mapping(raw: str, expected: str):
    result = normalize_merchant_category(raw)
    assert result.category == expected
    assert result.source == "explicit_mapping"


def test_book_store_is_safely_unknown_until_business_approves_a_category():
    result = normalize_merchant_category("Book Store")
    assert result.normalized_input == "book_store"
    assert result.category is None
    assert result.source == "unknown"


def test_unknown_does_not_call_llm_when_disabled():
    classifier = StubClassifier(response="electronics")
    result = normalize_merchant_category(
        "unfamiliar merchant", normalizer=MerchantCategoryNormalizer(classifier, llm_enabled=False)
    )
    assert result.category is None
    assert result.source == "unknown"
    assert classifier.calls == 0


def test_valid_llm_category_is_validated_and_used():
    classifier = StubClassifier(response="electronics")
    result = normalize_merchant_category(
        "new handset kiosk", normalizer=MerchantCategoryNormalizer(classifier, llm_enabled=True)
    )
    assert result.category == "electronics"
    assert result.source == "llm"
    assert classifier.calls == 1


@pytest.mark.parametrize("response", ["some_random_category", "unknown", None])
def test_invalid_llm_category_has_safe_unknown_fallback(response: str | None):
    result = normalize_merchant_category(
        "unfamiliar merchant",
        normalizer=MerchantCategoryNormalizer(StubClassifier(response=response), llm_enabled=True),
    )
    assert result.category is None
    assert result.source == "unknown"


def test_llm_error_has_safe_unknown_fallback():
    result = normalize_merchant_category(
        "unfamiliar merchant",
        normalizer=MerchantCategoryNormalizer(
            StubClassifier(error=TimeoutError()), llm_enabled=True
        ),
    )
    assert result.category is None
    assert result.source == "unknown"


def test_standardized_category_reaches_existing_preprocessing_contract():
    features = build_serving_features(
        pd.DataFrame(
            [
                {
                    "transaction_id": "txn-1",
                    "amount": 45.0,
                    "merchant_category": "phone_shop",
                    "hour_of_day": 12,
                    "device_risk": 0.2,
                }
            ]
        )
    )
    assert features.loc[0, "merchant_category"] == "electronics"
