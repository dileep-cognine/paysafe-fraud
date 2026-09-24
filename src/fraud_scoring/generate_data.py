"""Synthetic transaction dataset generator for PaySafe Fraud Scoring.

Generates realistic but strictly synthetic transaction data for development,
testing, and demonstration purposes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "fashion",
    "travel",
    "gaming",
    "dining",
    "crypto",
    "utilities",
]


def generate_synthetic_transactions(
    n_records: int = 5000,
    fraud_ratio: float = 0.04,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic transaction data with controlled fraud signals.

    Args:
        n_records: Total number of transaction rows to generate.
        fraud_ratio: Target approximate proportion of fraudulent transactions.
        random_seed: Random seed for reproducible output.

    Returns:
        pd.DataFrame containing synthetic transactions matching the schema contract.
    """
    rng = np.random.default_rng(seed=random_seed)

    # 1. Unique transaction IDs
    txn_ids = [f"TXN_{i + 1:06d}" for i in range(n_records)]

    # 2. Hours of day (0-23) with day-time peak
    hour_probs = np.array(
        [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,  # 00-05 Night
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.07,  # 06-11 Morning
            0.08,
            0.08,
            0.07,
            0.07,
            0.07,
            0.08,  # 12-17 Afternoon
            0.07,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,  # 18-23 Evening
        ]
    )
    hour_probs = hour_probs / hour_probs.sum()
    hours = rng.choice(np.arange(24), size=n_records, p=hour_probs)

    # 3. Merchant categories with realistic distribution
    category_weights = np.array([0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05])
    categories = rng.choice(ALLOWED_MERCHANT_CATEGORIES, size=n_records, p=category_weights)

    # 4. Amounts (log-normal base with occasional high-value transactions)
    base_amounts = rng.lognormal(mean=3.8, sigma=0.9, size=n_records)  # median ~45
    amounts = np.round(np.clip(base_amounts, 2.0, 5000.0), 2)

    # 5. Device risk scores (Beta distribution skewed towards lower risk for legit)
    device_risks = np.round(rng.beta(a=1.5, b=5.0, size=n_records), 4)

    # 6. Fraud probability calculation (latent risk modeling)
    # Higher risk: high amount, crypto/gaming/electronics, late night hours (0-4), high device_risk
    category_risk_map = {
        "crypto": 1.8,
        "electronics": 1.4,
        "gaming": 1.3,
        "travel": 1.1,
        "fashion": 0.8,
        "dining": 0.6,
        "grocery": 0.4,
        "utilities": 0.3,
    }
    cat_multipliers = np.array([category_risk_map[c] for c in categories])

    is_night = (hours <= 4).astype(float)
    high_amount = (amounts > 350.0).astype(float)

    # Latent log-odds of fraud
    logits = (
        -4.2
        + 1.8 * device_risks
        + 1.0 * cat_multipliers
        + 0.9 * is_night
        + 1.2 * high_amount
        + 0.5 * (amounts / 500.0)
    )
    raw_probs = 1.0 / (1.0 + np.exp(-logits))

    # Scale to match approximate target fraud ratio
    scaling_factor = fraud_ratio / np.mean(raw_probs)
    scaled_probs = np.clip(raw_probs * scaling_factor, 0.001, 0.95)

    is_fraud = (rng.uniform(0, 1, size=n_records) < scaled_probs).astype(int)

    # Elevate device risk and amount for actual fraud cases to create realistic signal
    for i in range(n_records):
        if is_fraud[i] == 1:
            if rng.random() > 0.3:
                device_risks[i] = np.round(rng.uniform(0.60, 0.99), 4)
            if rng.random() > 0.4:
                amounts[i] = np.round(amounts[i] * rng.uniform(1.5, 3.0), 2)

    df = pd.DataFrame(
        {
            "transaction_id": txn_ids,
            "amount": amounts,
            "merchant_category": categories,
            "hour_of_day": hours.astype(int),
            "device_risk": device_risks,
            "is_fraud": is_fraud.astype(int),
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction dataset for PaySafe Fraud Scoring"
    )
    parser.add_argument(
        "--output", default="data/raw/transactions.csv", help="Target CSV output path"
    )
    parser.add_argument("--records", type=int, default=5000, help="Number of records to generate")
    parser.add_argument("--fraud-ratio", type=float, default=0.04, help="Target fraud ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_transactions(
        n_records=args.records,
        fraud_ratio=args.fraud_ratio,
        random_seed=args.seed,
    )
    df.to_csv(out_path, index=False)
    print(
        f"Generated {len(df)} transactions -> {out_path} (Fraud count: {df['is_fraud'].sum()}, Rate: {df['is_fraud'].mean():.2%})"
    )


if __name__ == "__main__":
    main()
