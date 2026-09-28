import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate_transactions
from features import prepare_features

def test_generated_data():
    df = generate_transactions(200)
    assert len(df) == 200
    assert df["is_fraud"].isin([0,1]).all()

def test_features():
    df = generate_transactions(10).drop(columns=["is_fraud"])
    X = prepare_features(df)
    assert len(X) == 10
    assert "amount" in X.columns
