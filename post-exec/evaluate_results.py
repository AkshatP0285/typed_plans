"""Compute per-experiment metrics from results/predictions.csv."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "predictions.csv"
LABELS = ["OKAY", "ERROR", "NEEDS_FORMAT_CHANGE"]

def main():
    if not RESULTS.exists():
        raise SystemExit("No predictions found. Run run_experiments.py first.")
    df = pd.read_csv(RESULTS)
    for mode, group in df.groupby("experiment"):
        valid = group[(group["error"].fillna("") == "") &
                      group["prediction"].notna() &
                      (group["prediction"] != "")]
        print(f"\n{'=' * 70}\nExperiment: {mode}")
        print(f"Successful predictions: {len(valid)}/{len(group)}")
        if valid.empty:
            print("No successful predictions to score.")
            continue
        y_true = valid["ground_truth"]
        y_pred = valid["prediction"]
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
        print("Confusion matrix (rows=true, columns=predicted):")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS),
                           index=LABELS, columns=LABELS))
        print("\nMean latency (s):", round(valid["latency_seconds"].mean(), 3))
        print("Mean input tokens:", round(valid["input_tokens"].mean(), 2))
        print("Mean state characters:", round(valid["state_chars"].mean(), 1))
        print("Observed model versions:", sorted(valid["model"].dropna().unique().tolist()))
    print("\nImportant: this is a tiny synthetic pilot, not a research-quality estimate.")

if __name__ == "__main__":
    main()
