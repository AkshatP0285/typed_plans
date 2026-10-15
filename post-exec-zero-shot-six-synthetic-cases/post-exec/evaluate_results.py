"""Evaluate zero-shot JEV predictions against pilot labels.""" 
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "predictions.csv"
LABELS = ["OKAY", "ERROR", "NEEDS_FORMAT_CHANGE"]

def main():
    if not RESULTS.exists():
        raise SystemExit("No predictions found. Run python run_experiments.py first.")
    df = pd.read_csv(RESULTS)
    valid = df[(df["error"].fillna("") == "") &
               df["prediction"].notna() & (df["prediction"] != "")]
    print(f"Successful predictions: {len(valid)}/{len(df)}")
    if valid.empty:
        print("No successful predictions to score.")
        return
    y_true, y_pred = valid["ground_truth"], valid["prediction"]
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}")
    print("\\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS),
                       index=LABELS, columns=LABELS))
    print("\\nMean latency (s):", round(valid["latency_seconds"].mean(), 3))
    print("Mean input tokens:", round(valid["input_tokens"].mean(), 2))
    print("Mean state characters:", round(valid["state_chars"].mean(), 1))
    print("\\nWARNING: all six labels/outcomes in this pilot are synthetic; these metrics are only a pipeline smoke test.")
    RESULTS.parent.mkdir(exist_ok=True)
    summary = {
        "experiment": "zero_shot",
        "pilot_labels_are_synthetic": True,
        "n_successful_predictions": int(len(valid)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    }
    (RESULTS.parent / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved results/evaluation_summary.json")

if __name__ == "__main__":
    main()
