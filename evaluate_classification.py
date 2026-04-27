import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

DATA_PATH = 'data/processed/medical_dataset.csv'
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def compute_classification_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        'accuracy': float(acc),
        'per_label': {}
    }
    for i, lab in enumerate(labels or sorted(set(y_true) | set(y_pred))):
        metrics['per_label'][lab] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    # Macro / weighted
    metrics['macro_f1'] = float(sum(f1) / len(f1)) if len(f1) else 0.0
    return metrics


def main():
    print('Loading dataset...')
    df = load_dataset(DATA_PATH)

    if 'label' not in df.columns:
        raise ValueError('Dataset must contain a `label` column with true labels')

    # If predictions present, use them; otherwise try to load a saved ML model predictions column
    if 'predicted_label' in df.columns:
        y_true = df['label'].astype(str).tolist()
        y_pred = df['predicted_label'].astype(str).tolist()
    else:
        # No predictions available; attempt to use simple stored ML model outputs if present
        # This keeps the script minimal: user can add `predicted_label` column to dataset to evaluate.
        raise ValueError('No `predicted_label` column found. Add predictions to the dataset before evaluation.')

    labels = sorted(list(set(y_true) | set(y_pred)))
    print('Computing metrics for labels:', labels)

    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

    # Save metrics
    out_json = os.path.join(OUTPUT_DIR, 'classification_metrics.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Also save a simple CSV summary
    rows = []
    for lab, m in metrics['per_label'].items():
        rows.append({
            'label': lab,
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'support': m['support']
        })
    summary_df = pd.DataFrame(rows)
    summary_df.loc[len(summary_df)] = ['macro', None, None, metrics.get('macro_f1'), None]
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_metrics_per_label.csv'), index=False)

    print('Saved metrics to', out_json)
    print('Summary per-label CSV saved to', os.path.join(OUTPUT_DIR, 'classification_metrics_per_label.csv'))


if __name__ == '__main__':
    main()
