import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch

from train_kaggle_rnn_cnn import CNNClassifier, RNNClassifier, text_to_sequence
from utils.gemini_integration import check_gemini_api_key, gemini_explain_classification


st.set_page_config(page_title="Medical Statement Classifier", page_icon="🩺", layout="wide")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_file(filename):
    search_roots = [".", os.getcwd(), "/kaggle/working", "/kaggle/input"]
    candidates = []

    for root in search_roots:
        if os.path.exists(root):
            candidates.extend(glob.glob(os.path.join(root, "**", filename), recursive=True))

    seen = set()
    candidates = [path for path in candidates if not (path in seen or seen.add(path))]

    if not candidates:
        raise FileNotFoundError(f"Could not find {filename}")

    candidates.sort(key=lambda path: (0 if os.path.dirname(path).endswith("models") else 1, len(path)))
    return candidates[0]


def load_metadata():
    metrics_path = find_file("metrics_summary.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    vocab_path = find_file("vocabulary.json")
    with open(vocab_path, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    labels = metrics.get("label_encoder_classes")
    if not labels:
        labels = ["credible", "false", "misleading"]

    max_seq_length = int(metrics.get("max_seq_length", 512))
    return vocab, labels, max_seq_length


def build_model(model_type, vocab_size, num_classes):
    if model_type == "CNN":
        return CNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            num_classes=num_classes,
            dropout=0.5,
        )

    if model_type == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.5,
        )

    raise ValueError("model_type must be CNN or RNN")


@st.cache_resource
def load_models_and_assets():
    device = get_device()
    vocab, labels, max_seq_length = load_metadata()

    cnn_model_path = find_file("cnn_final.pt")
    rnn_model_path = find_file("rnn_final.pt")

    cnn_model = build_model("CNN", len(vocab), len(labels)).to(device)
    cnn_state = torch.load(cnn_model_path, map_location=device)
    cnn_model.load_state_dict(cnn_state)
    cnn_model.eval()

    rnn_model = build_model("RNN", len(vocab), len(labels)).to(device)
    rnn_state = torch.load(rnn_model_path, map_location=device)
    rnn_model.load_state_dict(rnn_state)
    rnn_model.eval()

    return {
        "device": device,
        "vocab": vocab,
        "labels": labels,
        "max_seq_length": max_seq_length,
        "cnn_model": cnn_model,
        "rnn_model": rnn_model,
        "cnn_model_path": cnn_model_path,
        "rnn_model_path": rnn_model_path,
    }


def load_report_assets():
    metrics_path = find_file("metrics_summary.json")
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    assets = {
        "metrics": metrics,
        "cnn_training_history": None,
        "rnn_training_history": None,
        "cnn_confusion_matrix": None,
        "rnn_confusion_matrix": None,
    }

    optional_files = {
        "cnn_training_history": "cnn_training_history.png",
        "rnn_training_history": "rnn_training_history.png",
        "cnn_confusion_matrix": "cnn_confusion_matrix.png",
        "rnn_confusion_matrix": "rnn_confusion_matrix.png",
    }

    for key, filename in optional_files.items():
        try:
            assets[key] = find_file(filename)
        except FileNotFoundError:
            assets[key] = None

    return assets


def predict_probs(model, statement, vocab, max_seq_length, device):
    seq = np.array([text_to_sequence(statement, vocab, max_seq_length)])
    tensor_x = torch.LongTensor(seq).to(device)

    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def probs_to_table(labels, probs, prefix):
    rows = []
    for idx, label in enumerate(labels):
        rows.append({
            "label": label,
            f"{prefix}_probability": float(probs[idx]),
        })
    return pd.DataFrame(rows).sort_values(by=f"{prefix}_probability", ascending=False)


def artifact_available(filename):
    try:
        find_file(filename)
        return True
    except FileNotFoundError:
        return False


def render_sidebar():
    st.sidebar.title("Medical AI Console")
    st.sidebar.caption("CNN + RNN credibility checker")

    if "page" not in st.session_state:
        st.session_state.page = "Classifier"

    st.sidebar.markdown("### Pages")
    if st.sidebar.button("Classifier", use_container_width=True):
        st.session_state.page = "Classifier"
    if st.sidebar.button("Training Report", use_container_width=True):
        st.session_state.page = "Training Report"

    return st.session_state.page


def render_classification_page():
    st.title("Medical Misinformation Detection")
    st.caption("CNN + RNN ensemble prediction with per-label confidence scores")

    try:
        assets = load_models_and_assets()
    except Exception as exc:
        st.error("Model files not found. Train models first to generate artifacts in models/.")
        st.code(str(exc))
        st.info("Required files: cnn_final.pt, rnn_final.pt, vocabulary.json, metrics_summary.json")
        return



    with st.expander("Loaded artifacts"):
        st.write(f"CNN model: {assets['cnn_model_path']}")
        st.write(f"RNN model: {assets['rnn_model_path']}")

    statement = st.text_area(
        "Enter a medical statement",
        placeholder="Example: Regular exercise and a balanced diet can help reduce the risk of heart disease.",
        height=120,
    )

    if st.button("Classify Statement", type="primary"):
        if not statement.strip():
            st.warning("Please enter a statement first.")
            return

        labels = assets["labels"]
        device = assets["device"]

        cnn_probs = predict_probs(
            assets["cnn_model"],
            statement,
            assets["vocab"],
            assets["max_seq_length"],
            device,
        )
        rnn_probs = predict_probs(
            assets["rnn_model"],
            statement,
            assets["vocab"],
            assets["max_seq_length"],
            device,
        )

        final_probs = (cnn_probs + rnn_probs) / 2.0

        cnn_idx = int(np.argmax(cnn_probs))
        rnn_idx = int(np.argmax(rnn_probs))
        final_idx = int(np.argmax(final_probs))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("CNN")
            st.success(f"{labels[cnn_idx]} ({cnn_probs[cnn_idx]:.4f})")
        with c2:
            st.subheader("RNN")
            st.success(f"{labels[rnn_idx]} ({rnn_probs[rnn_idx]:.4f})")
        with c3:
            st.subheader("Final")
            st.success(f"{labels[final_idx]} ({final_probs[final_idx]:.4f})")

        st.markdown("### Confidence By Label")
        t1, t2, t3 = st.tabs(["CNN", "RNN", "Final (Avg)"])
        with t1:
            st.dataframe(probs_to_table(labels, cnn_probs, "cnn"), use_container_width=True)
        with t2:
            st.dataframe(probs_to_table(labels, rnn_probs, "rnn"), use_container_width=True)
        with t3:
            st.dataframe(probs_to_table(labels, final_probs, "final"), use_container_width=True)

        st.markdown("### AI Explanation")
        st.markdown(
            f"""
            <div style="padding: 0.85rem 1rem; border-radius: 0.75rem; background: #e3f2fd; border: 1px solid #90caf9; margin-bottom: 1rem;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #1565c0; text-transform: uppercase; letter-spacing: 0.04em;">Model Prediction</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0d47a1; margin-top: 0.15rem;">{labels[final_idx]} ({final_probs[final_idx]:.2%})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        api_ready, api_message = check_gemini_api_key()
        if api_ready:
            with st.spinner("Generating Gemini fact-check..."):
                explanation = gemini_explain_classification(
                    statement,
                    labels[final_idx],
                    float(final_probs[final_idx]),
                )
            if explanation:
                st.markdown(
                    f"""
                    <div style="padding: 0.85rem 1rem; border-radius: 0.75rem; background: #f3e5f5; border: 1px solid #ce93d8; margin-bottom: 1rem;">
                        <div style="font-size: 0.9rem; font-weight: 700; color: #6a1b9a; text-transform: uppercase; letter-spacing: 0.04em;">Is Prediction Correct?</div>
                        <div style="font-size: 1rem; color: #4a148c; margin-top: 0.5rem;">{explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Gemini did not return an explanation.")
        else:
            st.info(f"Gemini explanation unavailable: {api_message}")


def render_report_page():
    st.title("Training Report")
    st.caption("Model metrics, loss/accuracy curves, and confusion matrices")

    try:
        report_assets = load_report_assets()
    except Exception as exc:
        st.error("Report assets not found. Please run training first.")
        st.code(str(exc))
        st.info("Required: metrics_summary.json and chart images in models/.")
        return

    metrics = report_assets["metrics"]
    cnn_metrics = metrics.get("CNN", {})
    rnn_metrics = metrics.get("RNN", {})

    st.subheader("Core Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CNN Accuracy", f"{cnn_metrics.get('accuracy', 0.0):.4f}")
    with c2:
        st.metric("RNN Accuracy", f"{rnn_metrics.get('accuracy', 0.0):.4f}")
    with c3:
        st.metric("CNN F1-Macro", f"{cnn_metrics.get('f1_macro', 0.0):.4f}")
    with c4:
        st.metric("RNN F1-Macro", f"{rnn_metrics.get('f1_macro', 0.0):.4f}")

    st.subheader("Detailed Metrics")
    details_rows = [
        {
            "model": "CNN",
            "accuracy": cnn_metrics.get("accuracy", 0.0),
            "precision_macro": cnn_metrics.get("precision_macro", 0.0),
            "recall_macro": cnn_metrics.get("recall_macro", 0.0),
            "f1_macro": cnn_metrics.get("f1_macro", 0.0),
            "auc_macro": cnn_metrics.get("auc_macro", 0.0),
        },
        {
            "model": "RNN",
            "accuracy": rnn_metrics.get("accuracy", 0.0),
            "precision_macro": rnn_metrics.get("precision_macro", 0.0),
            "recall_macro": rnn_metrics.get("recall_macro", 0.0),
            "f1_macro": rnn_metrics.get("f1_macro", 0.0),
            "auc_macro": rnn_metrics.get("auc_macro", 0.0),
        },
    ]
    st.dataframe(pd.DataFrame(details_rows), use_container_width=True)

    st.subheader("Training Curves")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**CNN Loss/Accuracy**")
        if report_assets["cnn_training_history"]:
            st.image(report_assets["cnn_training_history"], use_container_width=True)
        else:
            st.warning("cnn_training_history.png not found")
    with tc2:
        st.markdown("**RNN Loss/Accuracy**")
        if report_assets["rnn_training_history"]:
            st.image(report_assets["rnn_training_history"], use_container_width=True)
        else:
            st.warning("rnn_training_history.png not found")

    st.subheader("Confusion Matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("**CNN Confusion Matrix**")
        if report_assets["cnn_confusion_matrix"]:
            st.image(report_assets["cnn_confusion_matrix"], use_container_width=True)
        else:
            st.warning("cnn_confusion_matrix.png not found")
    with cm2:
        st.markdown("**RNN Confusion Matrix**")
        if report_assets["rnn_confusion_matrix"]:
            st.image(report_assets["rnn_confusion_matrix"], use_container_width=True)
        else:
            st.warning("rnn_confusion_matrix.png not found")

    with st.expander("Raw Metrics JSON"):
        st.json(metrics)


def main():
    page = render_sidebar()

    if page == "Classifier":
        render_classification_page()
    else:
        render_report_page()


if __name__ == "__main__":
    main()
