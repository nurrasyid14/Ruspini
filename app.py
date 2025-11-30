import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import io, zipfile
from datetime import datetime
from sklearn.utils.multiclass import unique_labels
import splitter
import methods.xnn as xnn
from dataset import df
import os

st.title("Nearest Neighbor Exploration (Enhanced Version)")

# ------------------------
# Sidebar: Split Settings
# ------------------------
split_method = st.sidebar.selectbox(
    "Choose Split Method",
    ["KFold", "StratifiedKFold", "RandomSubsampling", "HoldOut", "LeaveOneOut", "LeavePOut", "Bootstrap"]
)

if split_method in ["KFold", "StratifiedKFold", "Bootstrap"]:
    n_splits = st.sidebar.slider("Number of Splits", 2, 10, 5)
else:
    n_splits = None

if split_method in ["RandomSubsampling", "HoldOut"]:
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
else:
    test_size = None

if split_method == "LeavePOut":
    p_value = st.sidebar.number_input("Leave-P-Out (p)", min_value=1, max_value=len(df)-1, value=2, step=1)
    max_splits = st.sidebar.number_input("Max Splits to Run", min_value=1, max_value=1000, value=10, step=1)
else:
    p_value, max_splits = None, None

n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 10, 3)

# ------------------------
# Prepare Data
# ------------------------
X = df[["X", "y"]].values
y = df["Class"].values

# ------------------------
# Run Experiment
# ------------------------
results = []
if split_method == "KFold":
    results = xnn.explore(X, y, splitter.KFold, n_neighbors=n_neighbors, n_splits=n_splits)
elif split_method == "StratifiedKFold":
    results = xnn.explore(X, y, splitter.StratKFoldSplit, n_neighbors=n_neighbors, n_splits=n_splits)
elif split_method == "RandomSubsampling":
    n_splits = st.sidebar.slider("Number of Splits", 2, 100, 5)
    results = xnn.explore(X, y, splitter.RandomSubsampling,
                          n_neighbors=n_neighbors, n_splits=n_splits, test_size=test_size)
elif split_method == "HoldOut":
    train_idx, test_idx = splitter.HoldOut(X, test_size=test_size)
    results = xnn.explore(X, y, splitter.CustomSplit,
                          n_neighbors=n_neighbors, custom_indices=[(train_idx, test_idx)])
elif split_method == "LeaveOneOut":
    results = xnn.explore(X, y, splitter.LeaveOneOut, n_neighbors=n_neighbors)
elif split_method == "LeavePOut":
    results = xnn.explore(
        X, y, splitter.LeavePOut,
        n_neighbors=n_neighbors,
        p=p_value,
        max_splits=max_splits
    )
elif split_method == "Bootstrap":
    results = xnn.explore(X, y, splitter.Bootstrap, n_neighbors=n_neighbors, n_splits=n_splits)

# ------------------------
# Results + Visualization
# ------------------------
for i, r in enumerate(results):

    # LABEL FOR SECTION
    if split_method == "HoldOut":
        subheader_title = "Results Summary for HoldOut"
    elif split_method == "LeaveOneOut":
        subheader_title = f"Results Summary for Leave-One-Out (Sample {i+1})"
    elif split_method == "LeavePOut":
        subheader_title = f"Results Summary for Leave-{p_value}-Out (Split {i+1})"
    else:
        subheader_title = f"Results Summary for {split_method} (Split {i+1})"

    st.subheader(subheader_title)
    st.write(f"Train size: {r['train_size']}, Test size: {r['test_size']}")
    st.write(f"Accuracy: {r['accuracy']:.2f}")
    st.json(r["report"])

    st.write("Confusion Matrix")
    cm = r["confusion_matrix"]
    labels = [str(lbl) for lbl in unique_labels(y)]
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=[f"Pred {lbl}" for lbl in labels],
        y=[f"True {lbl}" for lbl in labels],
        colorscale="Blues",
        showscale=True
    )
    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{i}")

    # ----------------------------------------------
    # NEW: Average distance per predicted class
    # ----------------------------------------------
    test_indices = r["test_indices"]
    dists = r["distances"]

    summary = {}
    for j, idx in enumerate(test_indices):
        pred_class = r["preds"][j]
        if pred_class not in summary:
            summary[pred_class] = []
        summary[pred_class].append(sum(dists[j]) / len(dists[j]))

    st.write("### Average Distance Per Predicted Class")
    st.write(pd.DataFrame({
        "Class": list(summary.keys()),
        "Avg Distance": [sum(v) / len(v) for v in summary.values()]
    }))

    # ----------------------------------------------
    # NEW: Distance visualization for each test item
    # ----------------------------------------------
    st.write("### Distance Visualization (per test sample)")
    for t_idx, dist_list in zip(test_indices, dists):
        fig_dist = go.Figure(data=[
            go.Bar(y=list(range(1, len(dist_list)+1)), x=dist_list, orientation='h')
        ])
        fig_dist.update_layout(
            title=f"Test Index {t_idx}: Distances to {n_neighbors} Neighbors",
            xaxis_title="Distance",
            yaxis_title="Neighbor Rank"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ------------------------
# Export Split Datasets (Memory + Distances)
# ------------------------
st.subheader("Export Split Datasets")

filename_base = f"Ruspini_{split_method}"
if split_method in ["KFold", "StratifiedKFold", "RandomSubsampling", "Bootstrap"]:
    filename_base += f"_splits{n_splits}"
if split_method in ["HoldOut", "RandomSubsampling"]:
    filename_base += f"_test{int(test_size*100)}"
if split_method == "LeavePOut":
    filename_base += f"_p{p_value}"
filename_base += f"_k{n_neighbors}"

excel_path = f"outputs/{filename_base}.xlsx"
os.makedirs("outputs", exist_ok=True)

all_sheets = {"Original": df}

for i, r in enumerate(results):
    split_df = df.copy()
    split_df["Split"] = "Train"

    test_idx_list = list(map(int, r["test_indices"]))
    split_df.loc[test_idx_list, "Split"] = "Test"

    # NEW: Distances column
    dist_col = [None] * len(split_df)
    for j, idx in enumerate(test_idx_list):
        dist_col[idx] = ", ".join(str(v) for v in r["distances"][j])
    split_df["Distances"] = dist_col

    # NEW: Neighbor indices
    nn_col = [None] * len(split_df)
    for j, idx in enumerate(test_idx_list):
        nn_col[idx] = ", ".join(str(int(v)) for v in r["neighbors"][j])
    split_df["Neighbor_Indices"] = nn_col

    all_sheets[f"Split_{i+1}"] = split_df

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    for name, sheet_df in all_sheets.items():
        sheet_df.to_excel(writer, sheet_name=name[:31], index=False)

st.success(f"Latest results automatically exported to {excel_path}")


# ------------------------
# Manual ZIP Export
# ------------------------
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    excel_bytes = io.BytesIO()
    with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
        for name, sheet_df in all_sheets.items():
            sheet_df.to_excel(writer, sheet_name=name[:31], index=False)
    excel_bytes.seek(0)
    zf.writestr(f"{filename_base}.xlsx", excel_bytes.read())

    for name, sheet_df in all_sheets.items():
        zf.writestr(f"{filename_base}_{name}.csv", sheet_df.to_csv(index=False))
        zf.writestr(f"{filename_base}_{name}.tsv", sheet_df.to_csv(index=False, sep="\t"))

zip_buffer.seek(0)

st.download_button(
    label="Download All Splits (Excel + CSV + TSV in ZIP)",
    data=zip_buffer,
    file_name=f"{filename_base}_bundle.zip",
    mime="application/zip"
)


# ------------------------
# Cluster Visualization
# ------------------------
st.subheader("Cluster Visualization (Class = Color+Shape, Split = Opacity)")

class_styles = {
    1: {"color": "red", "symbol": "circle"},
    2: {"color": "blue", "symbol": "square"},
    3: {"color": "green", "symbol": "diamond"},
    4: {"color": "purple", "symbol": "triangle-up"}
}

for i, r in enumerate(results):
    split_df = df.copy()
    split_df["Split"] = "Train"
    split_df.loc[list(map(int, r["test_indices"])), "Split"] = "Test"

    fig_split = go.Figure()
    for class_label, style in class_styles.items():
        for split in ["Train", "Test"]:
            subset = split_df[(split_df["Class"] == class_label) & (split_df["Split"] == split)]
            if not subset.empty:
                fig_split.add_trace(go.Scatter(
                    x=subset["X"], y=subset["y"],
                    mode="markers",
                    name=f"Class {class_label} ({split})",
                    marker=dict(
                        color=style["color"],
                        symbol=style["symbol"],
                        size=12,
                        opacity=1.0 if split == "Train" else 0.4,
                        line=dict(width=1, color="black")
                    )
                ))
    fig_split.update_layout(title=f"Split {i+1}: Class Clusters with Train/Test Split")
    st.plotly_chart(fig_split, use_container_width=True)
