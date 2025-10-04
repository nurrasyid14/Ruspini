import pandas as pd
import numpy as np

# ------------------------
# Internal helper
# ------------------------
def _ensure_dataframe(source):
    """Convert supported input types to pandas DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source
    elif isinstance(source, dict):
        return pd.DataFrame(source)
    elif isinstance(source, np.ndarray):
        return pd.DataFrame(source)
    else:
        raise ValueError("Unsupported data type. Must be DataFrame, dict, or ndarray.")


# ------------------------
# Saving functions
# ------------------------
def toCSV(source, out="output.csv"):
    df = _ensure_dataframe(source)
    df.to_csv(out, index=False)
    print(f"Dataset saved to {out}")


def toTSV(source, out="output.tsv"):
    df = _ensure_dataframe(source)
    df.to_csv(out, index=False, sep="\t")
    print(f"Dataset saved to {out}")


def toXLS(source, out="output.xlsx"):
    df = _ensure_dataframe(source)
    df.to_excel(out, index=False, engine="openpyxl")
    print(f"Dataset saved to {out}")


def saveDataset(source, out):
    """
    Auto-detect format from file extension (.csv, .tsv, .xlsx).
    """
    df = _ensure_dataframe(source)

    if out.endswith(".csv"):
        df.to_csv(out, index=False)
    elif out.endswith(".tsv"):
        df.to_csv(out, index=False, sep="\t")
    elif out.endswith(".xls") or out.endswith(".xlsx"):
        df.to_excel(out, index=False, engine="openpyxl")
    else:
        raise ValueError("Unsupported extension. Use .csv, .tsv, or .xlsx")

    print(f"Dataset saved to {out}")

def toMultiXLS(datasets: dict, out="outputs/all_splits.xlsx"):
    """
    Save multiple datasets into one Excel file with multiple sheets.
    datasets : dict of {sheet_name: DataFrame}
    """
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in datasets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"All datasets saved to {out}")
    return out

# ------------------------
# Loading functions
# ------------------------
def fromCSV(path):
    return pd.read_csv(path)


def fromTSV(path):
    return pd.read_csv(path, sep="\t")


def fromXLS(path):
    return pd.read_excel(path, engine="openpyxl")

