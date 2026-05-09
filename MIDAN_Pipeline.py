"""
Compatibility wrapper for the MIDAN training pipeline.

The real training workflow lives in `MIDAN_Pipeline.ipynb`, which depends on a
notebook/Colab runtime, uploaded CSVs, and interactive cells. The previous
`.py` export contained raw notebook shell magics and Colab-only calls, so it
could not be imported or executed as normal Python.

This wrapper keeps the repository runnable from standard Python tools while
pointing developers to the supported training entrypoint.
"""

from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_suffix(".ipynb")


def main() -> None:
    print("MIDAN training is notebook-driven.")
    print(f"Open the notebook instead: {NOTEBOOK_PATH.name}")
    print("Run it in Jupyter or Google Colab to regenerate the model artifacts in models/.")


if __name__ == "__main__":
    main()
