"""Interactive Matplotlib viewer for the volatility clustering dot plot."""

import pickle
import tkinter as tk
from pathlib import Path

from matplotlib.backends.backend_tkagg import new_figure_manager_given_figure


def main():
    root_dir = Path(__file__).resolve().parent.parent
    pickle_path = root_dir / "imgs" / "data_properties" / "volatility_clustering_dots.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        fig = pickle.load(f)

    mgr = new_figure_manager_given_figure(1, fig)
    mgr.set_window_title("Figure 6.23: Volatility Clustering Dots - Matplotlib Interactive Inspector")
    mgr.show()

    # Bring window to front
    mgr.window.lift()
    mgr.window.attributes("-topmost", True)
    mgr.window.after_idle(mgr.window.attributes, "-topmost", False)
    mgr.window.focus_force()

    tk.mainloop()


if __name__ == "__main__":
    main()
