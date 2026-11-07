# -*- coding: utf-8 -*-
"""
Helpers shared by the validation scripts: comparison tables, data files, plot style.

Data files are CSV files with a header of ``#`` comments giving their source. A
file with no data row is a blank to fill: the scripts say so and carry on without it.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
GREY = "#6b6b6b"


def read_table(name):
    """
    Read a CSV file of the ``data`` folder.

    Parameters
    ----------
    name : str, File name.

    Returns
    -------
    dict of column name to list of float (or str when not a number), or None if
    the file has no data row, in which case the blank is announced.
    """
    path = DATA / name
    with open(path, newline="", encoding="utf-8") as file:
        rows = [row for row in file if row.strip() and not row.startswith("#")]
    records = list(csv.DictReader(rows))
    if not records:
        print(f"  [INPUT NEEDED] data/{name} has no data yet, this comparison is skipped.")
        return None

    def convert(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    return {key: [convert(record[key]) for record in records] for key in records[0]}


def comparison_table(title, rows, tolerance=1.0):
    """
    Print published against computed values.

    Parameters
    ----------
    title : str, Title of the table.
    rows : list of (name, published value, computed value, note).
    tolerance : float, Difference under which a row reads ok [%].

    Returns
    -------
    list of the differences [%].
    """
    print(f"\n{title}")
    print(f"  {'quantity':<28}{'published':>13}{'archibald':>13}{'diff %':>9}  note")
    differences = []
    for name, published, computed, note in rows:
        difference = 100. * (float(computed) - published) / abs(published) if published else float("nan")
        differences.append(difference)
        flag = "" if abs(difference) <= tolerance else " <--"
        print(f"  {name:<28}{published:>13.5g}{float(computed):>13.5g}{difference:>9.2f}  {note}{flag}")
    return differences


def style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive grid, no top and right spines, title on the left."""
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def finish(fig, name):
    """Save the figure in the ``figures`` folder and show it."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.show()
