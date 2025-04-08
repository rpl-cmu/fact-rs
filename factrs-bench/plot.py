# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "polars",
#     "pyqt6",
#     "seaborn",
# ]
# ///
import json
from pathlib import Path
import polars as pl  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt
import matplotlib

DIR = Path("factrs-bench")


def setup_plot():
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"


# Load data from rust.json
with open(DIR / "rust.json", "r") as rust_file:
    rust_data = json.load(rust_file)
with open(DIR / "cpp_3d.json", "r") as cpp_file:
    cpp_data = json.load(cpp_file)
with open(DIR / "cpp_2d.json", "r") as cpp_file:
    cpp_2d_data = json.load(cpp_file)
    cpp_data["results"].extend(cpp_2d_data["results"])

# Extract benchmark results
rust_benchmarks = [
    {
        "method": func["name"],
        "filename": filename.replace('"', ""),
        "time": t / 1e9,  # Convert from us to ms
    }
    for group in rust_data["groups"].values()
    for func in group["function"]
    for timings, filename in zip(func["timings"], group["args"]["Named"])
    for t in timings
]
cpp_benchmarks = [
    {
        "method": result["name"].split("_")[0],
        "filename": result["name"].split("_")[1],
        "time": mm["elapsed"] * 1e3,  # Convert from s to ms
    }
    for result in cpp_data["results"]
    for mm in result["measurements"]
]

# Create a Polars DataFrame
df = pl.DataFrame(rust_benchmarks + cpp_benchmarks)

# Pretty up the names
df = df.with_columns(
    pl.col("filename")
    .replace(
        {
            "M3500.g2o": "M3500",
            "parking-garage.g2o": "Parking Garage",
            "sphere2500.g2o": "Sphere2500",
        }
    )
    .alias("Dataset"),
    pl.col("method")
    .replace(
        {
            "sophus": "sophus-rs",
            "factrs": "fact-rs",
            "tinysolver": "tiny-solver-rs",
        }
    )
    .alias("Library"),
)

# Plot them using seaborn
setup_plot()
fig, ax = plt.subplots(1, 1, figsize=(252.0 / 72.27 + 0.5, 2.5), layout="constrained")
sns.barplot(
    data=df,
    x="Dataset",
    y="time",
    hue="Library",
    ax=ax,
    estimator="median",
    errorbar=lambda x: (x.min(), x.max()),  # type: ignore
)

ax.tick_params(axis="x", pad=-2)
ax.tick_params(axis="y", pad=-1)
ax.set_ylabel("Time (ms)")


ax.set_yscale("symlog", linthresh=600, linscale=1.5)
ax.set_yticks([0, 200, 400, 600, 1000, 1600], [0, 200, 400, 600, 1000, 1600])

ax.legend().set_visible(False)
leg = fig.legend(
    ncol=3,
    borderpad=0.2,
    labelspacing=0.15,
    loc="outside upper left",
    columnspacing=3.65,
    bbox_to_anchor=(0.125, 1.15),
).get_frame()
leg.set_boxstyle("square")  # type: ignore
leg.set_linewidth(1.0)

plt.savefig(DIR / "benchmarks.png", dpi=300, bbox_inches="tight")
plt.savefig(DIR / "benchmarks.pdf", dpi=300, bbox_inches="tight")
# plt.show()
