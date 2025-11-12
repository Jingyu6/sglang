import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

tasks = ["humaneval_all", "mbpp_all", "gsm8k_cot", "minerva_math"]
models = ["qwen3", "qwen3_eagle", "tidar"]

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    metric_key = "output_throughput"
    x_positions = np.arange(len(tasks))
    bar_width = 0.22

    # Consistent colors per model
    palette = sns.color_palette("Set2", n_colors=len(models))
    model_to_color = {model: palette[idx] for idx, model in enumerate(models)}

    # Collect values: model -> list of metric values per task
    model_to_values = {model: [] for model in models}
    model_to_accept = {model: [] for model in models}
    for task in tasks:
        for model in models:
            path = os.path.join(base_dir, f"{model}_8b_{task}.jsonl")
            with open(path, "r") as f:
                stats = json.load(f)
            value = stats.get(metric_key, None)
            model_to_values[model].append(value)
            model_to_accept[model].append(stats.get("accept_length", None))

    plt.figure(figsize=(11, 6))

    # For annotation offsets
    all_values = [v for vals in model_to_values.values() for v in vals if v is not None]
    y_max = max(all_values) if all_values else 1.0
    y_offset = 0.02 * y_max
    baseline_values = model_to_values.get("qwen3", [None] * len(tasks))

    # Plot grouped bars
    for idx, model in enumerate(models):
        offsets = (idx - (len(models) - 1) / 2) * bar_width
        positions = x_positions + offsets
        plt.bar(
            positions,
            model_to_values[model],
            width=bar_width,
            color=model_to_color[model],
            label=model,
            edgecolor="black",
            linewidth=0.5,
        )
        # Annotate improvement over qwen3 for qwen3_eagle and tidar
        if model in ("qwen3_eagle", "tidar"):
            for x_pos, value, base in zip(positions, model_to_values[model], baseline_values):
                if value is None or base in (None, 0):
                    label = "NA"
                    y_text = (value or 0) + y_offset
                else:
                    ratio = value / base
                    label = f"{ratio:.2f}x"
                    y_text = value + y_offset
                plt.text(x_pos, y_text, label, ha="center", va="bottom", fontsize=9)
            # Annotate accept_length inside bars
            for x_pos, value, acc in zip(positions, model_to_values[model], model_to_accept[model]):
                if value is None:
                    continue
                inner_y = value * 0.5
                acc_label = f"{acc:.2f}\nT/NFE" if model == "tidar" else f"{acc + 1:.2f}\nT/NFE"
                plt.text(x_pos, inner_y, acc_label, ha="center", va="center", fontsize=8)

    # Axes and legend
    plt.xticks(x_positions, tasks, rotation=15)
    plt.ylabel("Output throughput (tokens/s)")
    plt.xlabel("Task")
    plt.title("Decode Throughput by Task and Model")
    plt.legend(title="Model")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.ylim(plt.ylim()[0], plt.ylim()[1] * 1.2)
    plt.tight_layout()

    # Save and show
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_throughput_barplot.png")
    plt.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()