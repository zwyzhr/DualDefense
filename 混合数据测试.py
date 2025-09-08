import re, ast
from pathlib import Path
import matplotlib.pyplot as plt
import math

# ========= 配置 =========
BASE_DIR = Path(r"原始日志数据整理")  # 修改为你的“父目录”
MARKER = "INFO - fl - start - summarization - simulation metrics:"
TAKE = "last"   # 多次出现 marker 时，取 "first" 或 "last"

# （可选）友好名称映射
ALGO_LABEL = {
    "dual_defense": "DDFed",
    "cos_defense":  "Cosine Defense",
    "clipping_median": "Clip Median",
    "trimmed_mean": "Trimmed Mean",
    "fedavg": "FedAvg",
    "krum": "Krum",
    "median": "Median",
    "average": "Average",
}

# （新增）哪些算法在图例中显示为 OUR（可按需增删）
OUR_ALGOS = {"median"}  # 例：把 median 显示为 OUR

# ========= 工具函数 =========
def extract_metric_dict(txt_path, marker=MARKER, which=TAKE):
    """
    从单个 .log 里抽出 marker 之后的 { ... } 数据块为 dict。
    允许日志中 marker 出现多次，which 控制取第一个或最后一个。
    """
    text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    ends = [m.end() for m in re.finditer(re.escape(marker), text)]
    if not ends:
        return None
    s = ends[0] if which == "first" else ends[-1]

    # 截取该段到下一个 "INFO -" 或文件结尾
    nxt = re.search(r"\nINFO\s+-", text[s:], flags=re.DOTALL)
    segment = text[s:] if nxt is None else text[s:s+nxt.start()]

    # 在 segment 中找到第一个完整的大括号块
    i = segment.find('{')
    if i == -1:
        return None
    depth = 0; end_idx = None
    for j, ch in enumerate(segment[i:], start=i):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = j
                break
    if end_idx is None:
        return None
    dict_str = segment[i:end_idx+1]

    # 更健壮一些的替换（按需可删）
    dict_str = (dict_str
                .replace("true", "True")
                .replace("false", "False")
                .replace("nan", "0")
                .replace("inf", "1e9"))

    try:
        dct = ast.literal_eval(dict_str)
        return dct if isinstance(dct, dict) else None
    except Exception as e:
        print(f"[解析失败] {txt_path} -> {e}")
        # print(dict_str[:400])  # 调试
        return None

def get_algo_name(filename: str) -> str:
    """
    从文件名提算法名：fmnist-fedavg-model_poisoning_ipm-xxx.log -> fedavg
    """
    parts = filename.split("-")
    return parts[1] if len(parts) >= 2 else Path(filename).stem

def series_from_log(log_dict):
    rounds = sorted(k for k in log_dict.keys() if isinstance(k, int))
    acc = []
    for r in rounds:
        try:
            acc.append(float(log_dict[r]["server"]["test_acc"]))
        except Exception:
            acc.append(float("nan"))
    return rounds, acc

def collect_series_from_folder(folder: Path):
    """
    读取 folder 内所有 .log，返回 {algo: (rounds, acc)}
    """
    algo_to_series = {}
    for log_file in folder.glob("*.log"):
        algo = get_algo_name(log_file.name)
        dct = extract_metric_dict(log_file)
        if not dct:
            continue
        r, acc = series_from_log(dct)
        if r:
            algo_to_series[algo] = (r, acc)
    return algo_to_series

def plot_folder(folder: Path, out_dir: Path):
    """
    处理单个实验文件夹：
    - 读取所有 .log -> 提取 server test_acc
    - 导出 CSV
    - 画单张图（SVG）
    """
    algo_to_series = collect_series_from_folder(folder)
    if not algo_to_series:
        print(f"[跳过] {folder} 未找到可解析的日志")
        return

    # 导出 CSV
    csv_dir = out_dir / folder.name / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for algo, (r, acc) in algo_to_series.items():
        csv_path = csv_dir / ( "our_server_acc.csv" if algo in OUR_ALGOS else f"{algo}_server_acc.csv" )
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("round,server_test_acc\n")
            for rr, aa in zip(r, acc):
                f.write(f"{rr},{aa}\n")

    # 画图（单图）
    plt.figure(figsize=(8,5))
    for algo, (r, acc) in sorted(algo_to_series.items()):
        label = "OUR" if algo in OUR_ALGOS else ALGO_LABEL.get(algo, algo)
        plt.plot(r, acc, linewidth=2, label=label)

    plt.xlabel("FL training round")
    plt.ylabel("test accuracy (%)")
    plt.title(folder.name)
    plt.legend()
    plt.tight_layout()

    img_dir = out_dir / folder.name
    img_dir.mkdir(parents=True, exist_ok=True)
    # 单图保存路径（与你之前保持一致）
    png_path = Path(rf"原始日志数据整理\{folder.name}.svg")  # 或者改成：img_dir / f"{folder.name}.svg"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[完成-单图] {folder.name} -> {png_path}")

# ========= 新增：多子图总览 =========
def plot_overview_grid(subdirs, out_path, cols=3):
    """
    把多个实验子目录排版到一张大图里（多子图）。
    cols: 每行子图数量。行数自动=ceil(n/cols)。
    """
    subdirs = [p for p in subdirs if p.is_dir()]
    n = len(subdirs)
    if n == 0:
        print("[跳过] 无子目录可画总览")
        return

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)

    for idx, folder in enumerate(subdirs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        algo_to_series = collect_series_from_folder(folder)

        if not algo_to_series:
            ax.set_title(f"")
            ax.axis("off")
            continue

        for algo, (rr, acc) in sorted(algo_to_series.items()):
            label = "OUR" if algo in OUR_ALGOS else ALGO_LABEL.get(algo, algo)
            ax.plot(rr, acc, linewidth=2, label=label)

        ax.set_title(folder.name)
        ax.set_xlabel("FL training round")
        ax.set_ylabel("test accuracy (%)")
        ax.legend(fontsize=8)

    # 清理多余空白子图
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        fig.delaxes(axes[r][c])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成-总览] 多子图保存 -> {out_path}")

# ========= 批量处理：遍历 BASE_DIR 下的每个子文件夹 =========
def main():
    out_root = BASE_DIR / "_summary"
    out_root.mkdir(exist_ok=True, parents=True)

    # 只处理“文件夹”，忽略根目录下散落的 .log
    subdirs = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[提示] {BASE_DIR} 下没有子目录")
        return

    # 逐个输出单图 + CSV
    for folder in subdirs:
        plot_folder(folder, out_root)

    # 额外输出一个总览多子图（默认 3 列 × 自动行数；比如 6 个目录就是 2 行×3 列）
    overview_path =  "原始日志数据整理/总图.pdf"  # 你也可以改为 .png/.pdf
    plot_overview_grid(subdirs, overview_path, cols=3)

if __name__ == "__main__":
    main()
