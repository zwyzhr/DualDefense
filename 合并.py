import pandas as pd
from pathlib import Path

def merge_csv_folder(csv_dir: Path, out_path: Path):
    """
    合并单个实验文件夹 csv/ 下的 8 个 *_server_acc.csv
    输出为一个总表，每列对应一个算法。
    """
    dfs = []
    for f in csv_dir.glob("*_server_acc.csv"):
        algo = f.stem.replace("_server_acc", "")   # 文件名前缀作为算法名
        df = pd.read_csv(f)
        df = df.rename(columns={"server_test_acc": algo})
        dfs.append(df)

    # 按 round 合并
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="round", how="outer")

    # 按 round 排序
    merged = merged.sort_values("round").reset_index(drop=True)

    merged.to_csv(out_path, index=False)
    print(f"[完成] 合并 {csv_dir} -> {out_path}")

# ===== 批量处理所有实验文件夹 =====
BASE_DIR = Path(r"原始日志数据整理\_summary")   # 修改为你的父目录
print(BASE_DIR)
for exp_dir in BASE_DIR.iterdir():
    if not exp_dir.is_dir():
        print(1)
        continue
    csv_dir = exp_dir / "csv"
    if not csv_dir.exists():
        continue
    print(csv_dir)
    out_path = exp_dir / f"{exp_dir.name}_merged.csv"
    print(out_path)
    merge_csv_folder(csv_dir, out_path)
