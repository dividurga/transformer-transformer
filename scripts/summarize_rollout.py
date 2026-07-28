import sys

from t2.eval.utils import summarize_rollout

if __name__ == "__main__":
    data_path = sys.argv[1]
    summary_stats = summarize_rollout(data_path, use_pbar=True, use_cache=True)
    for k, v in summary_stats.items():
        if k.startswith("metric/actuator"):
            continue
        if k.endswith("/any"):
            print(f"{k}: {v * 100:.1f}%")
        elif k.endswith("/mean") or k == "metric/reward/sum":
            print(f"{k}: {v:.3f}")
