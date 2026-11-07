"""고정 평가셋으로 모델 성능 측정 (모델 버전 비교용)

  python evaluate.py model/mnist_mlp.npz [다른모델.npz ...]
"""
import json
import sys

import numpy as np

from train import evaluate, load_mnist, robustness_set


def load(path):
    d = np.load(path)
    n = len([k for k in d.files if k.startswith("W")])
    return [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)]


def main(paths):
    _, _, xte, yte = load_mnist()
    xr, yr = robustness_set(xte, yte)
    rows = []
    for p in paths:
        params = load(p)
        rows.append({"model": p, "test_acc": round(evaluate(params, xte, yte), 4),
                     "robust_acc": round(evaluate(params, xr, yr), 4)})
    print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or ["model/mnist_mlp.npz"])
