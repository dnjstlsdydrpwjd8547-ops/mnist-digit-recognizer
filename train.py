"""MNIST MLP 학습 스크립트 (NumPy only)

사용법:
  python train.py --version v1                 # 기본 모델 (HW2)
  python train.py --version v2 --augment affine --hidden 800 400 --epochs 20
결과: model/mnist_mlp.npz + model/model_info.json
"""
import argparse
import gzip
import json
import os
import sys
import urllib.request

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def load_mnist():
    os.makedirs(DATA, exist_ok=True)
    for n in FILES:
        p = os.path.join(DATA, n)
        if not os.path.exists(p):
            print("download", n, file=sys.stderr)
            urllib.request.urlretrieve(URL + n, p)

    def rd(n, off):
        return np.frombuffer(gzip.open(os.path.join(DATA, n)).read(), np.uint8, offset=off)

    xtr = rd(FILES[0], 16).reshape(-1, 28, 28).astype(np.float32) / 255
    ytr = rd(FILES[1], 8).astype(np.int64)
    xte = rd(FILES[2], 16).reshape(-1, 28, 28).astype(np.float32) / 255
    yte = rd(FILES[3], 8).astype(np.int64)
    return xtr, ytr, xte, yte


def affine_batch(b, rng, rot=15, scale=(0.85, 1.15), shift=2.5):
    """회전·확대/축소·이동 (손으로 그린 숫자의 변형을 흉내냄)"""
    from scipy.ndimage import affine_transform
    out = np.empty_like(b)
    for i in range(len(b)):
        a = np.deg2rad(rng.uniform(-rot, rot))
        s = rng.uniform(*scale)
        m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) / s
        c = np.array([13.5, 13.5])
        off = c - m @ c + rng.uniform(-shift, shift, 2)
        out[i] = affine_transform(b[i], m, offset=off, order=1, mode="constant")
    return out


def shift_batch(b, rng):
    out = np.zeros_like(b)
    for i in range(len(b)):
        dx, dy = rng.integers(-2, 3, 2)
        out[i] = np.roll(np.roll(b[i], dy, 0), dx, 1)
    return out


def forward(params, x, train=False, rng=None, dropout=0.0):
    W, B = params
    hs, masks = [x], []
    for i in range(len(W)):
        z = hs[-1] @ W[i] + B[i]
        if i < len(W) - 1:
            z = np.maximum(z, 0)
            if train and dropout > 0:
                m = (rng.random(z.shape) > dropout) / (1 - dropout)
                z = z * m
                masks.append(m)
            else:
                masks.append(None)
        hs.append(z)
    return hs, masks


def evaluate(params, x, y):
    return float((forward(params, x.reshape(len(x), -1))[0][-1].argmax(1) == y).mean())


def robustness_set(xte, yte, seed=1234):
    """고정 시드 변형 테스트셋: 모델 버전 간 공정 비교용"""
    return affine_batch(xte, np.random.default_rng(seed), rot=20, scale=(0.75, 1.2), shift=3), yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--augment", choices=["shift", "affine"], default="shift")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(BASE, "model"))
    args = ap.parse_args()

    xtr, ytr, xte, yte = load_mnist()
    rng = np.random.default_rng(0)
    sizes = [784] + args.hidden + [10]
    W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
    B = [np.zeros(b, np.float32) for b in sizes[1:]]
    P = W + B
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    t = 0
    aug = affine_batch if args.augment == "affine" else shift_batch
    L = len(W)
    for ep in range(args.epochs):
        lr = args.lr * (0.85 ** ep)
        idx = rng.permutation(len(xtr))
        for s in range(0, len(idx), 128):
            j = idx[s:s + 128]
            x = aug(xtr[j], rng).reshape(len(j), -1)
            y = ytr[j]
            hs, masks = forward((W, B), x, True, rng, args.dropout)
            z = hs[-1] - hs[-1].max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            g = p
            g[np.arange(len(y)), y] -= 1
            g /= len(y)
            gW, gB = [None] * L, [None] * L
            for i in reversed(range(L)):
                gW[i] = hs[i].T @ g + 1e-5 * W[i]
                gB[i] = g.sum(0)
                if i:
                    g = g @ W[i].T
                    if masks[i - 1] is not None:
                        g = g * masks[i - 1]
                    g = g * (hs[i] > 0)
            t += 1
            for k, (pp, gr) in enumerate(zip(P, gW + gB)):
                m[k] = 0.9 * m[k] + 0.1 * gr
                v[k] = 0.999 * v[k] + 0.001 * gr * gr
                pp -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        print(f"epoch {ep + 1}/{args.epochs} test_acc={evaluate((W, B), xte, yte):.4f}", flush=True)

    xr, yr = robustness_set(xte, yte)
    info = {
        "version": args.version,
        "architecture": "MLP " + "-".join(map(str, sizes)),
        "augment": args.augment,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "test_acc": round(evaluate((W, B), xte, yte), 4),
        "robust_acc": round(evaluate((W, B), xr, yr), 4),
    }
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "mnist_mlp.npz"),
                        **{f"W{i}": W[i] for i in range(L)}, **{f"b{i}": B[i] for i in range(L)})
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
