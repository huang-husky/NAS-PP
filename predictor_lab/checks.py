"""每个 TODO 的自动检查（已写好，不需要改）。

检查没通过时会告诉你哪里不对，然后停止运行；通过了才会继续往下跑。
"""

import sys

import numpy as np

from nb201 import OPS


def _fail(todo, message):
    print(f"\n[TODO {todo} 未通过] {message}")
    sys.exit(1)


def _ok(todo):
    print(f"[TODO {todo} 通过]")


def check_encode(encode):
    arch = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3", "none"]
    vec = encode(arch)
    if vec is None:
        _fail(1, "encode 没有返回值，检查函数末尾有没有 return vec")
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (30,):
        _fail(1, f"向量长度应该是 30（6 条边 x 5 种操作），现在的形状是 {vec.shape}")
    if vec.sum() == 0:
        _fail(1, "向量全是 0，说明还没有把任何位置设成 1")
    if not np.all((vec == 0) | (vec == 1)):
        _fail(1, "向量里只应该出现 0 和 1")
    if vec.sum() != 6:
        _fail(1, f"每条边恰好有一个位置是 1，所以总和应该是 6，现在是 {vec.sum():.0f}")
    blocks = vec.reshape(6, 5)
    if not np.all(blocks.sum(axis=1) == 1):
        _fail(1, "第 k 条边应该占用第 5k ~ 5k+4 这 5 个位置，并且其中恰好一个是 1")
    expected = [OPS.index(op) for op in arch]
    if list(blocks.argmax(axis=1)) != expected:
        _fail(1, "1 的位置不对。第 k 条边的操作在 OPS 里排第 j 个，就应该把第 5k+j 个位置设成 1")
    if np.array_equal(encode(["none"] * 6), encode(["nor_conv_3x3"] * 6)):
        _fail(1, "两个不同的结构得到了相同的编码")
    _ok(1)


def check_split(train_idx, test_idx, n_train, n_total):
    if train_idx is None or test_idx is None:
        _fail(2, "train_idx 或 test_idx 还是 None")
    train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
    if len(train_idx) != n_train:
        _fail(2, f"训练集应该有 {n_train} 个结构，现在有 {len(train_idx)} 个")
    if len(test_idx) != n_total - n_train:
        _fail(2, f"测试集应该是剩下的 {n_total - n_train} 个结构，现在有 {len(test_idx)} 个")
    if len(set(train_idx.tolist()) & set(test_idx.tolist())) > 0:
        _fail(2, "训练集和测试集有重叠。测试用的结构绝对不能在训练时见过")
    if np.array_equal(train_idx, np.arange(n_train)):
        _fail(2, "训练集直接取了前几个结构，应该用打乱后的 perm 来取")
    _ok(2)


def check_pred(model, pred, n_test):
    if model is None:
        _fail(3, "model 还是 None")
    if pred is None:
        _fail(3, "pred 还是 None")
    pred = np.asarray(pred)
    if pred.shape != (n_test,):
        _fail(3, f"应该对测试集里的 {n_test} 个结构各预测一个精度，现在 pred 的形状是 {pred.shape}")
    if np.std(pred) == 0:
        _fail(3, "所有结构的预测值都一样，模型可能没有训练（检查有没有调用 fit）")
    _ok(3)


def check_tau(tau, pred, y_test):
    if tau is None:
        _fail(4, "tau 还是 None")
    try:
        tau = float(tau)
    except TypeError:
        _fail(4, "tau 应该是一个数。kendalltau 会返回两个值（相关系数和 p 值），只取第一个")
    from scipy.stats import kendalltau

    expected = kendalltau(pred, y_test)[0]
    if abs(tau - expected) > 1e-9:
        _fail(4, f"数值不对，应该是 pred 和 y_test 之间的 Kendall's Tau（约 {expected:.4f}）")
    _ok(4)
