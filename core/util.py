import os
import sys

import numpy as np
from numpy.typing import NDArray


def resource_path(relative_path: str) -> str:
    """取得資源檔案的絕對路徑

    Parameters
    ----------
    relative_path : str
        相對於資源目錄的路徑

    Returns
    -------
    str
        對應的絕對路徑
    """
    if hasattr(sys, "_MEIPASS"):  # 如果是由 PyInstaller 打包執行
        base_path = getattr(sys, "_MEIPASS")  # noqa: B009
    else:
        base_path = os.path.abspath(".")  # 開發模式下，取當前工作目錄
    return os.path.join(base_path, relative_path)


def read_data(filepath: str, *, dim: int | None = None) -> tuple[NDArray, NDArray]:
    """讀取資料，並切分為特徵資料與標籤資料

    Parameters
    ----------
    filepath : str
        資料檔路徑
    dim : int
        特徵維度，若為 None 則預設最後一欄為標籤

    Returns
    -------
    tuple[Matrix, Matrix]
        X : Matrix
            特徵資料，shape: (n_samples, dim)
        Y : Matrix
            標籤資料，shape: (n_samples, n_labels)
    """
    if dim is None:
        dim = -1  # 最後一欄是標籤
    data = np.loadtxt(filepath)
    X = data[:, :dim]
    Y = data[:, dim:]
    return X, Y
