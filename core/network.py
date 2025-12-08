from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray


class ConscienceMechanism:
    def __init__(
        self,
        beta: float = 0.001,
        c_factor: float = 10.0,
    ) -> None:
        """
        良心機構 (Conscience Mechanism) 模組。

        初始化時僅設定超參數，網格尺寸由 SOM 在整合時透過 `setup` 方法注入。

        Parameters
        ----------
        beta : float, optional
            參數 B (Beta)，控制獲勝機率 p_j 的更新速率 (歷史記憶長度)。
        c_factor : float, optional
            參數 C (Bias Factor)，控制偏差項的影響力。
        """
        self.beta = beta
        self.c_factor = c_factor

        # 在 setup() 初始化
        self.n_nodes: int = 0
        # 在 setup() 初始化
        self.win_prob: NDArray[np.float64] | None = None

    def setup(self, x: int, y: int) -> None:
        """
        根據 SOM 的網格尺寸初始化內部狀態。

        Parameters
        ----------
        x : int
            SOM 網格寬度。
        y : int
            SOM 網格高度。
        """
        self.n_nodes = x * y
        self.win_prob = np.full((x, y), 1.0 / self.n_nodes)

    def compute_bias(self) -> NDArray[np.float64]:
        """
        計算距離偏差值 (Bias)。

        對應公式: b_j = C * (1/N - p_j)

        Returns
        -------
        NDArray[np.float64]
            計算出的偏差矩陣，形狀為 (x, y)。

        Raises
        ------
        RuntimeError
            若此機制尚未透過 setup 初始化。
        """
        if self.win_prob is None:
            raise RuntimeError("ConscienceMechanism has not been setup with grid dimensions.")

        return self.c_factor * (1.0 / self.n_nodes - self.win_prob)

    def update(self, bmu_idx: tuple[np.int64, ...]) -> None:
        """
        更新獲勝機率狀態 (State Update)。

        對應公式: p_j_new = p_j_old + B * (y_j - p_j_old)
        優化邏輯:
            1. 全體衰減: p_j = p_j * (1 - B)
            2. 勝者增強: p_win += B * 1

        Parameters
        ----------
        bmu_idx : tuple[np.int64, ...]
            BMU 網格座標。

        Raises
        ------
        RuntimeError
            若尚未透過 setup 初始化。
        """
        if self.win_prob is None:
            raise RuntimeError("ConscienceMechanism has not been setup with grid dimensions.")

        # 公式展開：p_new = p_old + beta * (y - p_old)
        #         p_new = p_old * (1 - beta) + beta * y

        # 對於非 BMU (y=0): p_new = p_old * (1 - beta)
        # 對於是 BMU (y=1): p_new = p_old * (1 - beta) + beta

        self.win_prob *= 1.0 - self.beta
        self.win_prob[bmu_idx] += self.beta


class SOM:
    def __init__(
        self,
        x: int,
        y: int,
        feature_dim: int,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        random_seed: int | None = None,
        conscience: ConscienceMechanism | None = None,
    ) -> None:
        """
        初始化 Self-Organizing Map (SOM) 模型。

        Parameters
        ----------
        x : int
            SOM 網格的寬度。
        y : int
            SOM 網格的高度。
        feature_dim : int
            輸入向量的維度 (Feature dimension)。
        sigma : float, optional
            Neighborhood function 的初始半徑，預設為 1.0。
        learning_rate : float, optional
            初始學習率，預設為 0.5。
        random_seed : int | None, optional
            隨機種子，用於初始化 `self.rng`，確保實驗結果可重現。預設為 None。
        conscience : ConscienceMechanism | None, optional
            良心機構模組實例。
        """
        self.x = x
        self.y = y
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_seed)

        self.conscience = conscience
        if self.conscience is not None:
            self.conscience.setup(x, y)

        # Neuron 網格座標
        self._grid_indices = np.indices((x, y))

        # Neuron 權重
        self.weights: NDArray[np.float64] = np.empty((x, y, feature_dim), dtype=np.float64)

    def initialize_weights(self, features: NDArray[np.float64]) -> None:
        """
        根據輸入資料的數值範圍 (Min-Max) 初始化權重。

        Parameters
        ----------
        features : NDArray[np.float64]
            用於計算初始化範圍的訓練資料集。Shape 為 (n_samples, input_len)。
        """
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)

        random_base = self.rng.random((self.x, self.y, self.feature_dim))

        # Broadcasting 自動處理維度
        self.weights = min_val + (max_val - min_val) * random_base

    def find_bmu(self, sample: NDArray[np.float64]) -> tuple[np.int64, ...]:
        """
        尋找給定樣本的最佳匹配單元 (Best Matching Unit, BMU)。

        Parameters
        ----------
        sample : NDArray[np.float64]
            單個輸入樣本向量。Shape 為 (input_len,)。

        Returns
        -------
        tuple[np.int64, ...]
            BMU 在網格中的座標索引 (x, y)。
        """

        # 1. 計算 Euclidean Distance
        distances: NDArray[np.float64] = np.linalg.norm(self.weights - sample, axis=-1)

        # 2. Conscience Mechanism 介入
        if self.conscience is not None:
            distances -= self.conscience.compute_bias()

        # 3. 找出 BMU 的網格座標
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def update_weights(
        self,
        sample: NDArray[np.float64],
        bmu_idx: tuple[np.int64, ...],
        iter_count: int,
        max_steps: int,
    ) -> None:
        """
        根據當前樣本與 BMU 更新 SOM 網格權重。

        Parameters
        ----------
        sample : NDArray[np.float64]
            當前的訓練樣本。
        bmu_idx : tuple[np.int64, ...]
            BMU 的座標。
        iter_count : int
            當前迭代次數 (從 1 開始)。
        max_steps : int
            總迭代次數。
        """

        # 1. 計算衰減後的 sigma 和 learning_rate
        decay = np.exp(-iter_count / max_steps)
        sigma = self.sigma * decay
        lr = self.learning_rate * decay

        # 2. 計算每個網格點到 BMU 的距離平方
        bmu_x, bmu_y = bmu_idx
        dist_sq = (self._grid_indices[0] - bmu_x) ** 2 + (self._grid_indices[1] - bmu_y) ** 2

        # 3. 以 Neighborhood Function 修正
        h = np.exp(-dist_sq / (2 * sigma**2))

        # 4. 更新 Neuron 權重
        self.weights += lr * h[:, :, np.newaxis] * (sample - self.weights)

    def train_stepwise(
        self,
        features: NDArray[np.float64],
        *,
        max_steps: int,
        auto_init: bool = True,
    ) -> Generator[tuple[int, NDArray[np.float64]]]:
        """
        逐步訓練 SOM，並生成每一步的狀態。

        Parameters
        ----------
        features : NDArray[np.float64]
            訓練資料集。
        max_steps : int
            預計執行的總步數。
        auto_init : bool, optional
            是否在開始訓練前自動根據 features 的範圍初始化權重，預設為 True。

        Yields
        ------
        tuple[int, NDArray[np.float64]]
            產生器，每次回傳 (當前步數, 當前權重的複本)。
        """
        if auto_init:
            self.initialize_weights(features)

        yield 0, self.weights

        n_samples = features.shape[0]

        for i in range(1, max_steps + 1):
            idx = self.rng.integers(0, n_samples)
            sample = features[idx]

            bmu_idx = self.find_bmu(sample)
            self.update_weights(sample, bmu_idx, i, max_steps)

            if self.conscience is not None:
                self.conscience.update(bmu_idx)

            yield i, self.weights

    def train(self, features: NDArray[np.float64], *, max_steps: int) -> None:
        """
        執行完整的訓練流程。

        Parameters
        ----------
        features : NDArray[np.float64]
            訓練資料集。
        max_steps : int
            總迭代次數。
        """
        for _ in self.train_stepwise(features, max_steps=max_steps):
            pass
