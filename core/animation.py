from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import RegularPolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from .network import SOM


class BaseSOMAnimator(ABC):
    def __init__(
        self,
        som: SOM,
        features: NDArray[np.float64],
        max_steps: int,
        steps_per_frame: int = 5,
        on_update: Callable[[int], None] | None = None,
        *,
        fig: Figure,
        ax: Axes,
    ) -> None:
        """
        SOM 動畫器的基礎抽象類別。

        負責管理訓練迴圈、動畫計時器與共用的狀態，具體的繪圖邏輯由子類別實作。

        Parameters
        ----------
        som : SOM
            SOM 模型實例。
        features : NDArray[np.float64]
            訓練資料集。
        max_steps : int
            總訓練步數。
        steps_per_frame : int, optional
            每一幀動畫推進的訓練步數。
        on_update : Callable[[int], None] | None, optional
            每幀更新時的回呼函式，接收當前步數。
        fig : Figure
            Matplotlib Figure 物件。
        ax : Axes
            Matplotlib Axes 物件。
        """
        self.som = som
        self.features = features
        self.max_steps = max_steps
        self.steps_per_frame = steps_per_frame
        self.on_update = on_update
        self.fig = fig
        self.ax = ax

        # 建立訓練生成器
        self.training_generator = self.som.train_stepwise(features, max_steps=max_steps)

        # 儲存需要更新的 Artists
        self.artists: list[Artist] = []

        # 呼叫抽象方法進行初始化
        self._setup_plot()

    @abstractmethod
    def _setup_plot(self) -> None:
        """
        初始化圖表元素。
        """
        pass

    @abstractmethod
    def _update_plot(self, weights: NDArray[np.float64]) -> None:
        """
        更新圖表元素。

        Parameters
        ----------
        weights : NDArray[np.float64]
            當前的 SOM 權重。
        """
        pass

    def _update(self, frame: int) -> Sequence[Artist]:
        """
        FuncAnimation 的核心更新迴圈。

        Parameters
        ----------
        frame : int
            當前的幀數索引 (由 FuncAnimation 傳入)。

        Returns
        -------
        Sequence[Artist]
            需要重繪的 Artist 列表。
        """
        try:
            current_iter = 0
            current_weights = None

            # 推進 N 步
            for _ in range(self.steps_per_frame):
                current_iter, current_weights = next(self.training_generator)

            # 只有當拿到有效權重時才進行繪圖更新
            if current_weights is not None:
                self._update_plot(current_weights)

                if self.on_update:
                    self.on_update(current_iter)

        except StopIteration:
            # 訓練結束
            pass

        return self.artists

    def animate(self, interval: int = 30, blit: bool = True) -> FuncAnimation:
        """
        建立並回傳 Matplotlib 動畫物件。

        Parameters
        ----------
        interval : int, optional
            幀與幀之間的間隔 (毫秒)。
        blit : bool, optional
            是否使用 Blitting 優化繪圖效能，預設為 True。

        Returns
        -------
        FuncAnimation
            動畫物件實例。
        """
        return FuncAnimation(
            self.fig,
            self._update,
            frames=range(0, self.max_steps + 1, self.steps_per_frame),
            init_func=lambda: self.artists,
            interval=interval,
            blit=blit,
            repeat=False,
        )


class GridAnimator(BaseSOMAnimator):
    def __init__(
        self,
        som: SOM,
        features: NDArray[np.float64],
        labels: NDArray[np.int64] | None = None,
        max_steps: int = 2000,
        steps_per_frame: int = 5,
        watch_dims: tuple[int, int] = (0, 1),
        cmap: str = "Set1",
        on_update: Callable[[int], None] | None = None,
        *,
        fig: Figure,
        ax: Axes,
    ) -> None:
        """
        網格動畫器

        Parameters
        ----------
        som : SOM
            SOM 模型實例。
        features : NDArray[np.float64]
            訓練資料集的特徵。
        labels : NDArray[np.int64] | None, optional
            訓練資料集的標籤，用於資料點著色。
        max_steps : int
            總訓練步數。
        steps_per_frame : int, optional
            每一幀動畫推進的訓練步數。
        watch_dims : tuple[int, int], optional
            要觀察的兩個特徵維度索引。
        cmap : str, optional
            Matplotlib 的 colormap 名稱，僅在 labels 不為 None 時生效。
        on_update : Callable[[int], None] | None, optional
            每幀更新時的回呼函式，接收當前步數。
        fig : Figure
            Matplotlib Figure 物件。
        ax : Axes
            Matplotlib Axes 物件。
        """
        self.watch_dims = watch_dims
        self.labels = labels
        self.cmap = cmap
        super().__init__(som, features, max_steps, steps_per_frame, on_update, fig=fig, ax=ax)

    def _setup_plot(self) -> None:
        d_x, d_y = self.watch_dims
        feat_x = self.features[:, d_x]
        feat_y = self.features[:, d_y]

        # 邊界
        margin_x = (np.max(feat_x) - np.min(feat_x)) * 0.05
        margin_y = (np.max(feat_y) - np.min(feat_y)) * 0.05
        self.ax.set_xlim(np.min(feat_x) - margin_x, np.max(feat_x) + margin_x)
        self.ax.set_ylim(np.min(feat_y) - margin_y, np.max(feat_y) + margin_y)

        # 顏色
        c_param = "lightblue"
        cmap_param = None
        if self.labels is not None:
            c_param = self.labels
            cmap_param = self.cmap

        # 資料點
        self.ax.scatter(
            feat_x,
            feat_y,
            s=10,
            alpha=0.6 if self.labels is not None else 0.2,  # 有 label 時透明度調高一點以便觀察
            c=c_param,
            cmap=cmap_param,
        )

        # 動態 Artists (Grid Lines + Neurons)
        self.scat_neurons = self.ax.scatter([], [], s=20, c="orange", edgecolors="black", zorder=10)
        self.lines_grid: list[Line2D] = []
        for _ in range(self.som.x + self.som.y):
            (line,) = self.ax.plot([], [], "w-", alpha=0.2, lw=1, zorder=5)
            self.lines_grid.append(line)

        # 5. 註冊 Artists
        self.artists = [self.scat_neurons, *self.lines_grid]

    def _update_plot(self, weights: NDArray[np.float64]) -> None:
        d_x, d_y = self.watch_dims

        # 更新 Neurons
        flat_weights = weights.reshape(-1, weights.shape[-1])
        self.scat_neurons.set_offsets(flat_weights[:, [d_x, d_y]])

        # 更新 Grid Lines
        line_idx = 0
        # 橫向線
        for i in range(self.som.x):
            self.lines_grid[line_idx].set_data(weights[i, :, d_x], weights[i, :, d_y])
            line_idx += 1
        # 縱向線
        for j in range(self.som.y):
            self.lines_grid[line_idx].set_data(weights[:, j, d_x], weights[:, j, d_y])
            line_idx += 1


class HexagonAnimator(BaseSOMAnimator):
    """
    六邊形動畫器
    """

    def _setup_plot(self) -> None:
        self.cmap = plt.get_cmap("viridis")
        self.norm = Normalize(vmin=0, vmax=1)

        offsets = []
        # 產生六邊形中心座標
        for y in range(self.som.y):
            for x in range(self.som.x):
                x_pos = x + 0.5 * (y % 2)
                y_pos = y * (np.sqrt(3) / 2)
                offsets.append([x_pos, y_pos])

        # 初始化 Collection
        self.collection = RegularPolyCollection(
            numsides=6,
            sizes=[200],  # 暫時值，會被 Canvas 覆蓋
            offsets=offsets,
            transOffset=self.ax.transData,
            cmap=self.cmap,
            norm=self.norm,
            edgecolors="white",
            linewidths=0.5,
        )

        self.ax.add_collection(self.collection)

        # 座標軸邊界
        self.ax.set_xlim(-1, self.som.x + 1)
        self.ax.set_ylim(-1, self.som.y * (np.sqrt(3) / 2) + 1)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # 加入 Colorbar
        self.fig.colorbar(self.collection, ax=self.ax, label="avg. neighbor distance (norm.)")

        # 註冊 Artists
        self.artists = [self.collection]

    def _calculate_umatrix(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        計算 U-Matrix。
        每個 Cell 計算與周圍 6 個鄰居的平均距離。

        Parameters
        ----------
        weights : NDArray[np.float64]
            當前的 SOM 權重矩陣，形狀為 (x, y, features)。

        Returns
        -------
        NDArray[np.float64]
            標準化 (0-1) 後的 U-Matrix，形狀為 (x, y)。
        """
        x, y, _ = weights.shape

        # 垂直距離 (上、下)
        d_vertical = np.linalg.norm(weights[1:] - weights[:-1], axis=-1)
        # 水平距離 (左、右)
        d_horizontal = np.linalg.norm(weights[:, 1:] - weights[:, :-1], axis=-1)
        # 對角線距離 (左上、右下)
        # 註：這裡為了計算速度，犧牲了視覺上的鄰居關係，然而拓樸結構仍是等價的
        d_diagonal = np.linalg.norm(weights[1:, 1:] - weights[:-1, :-1], axis=-1)

        dist_sum = np.zeros((x, y))
        neighbor_counts = np.zeros((x, y))

        # --- 累加垂直距離 (2個鄰居) ---
        dist_sum[:-1, :] += d_vertical
        neighbor_counts[:-1, :] += 1
        dist_sum[1:, :] += d_vertical
        neighbor_counts[1:, :] += 1

        # --- 累加水平距離 (2個鄰居) ---
        dist_sum[:, :-1] += d_horizontal
        neighbor_counts[:, :-1] += 1
        dist_sum[:, 1:] += d_horizontal
        neighbor_counts[:, 1:] += 1

        # --- 累加對角線距離 (2個鄰居) ---
        # 處理 "右下" 鄰居 (i+1, j+1)
        dist_sum[:-1, :-1] += d_diagonal
        neighbor_counts[:-1, :-1] += 1

        # 處理 "左上" 鄰居 (i-1, j-1)
        dist_sum[1:, 1:] += d_diagonal
        neighbor_counts[1:, 1:] += 1

        # 計算平均
        umatrix = dist_sum / (neighbor_counts + 1e-8)

        # Normalize 0-1
        _min, _max = umatrix.min(), umatrix.max()
        if _max > _min:
            umatrix = (umatrix - _min) / (_max - _min)

        return umatrix

    def _update_plot(self, weights: NDArray[np.float64]) -> None:
        """
        計算 U-Matrix 並更新顏色。

        Parameters
        ----------
        weights : NDArray[np.float64]
            當前的 SOM 權重。
        """
        umatrix = self._calculate_umatrix(weights)
        self.collection.set_array(umatrix.flatten())
