import os
from dataclasses import dataclass
from functools import lru_cache
from tkinter import Event, Misc, filedialog

import matplotlib.pyplot as plt
import numpy as np
import pywinstyles
import ttkbootstrap as tb
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import RegularPolyCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray
from ttkbootstrap.constants import BOTH, INFO, LEFT, NSEW, PRIMARY, X

from core.animation import BaseSOMAnimator, GridAnimator, HexagonAnimator
from core.network import SOM, ConscienceMechanism
from core.util import read_data, resource_path


@dataclass
class Dataset:
    """
    資料集 Data Class，用於儲存當前載入的資料及其相關資訊。

    Parameters
    ----------
    data : tuple[NDArray, NDArray] | None, optional
        包含特徵與標籤的資料 Tuple。
    filepath : str | None, optional
        資料檔案的完整路徑。
    filename : str | None, optional
        資料檔案名稱。
    feature_dim : int | None, optional
        特徵維度。
    label_dim : int | None, optional
        標籤維度。
    """

    data: tuple[NDArray, NDArray] | None = None
    filepath: str | None = None
    filename: str | None = None
    feature_dim: int | None = None
    label_dim: int | None = None

    def __repr__(self) -> str:
        return (
            f"Dataset(\n"
            f"  name={self.filename},\n"
            f"  filepath={self.filepath},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  label_dim={self.label_dim}\n"
            f")"
        )

    @classmethod
    def load(cls, filepath: str) -> "Dataset":
        """
        從指定路徑載入資料並建立 Dataset 實例。

        Parameters
        ----------
        filepath : str
            資料檔案的路徑。

        Returns
        -------
        Dataset
            初始化後的 Dataset 物件。
        """
        data = read_data(filepath)
        return cls(
            filepath=filepath,
            filename=os.path.basename(filepath),
            data=data,
            feature_dim=data[0].shape[1],
            label_dim=data[1].shape[1],
        )


@dataclass
class HyperParams:
    """
    超參數 Data Class，封裝所有 Tkinter 變數以供介面綁定。

    Parameters
    ----------
    seed : tb.IntVar
        隨機種子。
    x : tb.IntVar
        SOM 網格寬度。
    y : tb.IntVar
        SOM 網格高度。
    sigma : tb.DoubleVar
        鄰域半徑參數。
    learning_rate : tb.DoubleVar
        學習率。
    beta : tb.DoubleVar
        Conscience 機制的 Beta 參數。
    c_factor : tb.DoubleVar
        Conscience 機制的 C Factor 參數。
    """

    seed: tb.IntVar
    x: tb.IntVar
    y: tb.IntVar
    sigma: tb.DoubleVar
    learning_rate: tb.DoubleVar
    beta: tb.DoubleVar
    c_factor: tb.DoubleVar

    def __repr__(self) -> str:
        return (
            f"HyperParams(\n"
            f"  seed={self.seed.get()},\n"
            f"  x={self.x.get()},\n"
            f"  y={self.y.get()},\n"
            f"  sigma={self.sigma.get()},\n"
            f"  learning_rate={self.learning_rate.get()}\n"
            f"  beta={self.beta.get()},\n"
            f"  c_factor={self.c_factor.get()}\n"
            f")"
        )


@dataclass
class ConfigParams:
    """
    配置 Data Class。

    Parameters
    ----------
    max_steps : tb.IntVar
        最大訓練步數。
    steps_per_frame: tb.IntVar
        每一幀動畫推進的訓練步數。
    interval: tb.IntVar
        幀與幀之間的間隔 (毫秒)。
    """

    max_steps: tb.IntVar
    steps_per_frame: tb.IntVar
    interval: tb.IntVar

    def __repr__(self) -> str:
        return (
            f"StopCondition(\n"
            f"  max_steps={self.max_steps.get()}\n"
            f"  steps_per_frame={self.steps_per_frame.get()}\n"
            f"  interval={self.interval.get()}\n"
            f")"
        )


@dataclass
class AppState:
    """
    應用程式全域狀態 Data Class。

    Parameters
    ----------
    dataset : Dataset
        當前資料集狀態。
    hyperparams : HyperParams
        當前超參數狀態。
    configparams : ConfigParams
        當前配置狀態。
    """

    dataset: Dataset
    hyperparams: HyperParams
    configparams: ConfigParams

    def __repr__(self) -> str:
        return (
            f"AppState(\n"
            f"  dataset={self.dataset},\n"
            f"  hyperparams={self.hyperparams},\n"
            f"  configparams={self.configparams}\n"
            f")"
        )


@lru_cache
def get_app_state() -> AppState:
    """
    取得全域唯一的應用程式狀態 (Singleton 模式)。
    使用 lru_cache 確保只初始化一次。

    Returns
    -------
    AppState
        包含所有狀態變數的 AppState 實例。
    """
    dataset = Dataset()
    hyperparams = HyperParams(
        seed=tb.IntVar(value=42),
        sigma=tb.DoubleVar(value=3.0),
        x=tb.IntVar(value=20),
        y=tb.IntVar(value=20),
        learning_rate=tb.DoubleVar(value=1.0),
        beta=tb.DoubleVar(value=0.1),
        c_factor=tb.DoubleVar(value=0.5),
    )
    configparams = ConfigParams(
        max_steps=tb.IntVar(value=2000),
        steps_per_frame=tb.IntVar(value=5),
        interval=tb.IntVar(value=30),
    )
    return AppState(dataset=dataset, hyperparams=hyperparams, configparams=configparams)


def create_input_field(master: Misc, label_text: str, variable: tb.Variable) -> tb.Frame:
    """
    建立包含 Label 與 Entry 的輸入區塊。

    Parameters
    ----------
    master : Misc
        父容器元件。
    label_text : str
        標籤顯示文字。
    variable : tb.Variable
        綁定的 Tkinter 變數。

    Returns
    -------
    tb.Frame
        包含輸入元件的 Frame 容器。
    """
    frame = tb.Frame(master)
    frame.pack(fill=X, pady=5)
    tb.Label(frame, text=label_text, width=15).pack(side=LEFT, padx=5)
    tb.Entry(frame, textvariable=variable).pack(side=LEFT, fill=X, expand=True, padx=5)
    return frame


class SOMCanvas(tb.Frame):
    """
    繪圖區域元件。
    """

    def __init__(self, master: Misc, figsize: tuple[int, int] = (6, 6)) -> None:
        """
        初始化 SOMCanvas。

        Parameters
        ----------
        master : Misc
            父容器元件。
        figsize : tuple[int, int], optional
            圖表的初始尺寸。
        """
        super().__init__(master)

        # 初始化 Matplotlib 物件
        self._fig: Figure = plt.figure(figsize=figsize)
        self._fig.set_facecolor("#2b2b2b")
        self._ax: Axes | None = None
        self._resize_cid: int | None = None

        # 嵌入 Tkinter
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        # 繪製初始空白狀態
        self.reset_axes()

    @property
    def figure(self) -> Figure:
        """
        取得底層的 Figure 物件。

        Returns
        -------
        Figure
            Matplotlib 的 Figure 物件。
        """
        return self._fig

    def reset_axes(self) -> tuple[Figure, Axes]:
        """
        清除當前圖表並重置 Axes，準備進行新的繪圖。
        同時會解除舊的 resize 事件綁定。

        Returns
        -------
        tuple[Figure, Axes]
            回傳清理後的 Figure 物件與新建立的 Axes 物件。
        """
        # 解除舊的事件綁定
        if self._resize_cid is not None:
            self._fig.canvas.mpl_disconnect(self._resize_cid)
            self._resize_cid = None

        self._fig.clear()
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.axis("off")
        return self._fig, self._ax

    def draw_canvas(self) -> None:
        """
        觸發 Canvas 的重繪操作 (draw_idle 或 draw)。
        """
        self._canvas.draw()

    def bind_hex_scaling(self, collection: RegularPolyCollection) -> None:
        """
        綁定視窗縮放事件，以動態調整六邊形的大小，保持視覺比例。

        Parameters
        ----------
        collection : RegularPolyCollection
            需要調整大小的 Matplotlib Collection 物件。
        """

        def _update_sizes(event=None):
            if not self._ax:
                return

            bbox = self._ax.get_window_extent().transformed(self._fig.dpi_scale_trans.inverted())
            width_inches = bbox.width
            xlim = self._ax.get_xlim()
            data_width = xlim[1] - xlim[0]

            if data_width == 0:
                return

            scale_point_per_unit = (width_inches / data_width) * 72
            radius_in_points = scale_point_per_unit * 1.0
            area_in_points_squared = (radius_in_points * 0.95) ** 2
            collection.set_sizes([area_in_points_squared])

            if event:
                self._fig.canvas.draw_idle()

        _update_sizes()

        self._resize_cid = self._fig.canvas.mpl_connect("resize_event", _update_sizes)


class DatasetFrame(tb.Labelframe):
    """Dataset 區塊"""

    def __init__(self, master: Misc) -> None:
        """
        初始化 DatasetFrame。

        Parameters
        ----------
        master : Misc
            父容器元件。
        """
        super().__init__(master, text="⏹ Dataset", style=INFO)

        # variable
        self.dataset_var = tb.StringVar(value="Current Dataset: ...")

        # load button
        tb.Button(self, text="Load Dataset", style=PRIMARY, width=20, command=self.load_dataset).pack(padx=5, pady=5)

        # dataset label
        tb.Label(self, textvariable=self.dataset_var).pack(padx=5, pady=5)

    def load_dataset(self) -> None:
        """
        開啟檔案對話框載入資料集，並更新全域狀態與觸發事件。
        """
        filepath = filedialog.askopenfilename(title="選擇資料集", filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")])
        if filepath:
            get_app_state().dataset = Dataset.load(filepath)
            self.dataset_var.set(f"Current Dataset: {get_app_state().dataset.filename}")
            self.event_generate("<<DatasetLoaded>>", when="tail")  #! 發送 DatasetLoaded 事件


class HyperParamFrame(tb.Labelframe):
    """Hyper Parameter 區塊"""

    def __init__(self, master: Misc) -> None:
        """
        初始化 HyperParamFrame。

        Parameters
        ----------
        master : Misc
            父容器元件。
        """
        super().__init__(master, text="⏹ Hyper Parameter", style=INFO)

        hyperparams = get_app_state().hyperparams

        seed_frame = create_input_field(self, "Random Seed", hyperparams.seed)
        tb.Button(seed_frame, text="🎲", width=3, style=PRIMARY, command=self.roll_seed).pack(side=LEFT, padx=5)

        create_input_field(self, "Learning Rate", hyperparams.learning_rate)
        create_input_field(self, "Sigma", hyperparams.sigma)
        create_input_field(self, "X", hyperparams.x)
        create_input_field(self, "Y", hyperparams.y)
        create_input_field(self, "Beta", hyperparams.beta)
        create_input_field(self, "C Factor", hyperparams.c_factor)

    def roll_seed(self) -> None:
        """
        隨機產生 Seed 並更新至 State。
        """
        get_app_state().hyperparams.seed.set(np.random.randint(0, 9999))


class ConfigParamFrame(tb.Labelframe):
    """Config Parameter 區塊"""

    def __init__(self, master: Misc) -> None:
        """
        初始化 ConfigParamFrame

        Parameters
        ----------
        master : Misc
            父容器元件。
        """
        super().__init__(master, text="⏹ Config Parameter", style=INFO)

        configparams = get_app_state().configparams

        create_input_field(self, "Max Steps", configparams.max_steps)
        create_input_field(self, "Steps Per Frame", configparams.steps_per_frame)
        create_input_field(self, "Interval", configparams.interval)


class TrainFrame(tb.Labelframe):
    """Train 區塊"""

    def __init__(self, master: Misc) -> None:
        """
        初始化 TrainFrame。

        Parameters
        ----------
        master : Misc
            父容器元件。
        """
        super().__init__(master, text="⏹ Train", style=INFO)

        # train button
        train_btn = tb.Button(self, text="Train", style=PRIMARY, width=20, command=self.train_model)
        train_btn.pack(padx=10, pady=10)

    def train_model(self) -> None:
        """
        觸發訓練開始事件。
        """
        self.event_generate("<<Training>>", when="tail")


class PlotFrame(tb.Frame):
    """Plot 區塊"""

    def __init__(self, master: Misc) -> None:
        """
        初始化 PlotFrame。

        Parameters
        ----------
        master : Misc
            父容器元件。
        """
        super().__init__(master)

        self.animation: FuncAnimation | None = None
        self.animator: BaseSOMAnimator | None = None

        # grid layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 1. Info Bar
        self.info_frame = tb.Labelframe(self, text="⏹ Info", style=INFO)
        self.info_frame.grid(row=0, column=0, sticky=NSEW, padx=10, pady=10)

        self.iter_label = tb.Label(self.info_frame, text="Step 0")
        self.iter_label.pack(side=LEFT, padx=10, pady=10)

        # 2. Train Plot Area
        self.train_plot_frame = tb.Labelframe(self, text="⏹ Visualization", style=INFO)
        self.train_plot_frame.grid(row=1, column=0, sticky=NSEW, padx=10, pady=10)

        self.som_canvas = SOMCanvas(self.train_plot_frame)
        self.som_canvas.pack(expand=True)

        # event
        self.bind_all("<<Training>>", self.on_training)

    def update_steps_display(self, current_steps: int) -> None:
        """
        更新介面上的當前步數顯示。

        Parameters
        ----------
        current_steps : int
            當前的訓練步數。
        """
        self.iter_label.config(text=f"Step {current_steps}")

    def on_training(self, event: Event) -> None:
        """
        處理開始訓練的事件 (<<Training>>)。

        此方法執行以下步驟：
        1. 停止並清除舊的 Animation。
        2. 重置 SOMCanvas 取得新的 Figure 與 Axes。
        3. 初始化 SOM 模型。
        4. 根據 Feature 維度選擇 Animator：
            - 維度為 2：使用 ProjectionAnimator。
            - 其他維度：使用 HexagonAnimator。
        5. 啟動動畫並重繪 Canvas。

        Parameters
        ----------
        event : Event
            觸發此方法的 Tkinter Event 物件。
        """
        # 1. 停止舊動畫
        if self.animation and self.animation.event_source:
            self.animation.event_source.stop()
        self.animation = None

        # 2. 取得乾淨的 fig, ax
        fig, ax = self.som_canvas.reset_axes()

        # 3. 準備數據
        app_state = get_app_state()
        if app_state.dataset.data is None:
            return

        features, labels = app_state.dataset.data
        feature_dim: int = app_state.dataset.feature_dim  # type: ignore

        # 4. 建立模型
        som = SOM(
            x=app_state.hyperparams.x.get(),
            y=app_state.hyperparams.y.get(),
            feature_dim=feature_dim,
            sigma=app_state.hyperparams.sigma.get(),
            learning_rate=app_state.hyperparams.learning_rate.get(),
            random_seed=app_state.hyperparams.seed.get(),
            conscience=ConscienceMechanism(
                beta=app_state.hyperparams.beta.get(),
                c_factor=app_state.hyperparams.c_factor.get(),
            ),
        )

        # 5. 建立動畫 (根據 feature_dim 選擇 Animator)
        common_kwargs = {
            "som": som,
            "features": features,
            "steps_per_frame": app_state.configparams.steps_per_frame.get(),
            "max_steps": app_state.configparams.max_steps.get(),
            "on_update": self.update_steps_display,
            "fig": fig,
            "ax": ax,
        }

        if feature_dim == 2:
            self.animator = GridAnimator(**common_kwargs, labels=labels, watch_dims=(0, 1))
        else:
            self.animator = HexagonAnimator(**common_kwargs)

        # 若為 HexagonAnimator，需綁定縮放事件以維持六邊形比例
        if isinstance(self.animator, HexagonAnimator):
            self.som_canvas.bind_hex_scaling(self.animator.collection)

        # 6. 開始動畫
        self.animation = self.animator.animate(interval=app_state.configparams.interval.get())

        # 7. 重繪畫布
        self.som_canvas.draw_canvas()


class MainApp(tb.Window):
    """主應用程式視窗類別"""

    def __init__(self) -> None:
        """
        初始化主應用程式視窗與佈局。
        """
        super().__init__(themename="darkly")

        self.title("SOM")
        self.geometry("1200x900")
        self.iconbitmap(resource_path("resources/icon/SOM.ico"))
        pywinstyles.apply_style(self, "dark")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        left_frame = tb.Frame(self)
        left_frame.grid(row=0, column=0, sticky=NSEW, padx=15, pady=15)
        DatasetFrame(left_frame).pack(fill=X, pady=10)
        HyperParamFrame(left_frame).pack(fill=X, pady=10)
        ConfigParamFrame(left_frame).pack(fill=X, pady=10)
        TrainFrame(left_frame).pack(fill=X, pady=10)

        right_frame = tb.Frame(self)
        right_frame.grid(row=0, column=1, sticky=NSEW, padx=15, pady=15)
        PlotFrame(right_frame).pack(fill=BOTH, expand=True)

    def exit(self):
        """
        關閉應用程式。
        """
        self.quit()
        self.destroy()


if __name__ == "__main__":
    plt.style.use("dark_background")
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", app.exit)
    app.mainloop()
