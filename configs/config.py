"""
config.py
---------
Load cấu hình từ config.yaml thành các dataclass có kiểu dữ liệu rõ ràng.
Toàn bộ pipeline (data/, scripts/) chỉ import từ module này — không đọc
config.yaml trực tiếp ở nơi khác, tránh rải rác logic parse cấu hình.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

def find_project_root(marker: str = "pyproject.toml", start: Path = None) -> Path:
    """
    Tìm project root bằng cách đi ngược lên cây thư mục cho tới khi gặp
    file mốc (mặc định pyproject.toml).

    KHÔNG dùng `Path(__file__).resolve().parent` trực tiếp làm PROJECT_ROOT,
    vì nếu config.py bị di chuyển vào một thư mục con (ví dụ configs/),
    parent của nó sẽ là configs/ chứ không phải root thật của project —
    khiến mọi path tương đối (SAMPLES_DIR, cache_dir, ...) bị lệch một cấp
    (vd. ".../configs/samples/official" thay vì ".../samples/official").
    """
    current = (start or Path(__file__).resolve().parent)
    for candidate in [current, *current.parents]:
        if (candidate / marker).exists():
            return candidate
    # Không tìm thấy marker: fallback về thư mục chứa file này (giữ hành vi cũ)
    return current


PROJECT_ROOT = find_project_root()
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def resolve_project_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    """
    Resolve `value` LUÔN tương đối so với `base` (mặc định PROJECT_ROOT),
    kể cả khi `value` bị ghi nhầm có dấu "/" ở đầu (ví dụ "/samples/official"
    trong config.yaml).

    Lý do cần hàm này: `Path("/a/b") / "/c/d"` trong pathlib sẽ TRẢ VỀ
    "/c/d" (bỏ hẳn phần bên trái) vì toán tử "/" coi vế phải bắt đầu bằng
    "/" là một đường dẫn tuyệt đối độc lập. Nếu không chặn trường hợp này,
    một giá trị cấu hình như "/samples/official" sẽ âm thầm biến thành
    thư mục gốc ổ đĩa "/samples/official" thay vì "<project>/samples/official",
    gây lỗi PermissionError khi mkdir.
    """
    cleaned = str(value).lstrip("/\\")
    return (base / cleaned).resolve()


@dataclass(frozen=True)
class DatasetConfig:
    repository_id: str
    default_split: str


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int
    format: str
    layout: str


@dataclass(frozen=True)
class MouthRoiConfig:
    size: int
    landmarks: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TextConfig:
    keep_diacritics: bool


@dataclass(frozen=True)
class ModelConfig:
    repository_id: str
    revision: str
    cache_dir: str
    device: str
    dtype: str


@dataclass(frozen=True)
class TokenizerConfig:
    model_path: str
    units_path: str


@dataclass(frozen=True)
class PathsConfig:
    samples_dir: Path


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    audio: AudioConfig
    mouth_roi: MouthRoiConfig
    text: TextConfig
    model: ModelConfig
    tokenizer: TokenizerConfig
    paths: PathsConfig


def load_config(path: Path = CONFIG_PATH) -> Config:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        dataset=DatasetConfig(**raw["dataset"]),
        audio=AudioConfig(**raw["audio"]),
        mouth_roi=MouthRoiConfig(**raw["mouth_roi"]),
        text=TextConfig(**raw["text"]),
        model=ModelConfig(**raw["model"]),
        tokenizer=TokenizerConfig(**raw["tokenizer"]),
        paths=PathsConfig(samples_dir=resolve_project_path(raw["paths"]["samples_dir"])),
    )


# --- Instance dùng chung cho toàn bộ pipeline ---
CFG = load_config()

# --- Alias phẳng để tương thích ngược với code cũ (data/, scripts/) ---
DATASET_NAME = CFG.dataset.repository_id
DEFAULT_SPLIT = CFG.dataset.default_split

TARGET_SAMPLE_RATE = CFG.audio.target_sample_rate
AUDIO_FORMAT = CFG.audio.format
AUDIO_LAYOUT = CFG.audio.layout

MOUTH_ROI_SIZE = CFG.mouth_roi.size
MOUTH_LANDMARKS = CFG.mouth_roi.landmarks

KEEP_DIACRITICS = CFG.text.keep_diacritics

SAMPLES_DIR = CFG.paths.samples_dir

MODEL_REPOSITORY_ID = CFG.model.repository_id
MODEL_REVISION = CFG.model.revision
MODEL_CACHE_DIR = CFG.model.cache_dir
MODEL_DEVICE = CFG.model.device
MODEL_DTYPE = CFG.model.dtype

TOKENIZER_MODEL_PATH = CFG.tokenizer.model_path
TOKENIZER_UNITS_PATH = CFG.tokenizer.units_path