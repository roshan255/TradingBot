from pathlib import Path
import json
import os
import uuid

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = Path(os.environ.get("BFP_STORAGE_DIR", PROJECT_ROOT)).expanduser().resolve()
ROOT_DIR = STORAGE_ROOT
DATA_DIR = Path(os.environ.get("BFP_DATA_DIR", STORAGE_ROOT / "Data")).expanduser().resolve()
LOCAL_DIR = Path(os.environ.get("BFP_LOCAL_DIR", STORAGE_ROOT / "local")).expanduser().resolve()
MPLCONFIG_DIR = Path(os.environ.get("BFP_MPLCONFIGDIR", STORAGE_ROOT / ".mplcache")).expanduser().resolve()

for directory in [DATA_DIR, LOCAL_DIR, MPLCONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def resolve_runtime_path(path_like, base_dir: Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base_dir or STORAGE_ROOT) / path).resolve()


def get_symbol_dir(symbol: str) -> Path:
    symbol_dir = DATA_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    return symbol_dir


def get_symbol_file(symbol: str, file_name: str) -> Path:
    return get_symbol_dir(symbol) / file_name


def _temp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    temp_path = _temp_path(path)
    temp_path.write_text(text, encoding=encoding)
    os.replace(temp_path, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2), encoding="utf-8")


def dump_joblib_atomic(path: Path, obj) -> None:
    temp_path = _temp_path(path)
    joblib.dump(obj, temp_path)
    os.replace(temp_path, path)


def write_csv_atomic(path: Path, frame) -> None:
    temp_path = _temp_path(path)
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, path)

