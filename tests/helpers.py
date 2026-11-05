import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
