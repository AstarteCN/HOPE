from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TRAIN_SCRIPT = SRC_ROOT / "train" / "train_HOPE_sac.py"


def enable_fast_action_mask_by_default() -> None:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from model.action_mask import ActionMask  # noqa: PLC0415

    if getattr(ActionMask, "_stage3_fast_default_installed", False):
        return

    original_init = ActionMask.__init__

    def fast_default_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("fast_get_steps", True)
        original_init(self, *args, **kwargs)

    ActionMask._stage3_original_init = original_init
    ActionMask.__init__ = fast_default_init
    ActionMask._stage3_fast_default_installed = True


def main() -> int:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Original HOPE SAC training script not found: {TRAIN_SCRIPT}")

    enable_fast_action_mask_by_default()
    sys.argv[0] = str(TRAIN_SCRIPT)
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
