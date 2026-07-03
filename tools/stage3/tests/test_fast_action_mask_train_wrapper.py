from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.action_mask import ActionMask  # noqa: E402
from tools.stage3.train_HOPE_sac_fast_action_mask import enable_fast_action_mask_by_default  # noqa: E402


class FastActionMaskTrainWrapperTests(unittest.TestCase):
    def test_wrapper_makes_action_mask_fast_by_default_but_respects_explicit_false(self) -> None:
        original_init = ActionMask.__init__
        original_installed = getattr(ActionMask, "_stage3_fast_default_installed", None)
        original_saved_init = getattr(ActionMask, "_stage3_original_init", None)

        try:
            enable_fast_action_mask_by_default()

            self.assertTrue(ActionMask().fast_get_steps)
            self.assertFalse(ActionMask(fast_get_steps=False).fast_get_steps)
        finally:
            ActionMask.__init__ = original_init
            if original_installed is None:
                if hasattr(ActionMask, "_stage3_fast_default_installed"):
                    delattr(ActionMask, "_stage3_fast_default_installed")
            else:
                ActionMask._stage3_fast_default_installed = original_installed

            if original_saved_init is None:
                if hasattr(ActionMask, "_stage3_original_init"):
                    delattr(ActionMask, "_stage3_original_init")
            else:
                ActionMask._stage3_original_init = original_saved_init


if __name__ == "__main__":
    unittest.main()
