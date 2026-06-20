from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs import ACTOR_CONFIGS, CRITIC_CONFIGS, N_DISCRETE_ACTION  # noqa: E402
from model.agent.sac_agent import SACAgent  # noqa: E402
from model.network import MultiObsEmbedding, SACCriticAdapter  # noqa: E402
from model.state_norm import DEFAULT_UPDATE_MODAL, StateNorm  # noqa: E402


def actor_config() -> dict:
    config = deepcopy(ACTOR_CONFIGS)
    config.update(
        {
            "n_modal": 3,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": (2, 64, 64),
            "action_mask_shape": N_DISCRETE_ACTION,
        }
    )
    return config


def critic_config() -> dict:
    config = deepcopy(CRITIC_CONFIGS)
    config.update(
        {
            "n_modal": 4,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": (2, 64, 64),
            "action_mask_shape": N_DISCRETE_ACTION,
        }
    )
    return config


class OGMNetworkTests(unittest.TestCase):
    def test_state_norm_knows_ogm_is_not_running_normalized(self) -> None:
        self.assertIn("ogm", DEFAULT_UPDATE_MODAL)
        self.assertFalse(DEFAULT_UPDATE_MODAL["ogm"])
        norm = StateNorm({"target": (5,), "action_mask": (N_DISCRETE_ACTION,), "ogm": (2, 64, 64)})
        obs = {
            "target": np.zeros((5,), dtype=np.float32),
            "action_mask": np.ones((N_DISCRETE_ACTION,), dtype=np.float32),
            "ogm": np.ones((2, 64, 64), dtype=np.float32),
        }
        normalized = norm.state_norm(obs.copy(), update=True)
        np.testing.assert_array_equal(normalized["ogm"], obs["ogm"])

    def test_actor_accepts_target_action_mask_and_ogm_without_lidar_or_img(self) -> None:
        net = MultiObsEmbedding(actor_config())
        obs = {
            "target": torch.zeros((1, 5), dtype=torch.float32),
            "action_mask": torch.ones((1, N_DISCRETE_ACTION), dtype=torch.float32),
            "ogm": torch.zeros((1, 2, 64, 64), dtype=torch.float32),
        }
        output = net(obs)
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_critic_accepts_target_action_mask_ogm_and_action(self) -> None:
        critic = SACCriticAdapter(critic_config())
        obs = {
            "target": torch.zeros((1, 5), dtype=torch.float32),
            "action_mask": torch.ones((1, N_DISCRETE_ACTION), dtype=torch.float32),
            "ogm": torch.zeros((1, 2, 64, 64), dtype=torch.float32),
            "action": torch.zeros((1, 2), dtype=torch.float32),
        }
        output = critic(obs)
        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_sac_obs2tensor_ignores_extra_lidar_key_when_config_excludes_lidar(self) -> None:
        config = {
            "discrete": False,
            "observation_shape": {
                "target": (5,),
                "action_mask": (N_DISCRETE_ACTION,),
                "ogm": (2, 64, 64),
            },
            "action_dim": 2,
            "actor_layers": actor_config(),
            "critic_layers": critic_config(),
        }
        agent = SACAgent(config)
        obs = {
            "target": np.zeros((5,), dtype=np.float32),
            "action_mask": np.ones((N_DISCRETE_ACTION,), dtype=np.float32),
            "ogm": np.zeros((2, 64, 64), dtype=np.float32),
            "lidar": np.zeros((120,), dtype=np.float32),
        }
        tensor_obs = agent.obs2tensor(obs)
        self.assertIn("ogm", tensor_obs)
        self.assertNotIn("lidar", tensor_obs)


if __name__ == "__main__":
    unittest.main()
