import json
import unittest
from pathlib import Path

from rv_heston_hmm.model_guardrails import MAX_ABS_MU, MAX_KAPPA, MAX_VOL_OF_VOL, regime_feller_margin


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CalibrationArtifactTests(unittest.TestCase):
    def test_model_calibration_json_is_within_runtime_guardrails(self):
        config = json.loads((PROJECT_ROOT / "config" / "model_calibration.json").read_text())
        params = config["regime"]["params"]
        self.assertGreater(len(params), 0)
        for param in params:
            with self.subTest(regime=param.get("name")):
                self.assertLessEqual(abs(float(param["mu"])), MAX_ABS_MU)
                self.assertLessEqual(float(param["kappa"]), MAX_KAPPA)
                self.assertLessEqual(float(param["vol_of_vol"]), MAX_VOL_OF_VOL)
                self.assertGreaterEqual(regime_feller_margin(param), 0.0)

        metadata = config["metadata"]["historical_calibration"]
        self.assertLessEqual(float(metadata["kappa_estimate"]), MAX_KAPPA)
        self.assertLessEqual(float(metadata["vol_of_vol_estimate"]), MAX_VOL_OF_VOL)


if __name__ == "__main__":
    unittest.main()
