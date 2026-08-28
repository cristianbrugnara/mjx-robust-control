from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import (
    TrainConfig,
    append_learning_curve_point,
    parse_args,
    train,
    write_training_progress,
)


class TrainLifecycleTests(unittest.TestCase):
    def test_progress_json_records_resume_and_budget_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "training_progress.json"
            config = TrainConfig(
                progress_path=str(progress_path),
                start_epoch=7,
                resume_from_checkpoint="previous.eqx",
                initial_elapsed_training_seconds=10.5,
                progress_configured_fold_time_budget_seconds=60.0,
            )

            write_training_progress(
                config,
                save_path=Path(tmp) / "controller.eqx",
                last_epoch=9,
                elapsed_this_invocation=2.25,
                status="training",
                best_val=3.5,
            )

            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_epoch"], 9)
            self.assertEqual(payload["next_epoch"], 10)
            self.assertEqual(payload["status"], "training")
            self.assertEqual(payload["start_epoch"], 7)
            self.assertEqual(payload["resume_from_checkpoint"], "previous.eqx")
            self.assertAlmostEqual(payload["elapsed_training_seconds"], 12.75)
            self.assertAlmostEqual(payload["configured_fold_time_budget_seconds"], 60.0)
            self.assertEqual(payload["best_val"], 3.5)

    def test_learning_curve_jsonl_appends_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = Path(tmp) / "learning_curves.jsonl"
            config = TrainConfig(
                learning_curve_path=str(curve_path),
                start_epoch=3,
                resume_from_checkpoint="previous.eqx",
                initial_elapsed_training_seconds=4.0,
            )

            append_learning_curve_point(
                config,
                epoch=3,
                elapsed_this_invocation=1.5,
                train_loss=2.0,
                val_loss=1.5,
                best_val=1.5,
                tau=None,
                status="training",
            )
            append_learning_curve_point(
                config,
                epoch=4,
                elapsed_this_invocation=2.5,
                train_loss=1.75,
                val_loss=None,
                best_val=1.5,
                tau=0.1,
                status="training",
            )

            lines = [
                json.loads(line)
                for line in curve_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([line["epoch"] for line in lines], [3, 4])
            self.assertEqual(lines[0]["val_loss"], 1.5)
            self.assertIsNone(lines[1]["val_loss"])
            self.assertAlmostEqual(lines[1]["elapsed_training_seconds"], 6.5)
            self.assertEqual(lines[1]["resume_from_checkpoint"], "previous.eqx")

    def test_parse_args_accepts_lifecycle_options(self) -> None:
        argv = [
            "train.py",
            "--max_wall_time_seconds",
            "12.5",
            "--resume_from_checkpoint",
            "previous.eqx",
            "--start_epoch",
            "11",
            "--progress_path",
            "training_progress.json",
            "--learning_curve_path",
            "learning_curves.jsonl",
            "--initial_elapsed_training_seconds",
            "7.25",
            "--progress_configured_fold_time_budget_seconds",
            "100.0",
            "--early_stopping_epsilon",
            "0.001",
            "--early_stopping_patience",
            "9",
        ]
        with patch.object(sys, "argv", argv):
            config = parse_args()

        self.assertEqual(config.max_wall_time_seconds, 12.5)
        self.assertEqual(config.resume_from_checkpoint, "previous.eqx")
        self.assertEqual(config.start_epoch, 11)
        self.assertEqual(config.progress_path, "training_progress.json")
        self.assertEqual(config.learning_curve_path, "learning_curves.jsonl")
        self.assertEqual(config.initial_elapsed_training_seconds, 7.25)
        self.assertEqual(config.progress_configured_fold_time_budget_seconds, 100.0)
        self.assertEqual(config.early_stopping_epsilon, 0.001)
        self.assertEqual(config.early_stopping_patience, 9)

    def test_parse_args_can_disable_early_stopping(self) -> None:
        with patch.object(sys, "argv", ["train.py", "--disable_early_stopping"]):
            config = parse_args()
        self.assertIsNone(config.early_stopping_epsilon)

    def test_lifecycle_preflight_validation_happens_before_system_load(self) -> None:
        cases = (
            (TrainConfig(max_wall_time_seconds=0.0), "max_wall_time_seconds"),
            (TrainConfig(early_stopping_epsilon=-1.0e-4), "early_stopping_epsilon"),
            (TrainConfig(early_stopping_patience=0), "early_stopping_patience"),
            (TrainConfig(start_epoch=-1), "start_epoch"),
        )
        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    train(config)


if __name__ == "__main__":
    unittest.main()
