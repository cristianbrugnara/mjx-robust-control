from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import visualize
from visualize import (
    build_review_record_args,
    load_trajectory_set,
    matched_axis_limits,
    save_static_series_plot,
    save_static_trajectory_3d_plot,
    save_static_trajectory_plot,
    validate_compatible_sets,
    validate_recording_args,
)


class VisualizeInputTests(unittest.TestCase):
    def test_load_trajectory_set_validates_cost_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectories.npy"
            np.save(path, np.zeros((3, 5, 4), dtype=np.float32))
            np.save(path.with_name("costs.npy"), np.zeros((2,), dtype=np.float32))

            with self.assertRaises(ValueError):
                load_trajectory_set(str(path))

    def test_compare_sets_require_same_state_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary_path = Path(tmp) / "primary.npy"
            compare_path = Path(tmp) / "compare.npy"
            np.save(primary_path, np.zeros((3, 5, 4), dtype=np.float32))
            np.save(compare_path, np.zeros((2, 5, 6), dtype=np.float32))

            primary = load_trajectory_set(str(primary_path))
            compare = load_trajectory_set(str(compare_path))

            with self.assertRaises(ValueError):
                validate_compatible_sets(primary, compare)

    def test_compare_sets_return_shared_rollout_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary_path = Path(tmp) / "primary.npy"
            compare_path = Path(tmp) / "compare.npy"
            np.save(primary_path, np.zeros((3, 5, 4), dtype=np.float32))
            np.save(compare_path, np.zeros((2, 6, 4), dtype=np.float32))

            primary = load_trajectory_set(str(primary_path))
            compare = load_trajectory_set(str(compare_path))

            self.assertEqual(validate_compatible_sets(primary, compare), 2)

    def test_static_series_plot_writes_png_with_matched_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "series.png"
            series = {
                "Controller A": np.asarray([0.0, 1.0, 2.0]),
                "Controller B": np.asarray([2.0, 3.0, 4.0]),
            }

            xlim, ylim = matched_axis_limits(series)
            result = save_static_series_plot(
                series_by_label=series,
                output_path=output_path,
                title="test",
                ylabel="value",
                xlim=xlim,
                ylim=ylim,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(result.xlim, xlim)
            self.assertEqual(result.ylim, ylim)

    def test_static_trajectory_plot_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trajectory.png"
            positions = {
                "Controller A": np.zeros((3, 1, 2), dtype=np.float32),
                "Controller B": np.ones((3, 1, 2), dtype=np.float32),
            }

            result = save_static_trajectory_plot(
                positions_by_label=positions,
                output_path=output_path,
                title="trajectory",
            )

            self.assertTrue(output_path.exists())
            self.assertIsNotNone(result.xlim)
            self.assertIsNotNone(result.ylim)

    def test_static_trajectory_markers_reuse_line_colors(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figures = []
        original_subplots = plt.subplots

        def capture_subplots(*args, **kwargs):
            fig, ax = original_subplots(*args, **kwargs)
            figures.append(fig)
            return fig, ax

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trajectory.png"
            positions = {
                "Controller A": np.asarray(
                    [
                        [[0.0, 0.0], [0.0, 1.0]],
                        [[1.0, 0.5], [1.0, 1.5]],
                        [[2.0, 1.0], [2.0, 2.0]],
                    ],
                    dtype=np.float32,
                ),
            }
            agent_colors = [
                np.asarray([0.34, 0.52, 0.96, 0.92], dtype=np.float32),
                np.asarray([0.98, 0.56, 0.30, 0.92], dtype=np.float32),
            ]

            with (
                mock.patch.object(visualize, "_pyplot", return_value=plt),
                mock.patch.object(plt, "subplots", side_effect=capture_subplots),
                mock.patch.object(plt, "close"),
            ):
                save_static_trajectory_plot(
                    positions_by_label=positions,
                    output_path=output_path,
                    title="trajectory",
                    agent_colors=agent_colors,
                )

            ax = figures[0].axes[0]
            for agent_index in range(2):
                line_color = np.asarray(ax.lines[agent_index].get_color(), dtype=np.float32)
                start_marker_color = ax.collections[agent_index * 2].get_facecolors()[0]
                end_marker_color = ax.collections[agent_index * 2 + 1].get_facecolors()[0]

                np.testing.assert_allclose(line_color, agent_colors[agent_index], atol=1e-6)
                np.testing.assert_allclose(start_marker_color, line_color, atol=1e-6)
                np.testing.assert_allclose(end_marker_color, line_color, atol=1e-6)

    def test_static_trajectory_reuses_auto_colors_across_controllers(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figures = []
        original_subplots = plt.subplots

        def capture_subplots(*args, **kwargs):
            fig, ax = original_subplots(*args, **kwargs)
            figures.append(fig)
            return fig, ax

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trajectory.png"
            positions = {
                "Incumbent": np.asarray(
                    [
                        [[0.0, 0.0], [0.0, 1.0]],
                        [[1.0, 0.5], [1.0, 1.5]],
                    ],
                    dtype=np.float32,
                ),
                "Candidate": np.asarray(
                    [
                        [[0.1, 0.0], [0.1, 1.0]],
                        [[1.1, 0.5], [1.1, 1.5]],
                    ],
                    dtype=np.float32,
                ),
            }

            with (
                mock.patch.object(visualize, "_pyplot", return_value=plt),
                mock.patch.object(plt, "subplots", side_effect=capture_subplots),
                mock.patch.object(plt, "close"),
            ):
                save_static_trajectory_plot(
                    positions_by_label=positions,
                    output_path=output_path,
                    title="trajectory",
                )

            ax = figures[0].axes[0]
            self.assertEqual(ax.lines[0].get_color(), ax.lines[2].get_color())
            self.assertEqual(ax.lines[1].get_color(), ax.lines[3].get_color())
            self.assertEqual(ax.lines[0].get_linestyle(), "-")
            self.assertEqual(ax.lines[2].get_linestyle(), "--")

    def test_single_controller_trajectory_legend_labels_agents(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figures = []
        original_subplots = plt.subplots

        def capture_subplots(*args, **kwargs):
            fig, ax = original_subplots(*args, **kwargs)
            figures.append(fig)
            return fig, ax

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trajectory.png"
            positions = {
                "Baseline controller": np.asarray(
                    [
                        [[0.0, 0.0], [1.0, 0.0]],
                        [[0.5, 0.5], [1.5, 0.5]],
                        [[1.0, 1.0], [2.0, 1.0]],
                    ],
                    dtype=np.float32,
                ),
            }

            with (
                mock.patch.object(visualize, "_pyplot", return_value=plt),
                mock.patch.object(plt, "subplots", side_effect=capture_subplots),
                mock.patch.object(plt, "close"),
            ):
                save_static_trajectory_plot(
                    positions_by_label=positions,
                    output_path=output_path,
                    title="trajectory",
                )

            legend = figures[0].axes[0].get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual([text.get_text() for text in legend.get_texts()], ["Agent 1", "Agent 2"])

    def test_3d_box_aspect_keeps_small_z_readable(self) -> None:
        aspect = visualize._readable_3d_box_aspect((-2.0, 2.0), (-1.0, 2.0), (0.2, 0.6))

        self.assertEqual(aspect[:2], (4.0, 3.0))
        self.assertGreaterEqual(aspect[2], 2.2)

    def test_static_trajectory_3d_plot_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "trajectory_3d.png"
            positions = {
                "Controller A": np.zeros((3, 1, 3), dtype=np.float32),
                "Controller B": np.ones((3, 1, 3), dtype=np.float32),
            }

            result = save_static_trajectory_3d_plot(
                positions_by_label=positions,
                output_path=output_path,
                title="trajectory 3D",
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(result.kind, "trajectory_3d")
            self.assertIsNotNone(result.xlim)
            self.assertIsNotNone(result.ylim)

    def test_review_record_args_validate_video_settings(self) -> None:
        default_args = build_review_record_args(output_path="/tmp/default.mp4", rollout_index=0)

        self.assertEqual(default_args.record_camera, "orbit")
        self.assertEqual(default_args.record_distance, 4.0)
        self.assertEqual(default_args.trace_width, 0.004)
        self.assertEqual(default_args.trace_z, 0.012)

        args = build_review_record_args(
            output_path="/tmp/out.mp4",
            rollout_index=2,
            playback_speed=1.25,
            camera="angled",
            show_traces=True,
            qpos_dim_per_entity=7,
            qvel_dim_per_entity=6,
        )

        self.assertEqual(args.record_path, "/tmp/out.mp4")
        self.assertEqual(args.rollout_idx, 2)
        self.assertEqual(args.selection, "single")
        self.assertTrue(args.record_split_rollouts)
        validate_recording_args(args)

        with self.assertRaises(ValueError):
            build_review_record_args(output_path="/tmp/out.mp4", rollout_index=0, playback_speed=0.0)
        with self.assertRaises(ValueError):
            build_review_record_args(output_path="/tmp/out.mp4", rollout_index=0, camera="bad")
        with self.assertRaises(ValueError):
            build_review_record_args(output_path="/tmp/out.mp4", rollout_index=0, record_distance=0.0)
        with self.assertRaises(ValueError):
            build_review_record_args(output_path="/tmp/out.mp4", rollout_index=0, trace_width=0.0)


if __name__ == "__main__":
    unittest.main()
