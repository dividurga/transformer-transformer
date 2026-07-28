"""
Hardware optimization results plotting script.

This script parses experiment results from WandB runs and generates
publication-quality plots comparing different optimization approaches.

The script runs in two phases:
1. Filter phase: Discover and filter runs from wandb metadata files
2. Extract phase: Load data from the filtered runs

The filter results are cached to a human-editable YAML file
(filter_cache.yaml) by default. This allows manual curation of which
runs to include/exclude without re-running the expensive filtering step.

Usage:
    # paths.results_root is the directory holding your wandb eval run
    # directories (typically the repo's wandb/ directory)
    python scripts/plot_hardware_opt.py paths.results_root=./wandb
    python scripts/plot_hardware_opt.py paths.results_root=./wandb paths.output_dir=./my_output
    python scripts/plot_hardware_opt.py paths.results_root=./wandb processing.use_cache=false

    # Plot a partial reproduction (only the panels you have runs for)
    python scripts/plot_hardware_opt.py paths.results_root=./wandb \
        'experiments=[viperx/tracking_only/single_traj]'

    # Force regeneration of filter cache
    python scripts/plot_hardware_opt.py paths.results_root=./wandb processing.force_filter=true

See docs/visualization.md ("Reproducing the co-design results figure") for the
run-tagging convention this script uses to classify runs.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import yaml
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from t2.eval.utils import summarize_rollout

logger = logging.getLogger(__name__)


# =============================================================================
# Debug Logging
# =============================================================================


@dataclass
class ParseDecision:
    """Records a parsing decision for a single run."""

    run_dir: str
    metadata_path: str
    decision: str  # "added", "skipped", "replaced", "replaced_by"
    reason: str
    approach: str | None = None
    config_summary: str | None = None
    replaced_by: str | None = None  # For "replaced" decisions


class ParseLog:
    """Collects detailed parsing decisions for debugging."""

    def __init__(self):
        self.decisions: list[ParseDecision] = []

    def log(
        self,
        metadata_path: Path,
        decision: str,
        reason: str,
        approach: str | None = None,
        config: RunConfig | None = None,
        replaced_by: str | None = None,
    ) -> None:
        """Log a parsing decision."""
        run_dir = str(metadata_path.parent.parent)
        config_summary = None
        if config:
            config_summary = (
                f"{config.approach}/{config.design_space}/{config.reward_fn}/"
                f"budget={config.compute_budget}/multitraj={config.is_multitraj}"
            )
        self.decisions.append(
            ParseDecision(
                run_dir=run_dir,
                metadata_path=str(metadata_path),
                decision=decision,
                reason=reason,
                approach=approach,
                config_summary=config_summary,
                replaced_by=replaced_by,
            )
        )

    def save(self, output_path: Path) -> None:
        """Save the log to a text file."""
        # Group decisions by type
        added = [d for d in self.decisions if d.decision == "added"]
        skipped = [d for d in self.decisions if d.decision == "skipped"]
        replaced = [d for d in self.decisions if d.decision == "replaced"]
        data_skipped = [d for d in self.decisions if d.decision == "data_skipped"]

        lines = []
        lines.append("=" * 80)
        lines.append("HARDWARE OPTIMIZATION RESULTS PARSING LOG")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total runs processed: {len(self.decisions)}")
        lines.append(f"  Added (used in plots): {len(added)}")
        lines.append(f"  Skipped (filtered out): {len(skipped)}")
        lines.append(f"  Replaced (by newer run): {len(replaced)}")
        lines.append(f"  Data skipped (extraction failed): {len(data_skipped)}")
        lines.append("")

        # Skip reason breakdown
        skip_reasons: dict[str, int] = {}
        for d in skipped:
            skip_reasons[d.reason] = skip_reasons.get(d.reason, 0) + 1
        if skip_reasons:
            lines.append("SKIP REASONS BREAKDOWN")
            lines.append("-" * 40)
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {count:4d}  {reason}")
            lines.append("")

        # Added runs
        lines.append("=" * 80)
        lines.append(f"ADDED RUNS ({len(added)})")
        lines.append("=" * 80)
        for d in sorted(added, key=lambda x: x.run_dir):
            lines.append(f"\n{d.run_dir}")
            lines.append(f"  Config: {d.config_summary}")
            lines.append(f"  Reason: {d.reason}")

        # Data skipped runs
        if data_skipped:
            lines.append("")
            lines.append("=" * 80)
            lines.append(f"DATA SKIPPED RUNS ({len(data_skipped)})")
            lines.append("=" * 80)
            for d in sorted(data_skipped, key=lambda x: x.run_dir):
                lines.append(f"\n{d.run_dir}")
                lines.append(f"  Config: {d.config_summary}")
                lines.append(f"  Reason: {d.reason}")

        # Replaced runs
        if replaced:
            lines.append("")
            lines.append("=" * 80)
            lines.append(f"REPLACED RUNS ({len(replaced)})")
            lines.append("=" * 80)
            for d in sorted(replaced, key=lambda x: x.run_dir):
                lines.append(f"\n{d.run_dir}")
                lines.append(f"  Config: {d.config_summary}")
                lines.append(f"  Replaced by: {d.replaced_by}")

        # Skipped runs (grouped by reason)
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"SKIPPED RUNS ({len(skipped)})")
        lines.append("=" * 80)

        # Group skipped by reason
        skipped_by_reason: dict[str, list[ParseDecision]] = {}
        for d in skipped:
            if d.reason not in skipped_by_reason:
                skipped_by_reason[d.reason] = []
            skipped_by_reason[d.reason].append(d)

        for reason in sorted(skipped_by_reason.keys()):
            decisions = skipped_by_reason[reason]
            lines.append(f"\n--- {reason} ({len(decisions)} runs) ---")
            for d in sorted(decisions, key=lambda x: x.run_dir):
                lines.append(f"  {d.run_dir}")

        output_path.write_text("\n".join(lines))
        logger.info(f"Saved parsing log to {output_path}")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, eq=True)
class RunConfig:
    """Identifies a unique experiment run configuration."""

    approach: str
    design_space: str
    reward_fn: str
    compute_budget: int
    is_multitraj: bool

    def to_exp_name(self) -> str:
        """Convert to experiment name string."""
        traj_type = "multi_traj" if self.is_multitraj else "single_traj"
        return f"{self.design_space}/{self.reward_fn}/{traj_type}"


@dataclass
class RunData:
    """Holds parsed data for a single run."""

    config: RunConfig
    rewards: NDArray[np.floating]
    optimize_times: NDArray[np.floating]
    run_dir: Path


@dataclass
class ExperimentData:
    """Aggregated data for plotting a single experiment."""

    name: str
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    avg_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Data Parsing
# =============================================================================


@dataclass
class RunCandidate:
    """A candidate run before data extraction, used for deduplication."""

    config: RunConfig
    metadata_path: Path
    run_dir: Path
    zarr_filename: str  # "optimized_hardwares.zarr" or "hardware_opt_summary.zarr"

    def get_run_date(self) -> datetime:
        """Extract the run date from the run directory name.

        Handles both online ("run-<ts>-<id>") and offline
        ("offline-run-<ts>-<id>") wandb directory layouts.
        """
        try:
            return datetime.strptime(
                self.run_dir.name.split("-")[-2], "%Y%m%d_%H%M%S"
            )
        except (IndexError, ValueError):
            # e.g. a custom WANDB_RUN_ID containing hyphens. Fall back to the
            # metadata file's mtime rather than a sentinel: deduplication keeps
            # the newest run per configuration, and a sentinel would make an
            # unparseable directory always lose — silently discarding a rerun
            # in favour of the older run it was meant to replace.
            mtime = datetime.fromtimestamp(self.metadata_path.stat().st_mtime)
            logger.warning(
                f"{self.run_dir}: cannot read a timestamp from the directory "
                f"name; using its mtime ({mtime:%Y-%m-%d %H:%M:%S}) to pick "
                f"between duplicate runs"
            )
            return mtime


class ResultsParser:
    """Parses experiment results from WandB run directories."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.parse_log = ParseLog()

    def parse_all(self) -> dict[str, ExperimentData]:
        """Parse all experiment results and return aggregated data."""
        # Check if we should use cached filter results
        force_filter = getattr(self.cfg.processing, "force_filter", False)
        candidates: dict[RunConfig, RunCandidate] | None = None

        if not force_filter:
            candidates = self._load_filter_cache()

        # Phase 1: Collect all candidate runs (without loading data)
        if candidates is None:
            logger.info("Phase 1: Discovering and filtering runs...")
            results_root = Path(self.cfg.paths.results_root)
            wandb_metadata_paths = list(
                sorted(results_root.rglob("wandb-metadata.json"))
            )
            candidates = {}

            for metadata_path in wandb_metadata_paths:
                candidate = self._identify_cmaes_run(metadata_path)
                if candidate:
                    self._update_candidates(candidates, candidate, metadata_path)

            for metadata_path in wandb_metadata_paths:
                candidate = self._identify_guided_run(metadata_path)
                if candidate:
                    self._update_candidates(candidates, candidate, metadata_path)

            logger.info(f"  Found {len(candidates)} unique run configurations")

            # Save filter cache for future runs
            self._save_filter_cache(candidates)
        else:
            logger.info("Phase 1: Using cached filter results (skipping discovery)")

        # Phase 2: Extract data only from filtered runs
        logger.info("Phase 2: Extracting data from selected runs...")
        experiments = {name: ExperimentData(name=name) for name in self.cfg.experiments}

        for config, candidate in candidates.items():
            run_data = self._extract_run_data(candidate)
            if run_data:
                self._add_run_data(experiments, run_data)
                self.parse_log.log(
                    metadata_path=candidate.metadata_path,
                    decision="added",
                    reason="Data extraction successful",
                    config=config,
                )

        # Phase 3: Apply optimize time overrides from config
        if self.cfg.get("optimize_time_overrides"):
            logger.info("Phase 3: Applying optimize time overrides...")
            self._apply_optimize_time_overrides(experiments)

        return experiments

    def _update_candidates(
        self,
        candidates: dict[RunConfig, RunCandidate],
        new_candidate: RunCandidate,
        metadata_path: Path,
    ) -> None:
        """Update candidates dict, keeping only the latest run for each config."""
        config = new_candidate.config
        if config in candidates:
            new_date = new_candidate.get_run_date()
            old_date = candidates[config].get_run_date()
            if new_date > old_date:
                logger.info(
                    f"Picking {new_candidate.run_dir.name} over "
                    f"{candidates[config].run_dir.name}"
                )
                # Log the old candidate as replaced
                self.parse_log.log(
                    metadata_path=candidates[config].metadata_path,
                    decision="replaced",
                    reason="Newer run with same config found",
                    config=config,
                    replaced_by=str(new_candidate.run_dir),
                )
                candidates[config] = new_candidate
            else:
                logger.info(f"Skipping {new_candidate.run_dir.name} (older duplicate)")
                # Log the new candidate as replaced_by (it's older)
                self.parse_log.log(
                    metadata_path=metadata_path,
                    decision="replaced",
                    reason="Older duplicate of existing run",
                    config=config,
                    replaced_by=str(candidates[config].run_dir),
                )
        else:
            candidates[config] = new_candidate

    def _identify_cmaes_run(self, metadata_path: Path) -> RunCandidate | None:
        """Identify a CMA-ES run without loading data."""
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="Invalid JSON or IO error",
                approach="cmaes",
            )
            return None

        # Check if this is a CMA-ES run
        if not any(
            prog in metadata.get("program", "")
            for prog in self.cfg.filters.cmaes_programs
        ):
            # Don't log - this is expected for non-cmaes runs
            return None

        # Extract and validate tags
        tags = self._extract_tags(metadata)
        if tags is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="No tags found in metadata",
                approach="cmaes",
            )
            return None

        if "cmaes" not in tags:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Missing 'cmaes' tag (tags: {tags})",
                approach="cmaes",
            )
            return None

        if any(tag in tags for tag in self.cfg.filters.excluded_tags):
            excluded = [t for t in tags if t in self.cfg.filters.excluded_tags]
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Has excluded tag(s): {excluded}",
                approach="cmaes",
            )
            return None

        # Check for excluded commits
        git_commit = metadata.get("git", {}).get("commit", "")
        if git_commit in self.cfg.filters.excluded_commits:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Excluded git commit: {git_commit[:12]}",
                approach="cmaes",
            )
            return None

        # Check zarr exists (but don't load it yet)
        zarr_path = metadata_path.parent / "optimized_hardwares.zarr"
        if not zarr_path.exists():
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="Missing optimized_hardwares.zarr",
                approach="cmaes",
            )
            return None

        # Parse run configuration
        design_space = self._extract_design_space(tags)
        if design_space is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Unknown design space (tags: {tags})",
                approach="cmaes",
            )
            return None

        reward_fn = self._extract_reward_fn(tags)
        if reward_fn is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Could not extract reward function (tags: {tags})",
                approach="cmaes",
            )
            return None

        compute_budget = self._extract_compute_budget(metadata, "max_fun=")
        if compute_budget is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="Could not extract compute budget (max_fun=)",
                approach="cmaes",
            )
            return None

        is_multitraj = "multitraj" in tags
        if design_space == "bimanual":
            # HARDCODE for now
            is_multitraj = True
        exp_name = f"{design_space}/{reward_fn}/{'multi_traj' if is_multitraj else 'single_traj'}"

        if exp_name not in self.cfg.experiments:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Experiment '{exp_name}' not in configured experiments",
                approach="cmaes",
            )
            return None

        # Apply host filters
        if not self._check_host_filter(exp_name, metadata):
            host = metadata.get("host", "unknown")
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Host '{host}' not in allowed hosts for {exp_name}",
                approach="cmaes",
            )
            return None

        # Determine approach (random if compute_budget == 0)
        approach = "random" if compute_budget == 0 else "cmaes"

        run_config = RunConfig(
            approach=approach,
            design_space=design_space,
            reward_fn=reward_fn,
            compute_budget=compute_budget,
            is_multitraj=is_multitraj,
        )

        run_dir = metadata_path.parent.parent
        return RunCandidate(
            config=run_config,
            metadata_path=metadata_path,
            run_dir=run_dir,
            zarr_filename="optimized_hardwares.zarr",
        )

    def _identify_guided_run(self, metadata_path: Path) -> RunCandidate | None:
        """Identify a guided/unguided run without loading data."""
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="Invalid JSON or IO error",
                approach="guided",
            )
            return None

        # Check if this is a guided run
        if not any(
            prog in metadata.get("program", "")
            for prog in self.cfg.filters.guided_programs
        ):
            # Don't log - this is expected for non-guided runs
            return None

        # Extract and validate tags
        tags = self._extract_tags(metadata)
        if tags is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="No tags found in metadata",
                approach="guided",
            )
            return None

        if "guided" not in tags and "unguided" not in tags:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Missing 'guided' or 'unguided' tag (tags: {tags})",
                approach="guided",
            )
            return None

        approach = "guided" if "guided" in tags else "unguided"

        if any(tag in tags for tag in self.cfg.filters.excluded_tags):
            excluded = [t for t in tags if t in self.cfg.filters.excluded_tags]
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Has excluded tag(s): {excluded}",
                approach=approach,
            )
            return None

        # Check zarr exists (but don't load it yet)
        # either hardware_opt_summary.zarr or hardware_opt.zarr
        zarr_base_name = "hardware_opt_summary.zarr"
        zarr_path = metadata_path.parent / zarr_base_name
        if not zarr_path.exists():
            zarr_base_name = "hardware_opt.zarr"
            zarr_path = metadata_path.parent / zarr_base_name
        if not zarr_path.exists():
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="Missing hardware_opt_summary.zarr and hardware_opt.zarr",
                approach=approach,
            )
            return None

        # Parse run configuration
        design_space = self._extract_design_space(tags)
        if design_space is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Unknown design space (tags: {tags})",
                approach=approach,
            )
            return None

        # Check checkpoint filter
        if not self._check_checkpoint_filter(design_space, metadata):
            actual_ckpt = self._extract_checkpoint_path(metadata)
            allowed_ckpt = None
            if hasattr(self.cfg.filters, "guided_checkpoints"):
                guided_checkpoints = OmegaConf.to_container(
                    self.cfg.filters.guided_checkpoints
                )
                allowed_ckpt = guided_checkpoints.get(design_space)
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Checkpoint mismatch: got '{actual_ckpt}', allowed '{allowed_ckpt}'",
                approach=approach,
            )
            return None

        # Extract compute budget from num_seeds argument
        compute_budget = self._extract_guided_compute_budget(metadata)
        if compute_budget is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason="Could not extract compute budget (num_seeds)",
                approach=approach,
            )
            return None

        if compute_budget > self.cfg.filters.max_guided_compute_budget:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Compute budget {compute_budget} > max {self.cfg.filters.max_guided_compute_budget}",
                approach=approach,
            )
            return None

        reward_fn = self._extract_reward_fn(tags)
        if reward_fn is None:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Could not extract reward function (tags: {tags})",
                approach=approach,
            )
            return None

        # Normalize reward function name
        if reward_fn == "trackingonly":
            reward_fn = "tracking_only"

        # Check for excluded commits
        git_commit = metadata.get("git", {}).get("commit", "")
        if (
            reward_fn in {"tracking_size", "tracking_weight"}
            and git_commit in self.cfg.filters.excluded_commits
        ):
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Excluded commit for {reward_fn}: {git_commit[:12]}",
                approach=approach,
            )
            return None

        is_multitraj = "multitraj" in tags

        # Check multitraj-specific excluded commits
        if is_multitraj and git_commit in self.cfg.filters.excluded_multitraj_commits:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Excluded multitraj commit: {git_commit[:12]}",
                approach=approach,
            )
            return None

        exp_name = f"{design_space}/{reward_fn}/{'multi_traj' if is_multitraj else 'single_traj'}"

        if exp_name not in self.cfg.experiments:
            self.parse_log.log(
                metadata_path=metadata_path,
                decision="skipped",
                reason=f"Experiment '{exp_name}' not in configured experiments",
                approach=approach,
            )
            return None

        run_config = RunConfig(
            approach=approach,
            design_space=design_space,
            reward_fn=reward_fn,
            compute_budget=compute_budget,
            is_multitraj=is_multitraj,
        )

        run_dir = metadata_path.parent.parent
        return RunCandidate(
            config=run_config,
            metadata_path=metadata_path,
            run_dir=run_dir,
            zarr_filename=zarr_base_name,
        )

    def _extract_run_data(self, candidate: RunCandidate) -> RunData | None:
        """Extract data from a candidate run."""
        zarr_path = candidate.metadata_path.parent / candidate.zarr_filename

        try:
            episode_data = summarize_rollout(
                str(zarr_path),
                use_cache=self.cfg.processing.use_cache,
                use_pbar=self.cfg.processing.show_progress,
                reduce=False,
            )
        except Exception as e:
            logger.warning(f"Failed to load {zarr_path}: {e}")
            self.parse_log.log(
                metadata_path=candidate.metadata_path,
                decision="data_skipped",
                reason=f"Failed to load zarr: {e}",
                config=candidate.config,
            )
            return None

        # summarize_rollout() emits the hardware metadata under a
        # "hardware_meta/" prefix; fall back to the bare key for summaries
        # produced before that prefix existed.
        def get_meta(name: str) -> NDArray[np.floating]:
            for key in (f"hardware_meta/{name}", name):
                if key in episode_data:
                    return np.asarray(episode_data[key])
            raise KeyError(
                f"{zarr_path}: neither 'hardware_meta/{name}' nor '{name}' "
                f"found in rollout summary"
            )

        actual_values = get_meta("actual_value")
        optimize_times = get_meta("optimize_time")

        if len(actual_values) != self.cfg.filters.required_actual_values:
            logger.warning(
                f"{candidate.run_dir}: has {len(actual_values)} actual values"
            )

        return RunData(
            config=candidate.config,
            rewards=actual_values,
            optimize_times=optimize_times,
            run_dir=candidate.run_dir,
        )

    def _add_run_data(
        self, experiments: dict[str, ExperimentData], run_data: RunData
    ) -> None:
        """Add run data to the appropriate experiment."""
        exp_name = run_data.config.to_exp_name()
        if exp_name not in experiments:
            return

        exp = experiments[exp_name]
        config = run_data.config

        # Create raw DataFrame
        raw_df = pd.DataFrame(
            {
                "reward": run_data.rewards,
                "optimize_time": run_data.optimize_times,
                "approach": [config.approach] * len(run_data.rewards),
            }
        )
        exp.raw_df = pd.concat([exp.raw_df, raw_df])

        # Compute statistics
        avg_time = np.mean(run_data.optimize_times)
        avg_reward = np.mean(run_data.rewards)
        reward_std = np.std(run_data.rewards)
        reward_ci = st.t.interval(
            confidence=self.cfg.processing.confidence_level,
            df=len(run_data.rewards) - 1,
            loc=avg_reward,
            scale=st.sem(run_data.rewards),
        )

        # Get plot times for random approach endpoints
        plot_times_key = f"{config.design_space}/{'multi_traj' if config.is_multitraj else 'single_traj'}"
        plot_times = self.cfg.plot_times[plot_times_key]

        # Create averaged DataFrame
        if config.approach == "random":
            # Random approach spans the entire x-axis
            avg_df = pd.DataFrame(
                {
                    "avg_reward": [avg_reward, avg_reward],
                    "reward_std": [reward_std, reward_std],
                    "avg_optimize_time": [min(plot_times), max(plot_times)],
                    "approach": ["random", "random"],
                    "reward_ci_low": [reward_ci[0], reward_ci[0]],
                    "reward_ci_high": [reward_ci[1], reward_ci[1]],
                    "compute_budget": [config.compute_budget, config.compute_budget],
                }
            )
        else:
            avg_df = pd.DataFrame(
                {
                    "avg_reward": [avg_reward],
                    "reward_std": [reward_std],
                    "avg_optimize_time": [avg_time],
                    "approach": [config.approach],
                    "reward_ci_low": [reward_ci[0]],
                    "reward_ci_high": [reward_ci[1]],
                    "compute_budget": [config.compute_budget],
                }
            )

        exp.avg_df = pd.concat([exp.avg_df, avg_df])

    def _apply_optimize_time_overrides(
        self, experiments: dict[str, ExperimentData]
    ) -> None:
        """Apply optimize time overrides from config.

        The config format is:
            optimize_time_overrides:
              <design_space>:
                <traj_type>:  # "multi_traj" or "single_traj"
                  <approach>:  # "guided", "unguided", or "cmaes"
                    <compute_budget>: <override_time>

        Since changing reward functions doesn't affect optimize time,
        the config doesn't expose reward_fn - overrides apply across all
        reward functions for the same design_space/traj_type.
        """
        overrides = self.cfg.optimize_time_overrides

        for exp_name, exp in experiments.items():
            if exp.avg_df.empty:
                continue

            # Parse experiment name: "{design_space}/{reward_fn}/{traj_type}"
            parts = exp_name.split("/")
            if len(parts) != 3:
                continue

            design_space, _, traj_type = parts

            # Check if we have overrides for this design_space/traj_type
            if design_space not in overrides:
                continue
            if traj_type not in overrides[design_space]:
                continue

            traj_overrides = overrides[design_space][traj_type]

            # Reset index to avoid duplicate index issues after concat
            exp.avg_df = exp.avg_df.reset_index(drop=True)

            # Apply overrides to each row in avg_df
            for idx in range(len(exp.avg_df)):
                approach = exp.avg_df.loc[idx, "approach"]
                compute_budget = int(exp.avg_df.loc[idx, "compute_budget"])

                if approach not in traj_overrides:
                    continue
                if compute_budget not in traj_overrides[approach]:
                    continue

                override_time = traj_overrides[approach][compute_budget]
                old_time = exp.avg_df.loc[idx, "avg_optimize_time"]
                exp.avg_df.loc[idx, "avg_optimize_time"] = override_time
                logger.debug(
                    f"Override {exp_name}/{approach}/budget={compute_budget}: "
                    f"{old_time:.2f}s -> {override_time:.2f}s"
                )

    def _extract_tags(self, metadata: dict[str, Any]) -> list[str] | None:
        """Extract tags from metadata args."""
        args = metadata.get("args", [])
        try:
            tag_str = next(arg for arg in args if "tags=[" in arg)
            return tag_str.split("tags=[")[1].split("]")[0].split(",")
        except StopIteration:
            return None

    def _extract_design_space(self, tags: list[str]) -> str | None:
        """Extract design space from tags."""
        if "viperx" in tags:
            return "viperx"
        elif "quadruped" in tags:
            return "quadruped"
        elif "bimanual" in tags:
            return "bimanual"
        return None

    def _extract_reward_fn(self, tags: list[str]) -> str | None:
        """Extract reward function name from tags."""
        try:
            tag_string = next(tag for tag in tags if "track" in tag)
            if tag_string.endswith("_quadruped"):
                return tag_string.split("_quadruped")[0]
            elif tag_string.endswith("_bimanual"):
                return tag_string.split("_bimanual")[0]
            else:
                return tag_string
        except StopIteration:
            return None

    def _extract_compute_budget(
        self, metadata: dict[str, Any], prefix: str
    ) -> int | None:
        """Extract compute budget from metadata args."""
        args = metadata.get("args", [])
        try:
            arg = next(a for a in args if prefix in a)
            return int(arg.split(prefix)[1].split(" ")[0])
        except (StopIteration, ValueError):
            return None

    def _extract_guided_compute_budget(self, metadata: dict[str, Any]) -> int | None:
        """Extract compute budget for guided runs."""
        args = metadata.get("args", [])
        try:
            arg = next(
                a for a in args if "eval_fn.hardware_optimizer_fn.num_seeds=" in a
            )
            return int(arg.split("=")[1])
        except (StopIteration, ValueError):
            return None

    def _check_host_filter(self, exp_name: str, metadata: dict[str, Any]) -> bool:
        """Check if run passes host filter for the experiment."""
        host_filters: dict[str, list[str]] = OmegaConf.to_container(  # type: ignore[assignment]
            self.cfg.filters.host_filters
        )
        if exp_name not in host_filters:
            return True

        allowed_hosts = host_filters[exp_name]
        if not allowed_hosts:
            return True

        return metadata.get("host", "") in allowed_hosts

    def _extract_checkpoint_path(self, metadata: dict[str, Any]) -> str | None:
        """Extract checkpoint path from metadata args."""
        args = metadata.get("args", [])
        try:
            arg = next(a for a in args if a.startswith("ckpt_path="))
            return arg.split("=", 1)[1]
        except StopIteration:
            return None

    def _check_checkpoint_filter(
        self, design_space: str, metadata: dict[str, Any]
    ) -> bool:
        """Check if run uses an allowed checkpoint for the design space."""
        # If no checkpoint filters configured, allow all
        if not hasattr(self.cfg.filters, "guided_checkpoints"):
            return True

        guided_checkpoints: dict[str, str] = OmegaConf.to_container(  # type: ignore[assignment]
            self.cfg.filters.guided_checkpoints
        )

        # If design space not in filter, allow all checkpoints
        if design_space not in guided_checkpoints:
            return True

        allowed_ckpt = guided_checkpoints[design_space]
        actual_ckpt = self._extract_checkpoint_path(metadata)

        if actual_ckpt is None:
            logger.debug("No checkpoint path found in metadata")
            return False

        return actual_ckpt == allowed_ckpt

    def _get_filter_cache_path(self) -> Path:
        """Get the path to the filter cache file."""
        if hasattr(self.cfg.paths, "filter_cache") and self.cfg.paths.filter_cache:
            return Path(self.cfg.paths.filter_cache)
        return Path(self.cfg.paths.output_dir) / "filter_cache.yaml"

    def _save_filter_cache(self, candidates: dict[RunConfig, RunCandidate]) -> None:
        """Save filter results to a human-editable YAML file."""
        cache_path = self._get_filter_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to list of dicts for YAML serialization
        entries = []
        for config, candidate in sorted(
            candidates.items(),
            key=lambda x: (x[0].design_space, x[0].reward_fn, x[0].approach),
        ):
            entry = {
                "approach": config.approach,
                "design_space": config.design_space,
                "reward_fn": config.reward_fn,
                "compute_budget": config.compute_budget,
                "is_multitraj": config.is_multitraj,
                "metadata_path": str(candidate.metadata_path),
                "run_dir": str(candidate.run_dir),
                "zarr_filename": candidate.zarr_filename,
            }
            entries.append(entry)

        # Write YAML with custom formatting for human readability
        with open(cache_path, "w") as f:
            f.write("# Filter cache for hardware optimization results\n")
            f.write("# This file can be manually edited to include/exclude runs\n")
            f.write(
                "# To regenerate, delete this file or set processing.force_filter=true\n"
            )
            f.write("#\n")
            f.write("# Each entry contains:\n")
            f.write("#   approach: cmaes | guided | unguided | random\n")
            f.write("#   design_space: viperx | quadruped | bimanual\n")
            f.write("#   reward_fn: tracking_only | tracking_size | etc.\n")
            f.write("#   compute_budget: number of optimization iterations/samples\n")
            f.write("#   is_multitraj: true if using multiple trajectories\n")
            f.write("#   metadata_path: path to wandb-metadata.json\n")
            f.write("#   run_dir: path to the run directory\n")
            f.write("#   zarr_filename: name of the zarr file containing results\n")
            f.write("#\n")
            f.write(f"generated: {datetime.now().isoformat()}\n")
            f.write(f"num_candidates: {len(entries)}\n")
            f.write("\ncandidates:\n")
            for entry in entries:
                f.write(f"  - approach: {entry['approach']}\n")
                f.write(f"    design_space: {entry['design_space']}\n")
                f.write(f"    reward_fn: {entry['reward_fn']}\n")
                f.write(f"    compute_budget: {entry['compute_budget']}\n")
                f.write(f"    is_multitraj: {str(entry['is_multitraj']).lower()}\n")
                f.write(f"    metadata_path: {entry['metadata_path']}\n")
                f.write(f"    run_dir: {entry['run_dir']}\n")
                f.write(f"    zarr_filename: {entry['zarr_filename']}\n")

        logger.info(
            f"Saved filter cache with {len(entries)} candidates to {cache_path}"
        )

    def _load_filter_cache(self) -> dict[RunConfig, RunCandidate] | None:
        """Load filter results from cache if it exists.

        Returns:
            Dict of RunConfig -> RunCandidate if cache exists and is valid,
            None otherwise.
        """
        cache_path = self._get_filter_cache_path()
        if not cache_path.exists():
            logger.info(f"Filter cache not found at {cache_path}")
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = yaml.safe_load(f)

            if not cache_data or "candidates" not in cache_data:
                logger.warning(f"Invalid filter cache format at {cache_path}")
                return None

            candidates: dict[RunConfig, RunCandidate] = {}
            for entry in cache_data["candidates"]:
                config = RunConfig(
                    approach=entry["approach"],
                    design_space=entry["design_space"],
                    reward_fn=entry["reward_fn"],
                    compute_budget=entry["compute_budget"],
                    is_multitraj=entry["is_multitraj"],
                )
                candidate = RunCandidate(
                    config=config,
                    metadata_path=Path(entry["metadata_path"]),
                    run_dir=Path(entry["run_dir"]),
                    zarr_filename=entry["zarr_filename"],
                )
                candidates[config] = candidate

            logger.info(
                f"Loaded {len(candidates)} candidates from filter cache at {cache_path}"
            )
            return candidates

        except (yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load filter cache: {e}")
            return None


# =============================================================================
# Plotting
# =============================================================================


class HardwareOptPlotter:
    """Creates publication-quality plots for hardware optimization results."""

    def __init__(self, cfg: DictConfig, experiments: dict[str, ExperimentData]):
        self.cfg = cfg
        self.experiments = experiments
        self._setup_style()
        self._setup_colors()

    def _setup_style(self) -> None:
        """Configure matplotlib/seaborn style."""
        sns.set_style("whitegrid", {"font.family": self.cfg.plot.font.family})

    def _setup_colors(self) -> None:
        """Setup colormaps and color normalization functions."""
        # Colormaps
        self.cmaps = {
            "cmaes": plt.get_cmap(self.cfg.approaches.cmaes.colormap),
            "guided": plt.get_cmap(self.cfg.approaches.guided.colormap),
            "unguided": plt.get_cmap(self.cfg.approaches.unguided.colormap),
        }

        # Budget ranges
        cmaes_budgets = list(self.cfg.approaches.cmaes.budgets)
        guided_budgets = list(self.cfg.approaches.guided.budgets)

        # Normalization functions
        self.norms = {
            "cmaes": mcolors.Normalize(
                vmin=min(cmaes_budgets), vmax=max(cmaes_budgets)
            ),
            "guided": mcolors.Normalize(
                vmin=np.log2(min(guided_budgets)), vmax=np.log2(max(guided_budgets))
            ),
            "unguided": mcolors.Normalize(
                vmin=np.log2(min(guided_budgets)), vmax=np.log2(max(guided_budgets))
            ),
        }

        # Line colors (darker end of colormaps)
        self.line_colors = {
            "cmaes": self.cmaps["cmaes"](self.norms["cmaes"](25.0)),
            "guided": self.cmaps["guided"](self.norms["guided"](np.log2(16.0))),
            "unguided": self.cmaps["unguided"](self.norms["unguided"](np.log2(16.0))),
            "random": self.cfg.approaches.random.color,
        }

    def get_budget_color(self, approach: str, budget: int) -> Any:
        """Get color for a specific approach and budget."""
        if approach == "cmaes":
            return self.cmaps["cmaes"](self.norms["cmaes"](budget))
        elif approach in ("guided", "unguided"):
            return self.cmaps[approach](self.norms[approach](np.log2(budget)))
        return self.cfg.approaches.random.color

    def create_full_figure(self) -> tuple[Figure, np.ndarray]:
        """Create the full multi-panel figure."""
        # Organize experiments by (design_space, traj_type) -> list of (reward_fn, exp_name)
        row_definitions = self._get_row_definitions()
        experiments_by_row = self._organize_experiments_by_row(row_definitions)

        num_rows = len(row_definitions)
        num_cols = self.cfg.plot.grid.num_cols

        # Validate that no row has more experiments than columns
        for row_key, exps in experiments_by_row.items():
            if len(exps) > num_cols:
                raise ValueError(
                    f"Row '{row_key}' has {len(exps)} experiments but only "
                    f"{num_cols} columns are available. Experiments: {[e[0] for e in exps]}"
                )

        fig, axes_grid = plt.subplots(
            num_rows,
            num_cols,
            figsize=(self.cfg.plot.figure.width, self.cfg.plot.figure.height),
        )
        axes_2d: NDArray[np.object_] = np.asarray(axes_grid)
        if axes_2d.ndim == 1:
            axes_2d = axes_2d.reshape(1, -1)

        # Plot each row
        for row_idx, row_key in enumerate(row_definitions):
            exps_in_row = experiments_by_row.get(row_key, [])

            for col_idx in range(num_cols):
                ax = axes_2d[row_idx, col_idx]

                if col_idx < len(exps_in_row):
                    reward_fn, exp_name = exps_in_row[col_idx]
                    if exp_name in self.experiments:
                        exp = self.experiments[exp_name]
                        if exp.avg_df.empty:
                            # A partial reproduction shouldn't crash the whole
                            # figure — leave the panel empty instead.
                            logger.warning(
                                f"No runs found for experiment '{exp_name}'; "
                                f"leaving its panel empty"
                            )
                        else:
                            self._plot_experiment(ax, exp)
                    # Set title to reward name (replace underscores with spaces)
                    ax.set_title(reward_fn.replace("_", " "))
                else:
                    # Empty axis - hide it but keep space
                    ax.set_visible(False)

            # Add row label to the leftmost visible axis
            if len(exps_in_row) > 0:
                axes_2d[row_idx, 0].set_ylabel(
                    row_key,
                    fontsize=self.cfg.plot.font.tick_size + 2,
                    fontweight="bold",
                )

        # Clean up individual subplot legends
        for ax in axes_2d.flatten():
            if ax.get_visible() and ax.legend_ is not None:
                ax.legend_.remove()

        # Add unified legend
        self._add_legend(fig)

        plt.tight_layout(rect=(0, 0.0, 1, 0.88), pad=0.2)

        return fig, axes_2d

    def _get_row_definitions(self) -> list[str]:
        """Get the ordered list of row keys (design_space/traj_type)."""
        # Extract unique (design_space, traj_type) combinations from experiments
        row_keys_seen: dict[str, None] = {}  # Use dict to preserve order
        for exp_name in self.cfg.experiments:
            parts = exp_name.split("/")
            if len(parts) >= 3:
                design_space = parts[0]
                traj_type = "multi" if "multi" in parts[2] else "single"
                row_key = f"{design_space}/{traj_type}"
                row_keys_seen[row_key] = None
        return list(row_keys_seen.keys())

    def _organize_experiments_by_row(
        self, row_definitions: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Organize experiments into rows.

        Returns:
            Dict mapping row_key to list of (reward_fn, exp_name) tuples.
        """
        experiments_by_row: dict[str, list[tuple[str, str]]] = {
            row_key: [] for row_key in row_definitions
        }

        for exp_name in self.cfg.experiments:
            parts = exp_name.split("/")
            if len(parts) >= 3:
                design_space = parts[0]
                reward_fn = parts[1]
                traj_type = "multi" if "multi" in parts[2] else "single"
                row_key = f"{design_space}/{traj_type}"

                if row_key in experiments_by_row:
                    experiments_by_row[row_key].append((reward_fn, exp_name))

        return experiments_by_row

    def _plot_experiment(self, ax: Axes, exp: ExperimentData) -> None:
        """Plot a single experiment on an axis."""
        df = exp.avg_df.sort_values(by="avg_optimize_time")
        no_random_df = df[df["approach"] != "random"]

        scatter_kwargs = {
            "linewidths": self.cfg.plot.scatter.linewidths,
            "s": self.cfg.plot.scatter.size,
            "zorder": self.cfg.plot.scatter.zorder,
            "alpha": self.cfg.plot.scatter.alpha,
            "marker": self.cfg.plot.scatter.marker,
        }

        # Plot confidence intervals
        for approach in ["cmaes", "guided", "unguided"]:
            approach_df = no_random_df[no_random_df["approach"] == approach]
            if len(approach_df) > 0:
                ax.fill_between(
                    x=approach_df["avg_optimize_time"],
                    y1=approach_df["reward_ci_low"],
                    y2=approach_df["reward_ci_high"],
                    color=self.line_colors[approach],
                    alpha=0.5,
                    zorder=2,
                )

        # Adjust random line endpoints
        if len(no_random_df) > 0:
            min_time = min(no_random_df["avg_optimize_time"])
            max_time = max(no_random_df["avg_optimize_time"])
            df = df.copy()
            if len(df.loc[df["approach"] == "random", "avg_optimize_time"]) > 0:
                df.loc[df["approach"] == "random", "avg_optimize_time"] = [
                    min_time,
                    max_time,
                ]

        # Plot lines
        sns.lineplot(
            data=df,
            x="avg_optimize_time",
            y="avg_reward",
            hue="approach",
            ax=ax,
            legend=True,
            linestyle="solid",
            alpha=1.0,
            palette=self.line_colors,
            zorder=3,
        )

        # Plot scatter points for each approach
        for approach in ["cmaes", "guided", "unguided"]:
            approach_df = no_random_df[no_random_df["approach"] == approach]
            if len(approach_df) > 0:
                colors = [
                    self.get_budget_color(approach, b)
                    for b in approach_df["compute_budget"]
                ]
                ax.scatter(
                    x=approach_df["avg_optimize_time"],
                    y=approach_df["avg_reward"],
                    c=colors,
                    edgecolors=self.line_colors[approach],
                    **scatter_kwargs,
                )

        # Configure axis (title is set by caller)
        ax.set_xlabel("")
        ax.set_xscale("log")

        # Set tick marks
        design_space = exp.name.split("/")[0]
        is_multitraj = "multi_traj" in exp.name
        plot_times_key = (
            f"{design_space}/{'multi_traj' if is_multitraj else 'single_traj'}"
        )
        plot_times = self.cfg.plot_times[plot_times_key]
        ax.set_xticks(plot_times)
        ax.set_xticklabels([f"{int(t)}" for t in plot_times])
        ax.tick_params(
            axis="both", which="major", labelsize=self.cfg.plot.font.tick_size
        )

    def _add_legend(self, fig: Figure) -> Axes:
        """Add unified legend to the figure."""
        legend_width = self.cfg.plot.legend.width_fraction
        legend_ax = fig.add_axes(
            (
                (1 - legend_width) / 2,
                self.cfg.plot.legend.y_position,
                legend_width,
                0.03,
            )
        )
        legend_ax.set_xlim(0, 120)
        legend_ax.set_ylim(0, 1)

        # Style legend axis
        legend_ax.patch.set_facecolor("white")
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        for spine in legend_ax.spines.values():
            spine.set_visible(False)

        y_pos = 0.5
        dot_spacing = self.cfg.plot.legend.dot_spacing
        dot_size = self.cfg.plot.legend.dot_size
        x_pos = 2

        # Random entry
        legend_ax.plot(
            [x_pos, x_pos + 3],
            [y_pos, y_pos],
            "--",
            color=self.cfg.approaches.random.color,
            linewidth=1.5,
        )
        legend_ax.text(
            x_pos + 5,
            y_pos,
            "random",
            va="center",
            ha="left",
            fontsize=self.cfg.plot.font.legend_size,
        )
        x_pos += 14

        # CMA-ES entry
        legend_ax.plot(
            [x_pos, x_pos + 3],
            [y_pos, y_pos],
            "--",
            color=self.line_colors["cmaes"],
            linewidth=1.5,
        )
        legend_ax.text(
            x_pos + 5,
            y_pos,
            "cma-es with $n$ sim rollouts",
            va="center",
            ha="left",
            fontsize=self.cfg.plot.font.legend_size,
        )
        x_pos += 32

        cmaes_budgets = list(self.cfg.approaches.cmaes.budgets)
        for j, budget in enumerate(cmaes_budgets):
            color = self.get_budget_color("cmaes", budget)
            legend_ax.scatter(
                x_pos + j * dot_spacing,
                y_pos,
                c=[color],
                s=dot_size,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            text_color = (
                "white"
                if budget >= self.cfg.plot.legend.dark_text_threshold.cmaes
                else "black"
            )
            legend_ax.text(
                x_pos + j * dot_spacing,
                y_pos,
                str(budget),
                ha="center",
                va="center",
                fontsize=self.cfg.plot.font.budget_label_size,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

        # Guided entry
        x_pos += len(cmaes_budgets) * dot_spacing + 2
        legend_ax.plot(
            [x_pos, x_pos + 3],
            [y_pos, y_pos],
            "--",
            color=self.line_colors["guided"],
            linewidth=1.5,
        )
        legend_ax.text(
            x_pos + 5,
            y_pos,
            "DSG (ours) with $n$ samples",
            va="center",
            ha="left",
            fontsize=self.cfg.plot.font.legend_size,
        )
        x_pos += 32

        guided_budgets = list(self.cfg.approaches.guided.budgets)
        for j, budget in enumerate(guided_budgets):
            # Use blues colormap for the legend (matching unguided in plot)
            color = self.cmaps["unguided"](self.norms["guided"](np.log2(budget)))
            legend_ax.scatter(
                x_pos + j * dot_spacing,
                y_pos,
                c=[color],
                s=dot_size,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            text_color = (
                "white"
                if budget >= self.cfg.plot.legend.dark_text_threshold.guided
                else "black"
            )
            legend_ax.text(
                x_pos + j * dot_spacing,
                y_pos,
                str(budget),
                ha="center",
                va="center",
                fontsize=self.cfg.plot.font.budget_label_size,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

        return legend_ax

    def create_standalone_legend(self) -> Figure:
        """Create a standalone legend figure."""
        fig = plt.figure(figsize=(12, 0.5))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1)
        ax.patch.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        y_pos = 0.5
        dot_spacing = self.cfg.plot.legend.dot_spacing
        dot_size = self.cfg.plot.legend.dot_size
        x_pos = 2

        # Random entry
        ax.plot(
            [x_pos, x_pos + 3],
            [y_pos, y_pos],
            "--",
            color=self.cfg.approaches.random.color,
            linewidth=1.5,
        )
        ax.text(
            x_pos + 5,
            y_pos,
            "random",
            va="center",
            ha="left",
            fontsize=self.cfg.plot.font.legend_size,
        )
        x_pos += 14

        # CMA-ES entry
        ax.plot(
            [x_pos, x_pos + 3],
            [y_pos, y_pos],
            "--",
            color=self.line_colors["cmaes"],
            linewidth=1.5,
        )
        ax.text(
            x_pos + 5,
            y_pos,
            "cma-es with $n$ sim rollouts",
            va="center",
            ha="left",
            fontsize=self.cfg.plot.font.legend_size,
        )
        x_pos += 32

        cmaes_budgets = list(self.cfg.approaches.cmaes.budgets)
        for j, budget in enumerate(cmaes_budgets):
            color = self.get_budget_color("cmaes", budget)
            ax.scatter(
                x_pos + j * dot_spacing,
                y_pos,
                c=[color],
                s=dot_size,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            text_color = (
                "white"
                if budget >= self.cfg.plot.legend.dark_text_threshold.cmaes
                else "black"
            )
            ax.text(
                x_pos + j * dot_spacing,
                y_pos,
                str(budget),
                ha="center",
                va="center",
                fontsize=self.cfg.plot.font.budget_label_size,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

        # Guided entry
        x_pos += len(cmaes_budgets) * dot_spacing + 2
        ax.plot(
            [x_pos, x_pos + 3],
            [y_pos, y_pos],
            "--",
            color=self.line_colors["guided"],
            linewidth=1.5,
        )
        ax.text(
            x_pos + 5,
            y_pos,
            "DSG (ours) with $n$ samples",
            va="center",
            ha="left",
            fontsize=self.cfg.plot.font.legend_size,
        )
        x_pos += 32

        guided_budgets = list(self.cfg.approaches.guided.budgets)
        for j, budget in enumerate(guided_budgets):
            color = self.cmaps["unguided"](self.norms["guided"](np.log2(budget)))
            ax.scatter(
                x_pos + j * dot_spacing,
                y_pos,
                c=[color],
                s=dot_size,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            text_color = (
                "white"
                if budget >= self.cfg.plot.legend.dark_text_threshold.guided
                else "black"
            )
            ax.text(
                x_pos + j * dot_spacing,
                y_pos,
                str(budget),
                ha="center",
                va="center",
                fontsize=self.cfg.plot.font.budget_label_size,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

        return fig

    def create_individual_axis(self, exp_name: str) -> Figure | None:
        """Create a standalone figure for a single experiment."""
        if exp_name not in self.experiments:
            return None
        if self.experiments[exp_name].avg_df.empty:
            logger.warning(
                f"No runs found for experiment '{exp_name}'; "
                f"skipping its individual axis"
            )
            return None
        if exp_name.startswith("bimanual"):
            fig_size = (4.7, 2.6)
        elif exp_name.startswith("quadruped"):
            fig_size = (3.5, 2.6)
        else:
            fig_size = (3, 2.6)
        fig, ax = plt.subplots(figsize=fig_size)
        self._plot_experiment(ax, self.experiments[exp_name])

        # Set title for standalone figure (reward_fn with spaces)
        parts = exp_name.split("/")
        if len(parts) >= 2:
            reward_fn = parts[1]
            ax.set_title(reward_fn.replace("_", " "))
        else:
            ax.set_title(exp_name)

        # Remove legend from individual plot
        if ax.legend_ is not None:
            ax.legend_.remove()

        plt.tight_layout()
        return fig


# =============================================================================
# Output Management
# =============================================================================


class OutputManager:
    """Manages saving plots in various formats."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_full_figure(self, fig: Figure) -> None:
        """Save the full figure in all configured formats."""
        for fmt in self.cfg.output.formats:
            path = self.output_dir / f"hardware_opt_results.{fmt}"
            fig.savefig(path, dpi=self.cfg.plot.figure.dpi, bbox_inches="tight")
            logger.info(f"Saved full figure: {path}")

    def save_individual_axes(
        self, plotter: HardwareOptPlotter, exp_names: list[str]
    ) -> None:
        """Save each experiment as an individual SVG."""
        if not self.cfg.output.save_individual_axes:
            return

        axes_dir = self.output_dir / "individual_axes"
        axes_dir.mkdir(exist_ok=True)

        for exp_name in exp_names:
            fig = plotter.create_individual_axis(exp_name)
            if fig is not None:
                # Create safe filename
                safe_name = exp_name.replace("/", "_")
                path = axes_dir / f"{safe_name}.svg"
                fig.savefig(path, dpi=self.cfg.plot.figure.dpi, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved individual axis: {path}")

    def save_legend(self, plotter: HardwareOptPlotter) -> None:
        """Save the legend as a standalone SVG."""
        if not self.cfg.output.save_legend:
            return

        fig = plotter.create_standalone_legend()
        path = self.output_dir / "legend.svg"
        fig.savefig(path, dpi=self.cfg.plot.figure.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved legend: {path}")

    def save_cache(self, experiments: dict[str, ExperimentData]) -> None:
        """Save parsed data to cache files."""
        cache_dir = Path(self.cfg.paths.cache_dir or self.cfg.paths.output_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Convert to pickle-friendly format
        plot_df_avg = {name: exp.avg_df for name, exp in experiments.items()}
        plot_dfs = {name: exp.raw_df for name, exp in experiments.items()}

        with open(cache_dir / "plot_df_avg.pkl", "wb") as f:
            pickle.dump(plot_df_avg, f)
        with open(cache_dir / "plot_dfs.pkl", "wb") as f:
            pickle.dump(plot_dfs, f)

        logger.info(f"Saved cache to {cache_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(
    config_path="../config",
    config_name="plot_hardware_opt",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for hardware optimization plotting."""
    logger.info("Starting hardware optimization plotting")
    logger.info(f"Output directory: {cfg.paths.output_dir}")

    # Parse results
    logger.info("Parsing experiment results...")
    parser = ResultsParser(cfg)
    experiments = parser.parse_all()

    # Report what was found
    for exp_name, exp in experiments.items():
        n_approaches = (
            len(exp.avg_df["approach"].unique()) if len(exp.avg_df) > 0 else 0
        )
        n_points = len(exp.avg_df)
        logger.info(f"  {exp_name}: {n_points} data points, {n_approaches} approaches")

    # Create output manager
    output_manager = OutputManager(cfg)
    output_manager.save_cache(experiments)

    # Save detailed parsing log
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parser.parse_log.save(output_dir / "parsing_log.txt")

    # Create plotter and generate figures
    logger.info("Generating plots...")
    plotter = HardwareOptPlotter(cfg, experiments)

    # Full figure
    fig, _ = plotter.create_full_figure()
    output_manager.save_full_figure(fig)
    plt.close(fig)

    # Individual axes
    output_manager.save_individual_axes(plotter, list(cfg.experiments))

    # Legend
    output_manager.save_legend(plotter)

    logger.info("Done!")


if __name__ == "__main__":
    main()
