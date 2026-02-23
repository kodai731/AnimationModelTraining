from __future__ import annotations

import gc
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.curve_dataset import CurveCopilotDataset
from anim_ml.data.curve_extractor import extract_curve_samples
from anim_ml.models.curve_copilot.model import CurveCopilotConfig, CurveCopilotModel
from anim_ml.models.curve_copilot.train import (
    LossWeights,
    TrainConfig,
    TrainingConfig,
    _huber_loss,
    compute_frequency_loss,
    compute_loss,
    compute_multistep_loss,
    create_optimizer_and_scheduler,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train,
    train_one_epoch,
    validate,
)
from anim_ml.utils.checkpoint import CheckpointState, restore_checkpoint_state
from anim_ml.utils.optimizer import DmlAdamW
from anim_ml.utils.preparation_log import PreparationLog

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _create_test_hdf5(tmp_path: Path) -> Path:
    motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
    samples = extract_curve_samples(motion)

    tmp_path.mkdir(parents=True, exist_ok=True)
    hdf5_path = tmp_path / "test_curves.h5"
    with h5py.File(hdf5_path, "w") as f:
        for split_name in ("train", "val"):
            grp = f.create_group(split_name)
            grp.create_dataset(
                "context_keyframes",
                data=np.stack([s.context_keyframes for s in samples]),
            )
            grp.create_dataset(
                "target",
                data=np.stack([s.target_keyframes for s in samples]),
            )
            grp.create_dataset(
                "property_type",
                data=np.array([s.property_type for s in samples], dtype=np.int32),
            )
            grp.create_dataset(
                "topology_features",
                data=np.stack([s.topology_features for s in samples]),
            )
            grp.create_dataset(
                "bone_name_tokens",
                data=np.stack([s.bone_name_tokens for s in samples]),
            )
            grp.create_dataset(
                "query_times",
                data=np.stack([s.query_times for s in samples]),
            )
            grp.create_dataset(
                "curve_window",
                data=np.stack([s.curve_window for s in samples]),
            )
            grp.create_dataset(
                "curve_mean",
                data=np.array([s.curve_mean for s in samples], dtype=np.float32),
            )
            grp.create_dataset(
                "curve_std",
                data=np.array([s.curve_std for s in samples], dtype=np.float32),
            )
            grp.create_dataset(
                "valid_steps",
                data=np.array([s.valid_steps for s in samples], dtype=np.int32),
            )

    return hdf5_path


def _make_small_config() -> CurveCopilotConfig:
    return CurveCopilotConfig(d_model=32, n_heads=2, d_ff=64, n_layers=2, dropout=0.0)


def _make_train_config() -> TrainConfig:
    return TrainConfig(
        model=_make_small_config(),
        training=TrainingConfig(
            batch_size=16,
            epochs=5,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_epochs=1,
            gradient_clip=1.0,
        ),
    )


@pytest.mark.unit
class TestComputeLoss:
    def test_returns_scalar_loss(self) -> None:
        prediction = torch.randn(4, 6, requires_grad=True)
        confidence = torch.rand(4, 1, requires_grad=True)
        target = torch.randn(4, 6)
        weights = LossWeights()
        confidence_targets = torch.rand(4, 1)
        context = torch.randn(4, 8, 6)

        loss, metrics = compute_loss(
            prediction, confidence, target, weights, confidence_targets, context,
        )

        assert loss.shape == ()
        assert loss.requires_grad
        assert "loss/total" in metrics
        assert "loss/value" in metrics
        assert "loss/tangent" in metrics
        assert "loss/smoothness" in metrics
        assert "loss/frequency" in metrics

    def test_zero_loss_for_perfect_prediction(self) -> None:
        target = torch.tensor([[0.5, 1.0, 0.1, 0.2, 0.3, 0.4]])
        prediction = torch.tensor([[1.0, 0.1, 0.2, 0.3, 0.4, 100.0]])
        confidence = torch.ones(1, 1)
        confidence_targets = torch.ones(1, 1)

        _, metrics = compute_loss(
            prediction, confidence, target, LossWeights(), confidence_targets,
        )

        assert metrics["loss/value"] < 1e-6
        assert metrics["loss/tangent"] < 1e-6
        assert metrics["loss/confidence"] < 1e-6


@pytest.mark.unit
class TestComputeMultistepLoss:
    def test_returns_scalar_loss(self) -> None:
        predictions = torch.randn(4, 3, 6, requires_grad=True)
        confidences = torch.rand(4, 3, 1, requires_grad=True)
        targets = torch.randn(4, 3, 6)
        valid_steps = torch.tensor([3, 3, 2, 1])
        context = torch.randn(4, 8, 6)

        loss, metrics = compute_multistep_loss(
            predictions, confidences, targets, valid_steps,
            LossWeights(), context, [1.0, 0.8, 0.6],
        )

        assert loss.shape == ()
        assert loss.requires_grad
        assert "loss/total" in metrics
        assert "step0/loss/total" in metrics
        assert "step1/loss/total" in metrics

    def test_valid_steps_mask(self) -> None:
        predictions = torch.randn(4, 3, 6, requires_grad=True)
        confidences = torch.rand(4, 3, 1, requires_grad=True)
        targets = torch.randn(4, 3, 6)
        context = torch.randn(4, 8, 6)

        all_valid = torch.tensor([3, 3, 3, 3])
        loss_all, _ = compute_multistep_loss(
            predictions, confidences, targets, all_valid,
            LossWeights(), context, [1.0, 0.8, 0.6],
        )

        one_valid = torch.tensor([1, 1, 1, 1])
        loss_one, metrics_one = compute_multistep_loss(
            predictions, confidences, targets, one_valid,
            LossWeights(), context, [1.0, 0.8, 0.6],
        )

        assert "step1/loss/total" not in metrics_one
        assert loss_all.item() != loss_one.item()

    def test_step_decay_weighting(self) -> None:
        predictions = torch.randn(4, 2, 6, requires_grad=True)
        confidences = torch.rand(4, 2, 1, requires_grad=True)
        targets = torch.randn(4, 2, 6)
        valid_steps = torch.tensor([2, 2, 2, 2])
        context = torch.randn(4, 8, 6)

        loss_equal, _ = compute_multistep_loss(
            predictions, confidences, targets, valid_steps,
            LossWeights(), context, [1.0, 1.0],
        )

        loss_decayed, _ = compute_multistep_loss(
            predictions.detach().requires_grad_(True), confidences.detach().requires_grad_(True),
            targets, valid_steps,
            LossWeights(), context, [1.0, 0.5],
        )

        assert loss_equal.item() != loss_decayed.item()

    def test_backward_compatible_metrics(self) -> None:
        predictions = torch.randn(4, 2, 6, requires_grad=True)
        confidences = torch.rand(4, 2, 1, requires_grad=True)
        targets = torch.randn(4, 2, 6)
        valid_steps = torch.tensor([2, 2, 2, 2])
        context = torch.randn(4, 8, 6)

        _, metrics = compute_multistep_loss(
            predictions, confidences, targets, valid_steps,
            LossWeights(), context, [1.0, 0.8],
        )

        assert "loss/value" in metrics
        assert "loss/tangent" in metrics
        assert "loss/smoothness" in metrics


@pytest.mark.unit
class TestSingleStepBackwardCompat:
    def test_single_step_model_output_shape(self) -> None:
        config = CurveCopilotConfig(
            d_model=32, n_heads=2, d_ff=64, n_layers=2, dropout=0.0,
            max_steps=1,
        )
        model = CurveCopilotModel(config)
        model.eval()

        query_times = torch.tensor([0.5])
        with torch.no_grad():
            pred, conf = model(
                torch.randn(1, 8, 6),
                torch.zeros(1, dtype=torch.long),
                torch.randn(1, 6),
                torch.zeros(1, 32, dtype=torch.long),
                query_times,
            )

        assert pred.shape == (1, 1, 6)
        assert conf.shape == (1, 1, 1)

    def test_multistep_model_output_shape(self) -> None:
        config = CurveCopilotConfig(
            d_model=32, n_heads=2, d_ff=64, n_layers=2, dropout=0.0,
            max_steps=3,
        )
        model = CurveCopilotModel(config)
        model.eval()

        query_times = torch.rand(2, 3)
        with torch.no_grad():
            pred, conf = model(
                torch.randn(2, 8, 6),
                torch.zeros(2, dtype=torch.long),
                torch.randn(2, 6),
                torch.zeros(2, 32, dtype=torch.long),
                query_times,
            )

        assert pred.shape == (2, 3, 6)
        assert conf.shape == (2, 3, 1)


@pytest.mark.unit
class TestOptimizer:
    def test_warmup_starts_at_zero(self) -> None:
        model = CurveCopilotModel(_make_small_config())
        config = TrainingConfig(warmup_epochs=5, epochs=50, learning_rate=1e-3)
        _, scheduler = create_optimizer_and_scheduler(model, config, steps_per_epoch=10)

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr < 1e-6

    def test_lr_increases_during_warmup(self) -> None:
        model = CurveCopilotModel(_make_small_config())
        config = TrainingConfig(warmup_epochs=5, epochs=50, learning_rate=1e-3)
        optimizer, scheduler = create_optimizer_and_scheduler(model, config, steps_per_epoch=10)

        lrs = []
        for _ in range(20):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        assert lrs[-1] > lrs[0]


@pytest.mark.unit
class TestOverfitTinyBatch:
    def test_loss_converges(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        config = _make_train_config()
        config.training.batch_size = min(len(dataset), 64)
        config.training.epochs = 50
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999

        model = CurveCopilotModel(config.model)
        device = torch.device("cpu")

        steps_per_epoch = max(len(dataset) // config.training.batch_size, 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        first_loss = None
        last_loss = None
        for epoch in range(1, config.training.epochs + 1):
            metrics, _ = train_one_epoch(
                model, dataset, optimizer, scheduler, config, device, epoch,
            )
            if first_loss is None:
                first_loss = metrics["train/total"]
            last_loss = metrics["train/total"]

        dataset.close()
        assert last_loss is not None
        assert first_loss is not None
        assert last_loss < first_loss


@pytest.mark.unit
class TestLossDecreases:
    def test_five_epochs(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        config = _make_train_config()
        config.training.batch_size = min(len(dataset), 64)
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999
        model = CurveCopilotModel(config.model)
        device = torch.device("cpu")

        steps_per_epoch = max(len(dataset) // config.training.batch_size, 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        epoch_losses = []
        for epoch in range(1, 6):
            metrics, _ = train_one_epoch(
                model, dataset, optimizer, scheduler, config, device, epoch,
            )
            epoch_losses.append(metrics["train/total"])

        dataset.close()
        assert epoch_losses[-1] < epoch_losses[0]


@pytest.mark.unit
class TestCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        config = _make_small_config()
        model = CurveCopilotModel(config)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, TrainingConfig(), steps_per_epoch=10,
        )

        checkpoint_path = tmp_path / "test_ckpt.pt"
        state = CheckpointState(epoch=1, global_step=100, is_mid_epoch=False,
                                best_val_loss=0.5, epochs_without_improvement=0)
        save_checkpoint(model, optimizer, scheduler, state,
                        metrics={"loss/val": 0.5}, path=checkpoint_path)

        assert checkpoint_path.exists()

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        assert loaded["epoch"] == 1
        assert loaded["global_step"] == 100
        assert loaded["is_mid_epoch"] is False
        assert loaded["metrics"]["loss/val"] == 0.5

        model2 = CurveCopilotModel(config)
        model2.load_state_dict(loaded["model_state_dict"])

        inputs = {
            "context_keyframes": torch.randn(1, 8, 6),
            "property_type": torch.zeros(1, dtype=torch.long),
            "topology_features": torch.randn(1, 6),
            "bone_name_tokens": torch.zeros(1, 32, dtype=torch.long),
            "query_times": torch.tensor([0.5]),
        }

        model.eval()
        model2.eval()
        with torch.no_grad():
            pred1, conf1 = model(**inputs)
            pred2, conf2 = model2(**inputs)

        assert torch.allclose(pred1, pred2, atol=1e-6)
        assert torch.allclose(conf1, conf2, atol=1e-6)

    def test_mid_epoch_checkpoint_round_trip(self, tmp_path: Path) -> None:
        config = _make_small_config()
        model = CurveCopilotModel(config)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, TrainingConfig(), steps_per_epoch=10,
        )

        checkpoint_path = tmp_path / "mid_epoch.pt"
        state = CheckpointState(epoch=3, global_step=250, is_mid_epoch=True,
                                best_val_loss=0.42, epochs_without_improvement=2)
        save_checkpoint(model, optimizer, scheduler, state,
                        metrics={}, path=checkpoint_path)

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        restored = restore_checkpoint_state(loaded)

        assert restored.epoch == 3
        assert restored.global_step == 250
        assert restored.is_mid_epoch is True
        assert restored.best_val_loss == pytest.approx(0.42)
        assert restored.epochs_without_improvement == 2

    def test_resume_mid_epoch_restarts_same_epoch(self, tmp_path: Path) -> None:
        config = _make_small_config()
        model = CurveCopilotModel(config)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, TrainingConfig(), steps_per_epoch=10,
        )

        checkpoint_path = tmp_path / "mid.pt"
        state = CheckpointState(epoch=5, global_step=400, is_mid_epoch=True,
                                best_val_loss=0.3, epochs_without_improvement=1)
        save_checkpoint(model, optimizer, scheduler, state,
                        metrics={}, path=checkpoint_path)

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        restored = restore_checkpoint_state(loaded)

        start_epoch = restored.epoch if restored.is_mid_epoch else restored.epoch + 1
        assert start_epoch == 5

    def test_resume_completed_epoch_starts_next(self, tmp_path: Path) -> None:
        config = _make_small_config()
        model = CurveCopilotModel(config)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, TrainingConfig(), steps_per_epoch=10,
        )

        checkpoint_path = tmp_path / "done.pt"
        state = CheckpointState(epoch=5, global_step=500, is_mid_epoch=False,
                                best_val_loss=0.3, epochs_without_improvement=0)
        save_checkpoint(model, optimizer, scheduler, state,
                        metrics={}, path=checkpoint_path)

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        restored = restore_checkpoint_state(loaded)

        start_epoch = restored.epoch if restored.is_mid_epoch else restored.epoch + 1
        assert start_epoch == 6

    def test_backward_compat_old_checkpoint(self, tmp_path: Path) -> None:
        config = _make_small_config()
        model = CurveCopilotModel(config)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, TrainingConfig(), steps_per_epoch=10,
        )

        checkpoint_path = tmp_path / "old_format.pt"
        torch.save({  # pyright: ignore[reportUnknownMemberType]
            "epoch": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": {"loss/val": 0.6},
            "best_val_loss": 0.4,
            "epochs_without_improvement": 3,
        }, checkpoint_path)

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        restored = restore_checkpoint_state(loaded)

        assert restored.epoch == 10
        assert restored.global_step == 0
        assert restored.is_mid_epoch is False
        assert restored.best_val_loss == pytest.approx(0.4)
        assert restored.epochs_without_improvement == 3


@pytest.mark.unit
class TestConfigLoading:
    def test_load_yaml(self) -> None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "curve_copilot.yaml"
        config = load_config(config_path)

        assert config.model.d_model == 128
        assert config.model.n_heads == 4
        assert config.training.batch_size == 2048
        assert config.training.learning_rate == 2.83e-4
        assert config.training.loss_weights.value == 1.0
        assert config.training.loss_weights.frequency == 0.05
        assert config.data.val_split == "val"
        assert config.output.checkpoint_dir == "runs/curve_copilot"
        assert config.model.use_expert_mixing is True
        assert config.model.num_experts == 3
        assert config.model.use_pae is True
        assert config.model.pae_window_size == 64
        assert config.model.pae_latent_channels == 5
        assert config.model.max_steps == 3
        assert config.training.step_decay == [1.0, 0.8, 0.6]
        assert len(config.data.train_files) >= 1
        assert all(f.endswith(".h5") for f in config.data.train_files)


@pytest.mark.unit
class TestMemoryCleanupPreservesState:
    def test_gc_preserves_model_and_optimizer(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        config = _make_train_config()
        config.training.batch_size = min(len(dataset), 64)
        config.output.log_every_steps = 999
        model = CurveCopilotModel(config.model)
        device = torch.device("cpu")

        steps_per_epoch = max(len(dataset) // config.training.batch_size, 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        train_one_epoch(model, dataset, optimizer, scheduler, config, device, 1)

        weights_before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer_state_before = optimizer.state_dict()
        scheduler_lr_before = scheduler.get_last_lr()[0]

        gc.collect()

        weights_after = model.state_dict()
        for key in weights_before:
            assert torch.equal(weights_before[key], weights_after[key]), (
                f"Model weight '{key}' changed after gc.collect()"
            )

        scheduler_lr_after = scheduler.get_last_lr()[0]
        assert scheduler_lr_before == scheduler_lr_after

        optimizer_state_after = optimizer.state_dict()
        before_groups = len(optimizer_state_before["param_groups"])
        after_groups = len(optimizer_state_after["param_groups"])
        assert before_groups == after_groups

        dataset.close()

    def test_training_continues_after_gc(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        config = _make_train_config()
        config.training.batch_size = min(len(dataset), 64)
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999
        model = CurveCopilotModel(config.model)
        device = torch.device("cpu")

        steps_per_epoch = max(len(dataset) // config.training.batch_size, 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        epoch_losses = []
        for epoch in range(1, 6):
            metrics, _ = train_one_epoch(
                model, dataset, optimizer, scheduler, config, device, epoch,
            )
            epoch_losses.append(metrics["train/total"])
            gc.collect()

        dataset.close()
        assert epoch_losses[-1] < epoch_losses[0], (
            f"Loss did not decrease across epochs with gc.collect(): {epoch_losses}"
        )

    def test_dataloader_survives_gc(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 64), shuffle=True, num_workers=0,
        )

        first_batch = next(iter(loader))
        assert "context_keyframes" in first_batch

        gc.collect()

        second_batch = next(iter(loader))
        assert "context_keyframes" in second_batch
        assert first_batch["context_keyframes"].shape == second_batch["context_keyframes"].shape

        dataset.close()

    def test_full_train_with_memory_cleanup(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        config = _make_train_config()
        config.training.epochs = 3
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999
        config.output.save_every_epochs = 1
        config.data.train_files = [str(hdf5_path)]
        config.data.num_workers = 0
        config.output.checkpoint_dir = str(tmp_path / "runs")

        train(config, PreparationLog("curve_copilot_test"), device_override="cpu")

        checkpoint_dir = tmp_path / "runs"
        assert (checkpoint_dir / "last.pt").exists()
        loaded = load_checkpoint(checkpoint_dir / "last.pt", torch.device("cpu"))
        assert loaded["epoch"] == 3


@pytest.mark.unit
class TestFrequencyLoss:
    def test_returns_scalar(self) -> None:
        prediction_value = torch.randn(4)
        target_value = torch.randn(4)
        context = torch.randn(4, 8, 6)

        loss = compute_frequency_loss(prediction_value, target_value, context)

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_zero_for_identical(self) -> None:
        context = torch.randn(4, 8, 6)
        value = torch.randn(4)

        loss = compute_frequency_loss(value, value, context)

        assert loss.item() < 1e-6

    def test_differentiable(self) -> None:
        prediction_value = torch.randn(4, requires_grad=True)
        target_value = torch.randn(4)
        context = torch.randn(4, 8, 6)

        loss = compute_frequency_loss(prediction_value, target_value, context)
        loss.backward()

        assert prediction_value.grad is not None
        assert prediction_value.grad.shape == prediction_value.shape

    def test_short_context(self) -> None:
        prediction_value = torch.randn(4)
        target_value = torch.randn(4)
        context = torch.randn(4, 2, 6)

        loss = compute_frequency_loss(prediction_value, target_value, context)

        assert loss.shape == ()

    def test_low_freq_weighted_more(self) -> None:
        context = torch.zeros(8, 6, 6)
        context[:, :, 1] = torch.linspace(0, 1, 6).unsqueeze(0).expand(8, -1)

        smooth_pred = context[:, -1, 1] + 0.2
        noisy_pred = context[:, -1, 1] + 0.2
        noisy_pred[::2] += 0.5

        target = context[:, -1, 1] + 0.2

        smooth_loss = compute_frequency_loss(smooth_pred, target, context)
        noisy_loss = compute_frequency_loss(noisy_pred, target, context)

        assert smooth_loss.item() <= noisy_loss.item()


@pytest.mark.unit
class TestHuberLoss:
    def test_matches_torch_huber(self) -> None:
        input_t = torch.randn(32)
        target_t = torch.randn(32)

        expected = torch.nn.functional.huber_loss(input_t, target_t, delta=1.0)
        actual = _huber_loss(input_t, target_t, delta=1.0)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_matches_torch_huber_large_delta(self) -> None:
        input_t = torch.randn(32)
        target_t = torch.randn(32)

        expected = torch.nn.functional.huber_loss(input_t, target_t, delta=5.0)
        actual = _huber_loss(input_t, target_t, delta=5.0)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_differentiable(self) -> None:
        input_t = torch.randn(16, requires_grad=True)
        target_t = torch.randn(16)

        loss = _huber_loss(input_t, target_t, delta=1.0)
        loss.backward()

        assert input_t.grad is not None
        assert input_t.grad.shape == input_t.shape

    def test_zero_for_identical(self) -> None:
        value = torch.randn(16)
        loss = _huber_loss(value, value.clone())
        assert loss.item() < 1e-8


DML_BLOCKED_OPS = frozenset({
    "huber_loss",
    "smooth_l1_loss",
    "multinomial",
    "scatter_add",
    "repeat_interleave",
})

DML_UNSUPPORTED_DTYPES = frozenset({
    torch.complex64,
    torch.complex128,
    torch.bfloat16,
})


class _OpRecorder:
    """Record torch op names during a block of code."""

    def __init__(self) -> None:
        self.ops: set[str] = set()

    def __enter__(self) -> _OpRecorder:
        from torch.overrides import TorchFunctionMode

        recorder = self

        class _Mode(TorchFunctionMode):
            def __torch_function__(self_, func, types, args=(), kwargs=None):  # noqa: N805
                name = getattr(func, "__name__", str(func))
                recorder.ops.add(name)
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._mode.__exit__(None, None, None)

    @property
    def blocked(self) -> set[str]:
        return self.ops & DML_BLOCKED_OPS


def _make_full_model_config() -> CurveCopilotConfig:
    return CurveCopilotConfig(
        d_model=32, n_heads=2, d_ff=64, n_layers=2, dropout=0.0,
        num_experts=2, use_expert_mixing=True, enhanced_gating=True,
        use_pae=True, pae_window_size=64, pae_latent_channels=5,
        use_multi_resolution=True, multi_res_branch_dim=16,
        use_phase_detection=True,
    )


def _make_model_inputs(
    batch: int = 4,
    seq_len: int = 8,
    config: CurveCopilotConfig | None = None,
) -> dict[str, torch.Tensor]:
    if config is None:
        config = _make_full_model_config()
    return {
        "context_keyframes": torch.randn(batch, seq_len, 6),
        "property_type": torch.randint(0, config.num_property_types, (batch,)),
        "topology_features": torch.randn(batch, config.topology_dim),
        "bone_name_tokens": torch.randint(0, config.vocab_size, (batch, config.token_length)),
        "query_times": torch.rand(batch, config.max_steps),
        "curve_window": torch.randn(batch, config.pae_window_size),
    }


@pytest.mark.unit
class TestDmlCompatibility:

    def test_model_forward_no_blocked_ops(self) -> None:
        config = _make_full_model_config()
        model = CurveCopilotModel(config)
        model.eval()
        inputs = _make_model_inputs(config=config)

        with _OpRecorder() as rec:
            model(**inputs)

        assert not rec.blocked, f"DML-incompatible ops in forward: {rec.blocked}"

    def test_loss_computation_no_blocked_ops(self) -> None:
        prediction = torch.randn(4, 6)
        confidence = torch.rand(4, 1)
        target = torch.randn(4, 6)
        weights = LossWeights()
        confidence_targets = torch.rand(4, 1)
        context = torch.randn(4, 8, 6)

        with _OpRecorder() as rec:
            compute_loss(
                prediction, confidence, target, weights,
                confidence_targets, context,
            )

        assert not rec.blocked, f"DML-incompatible ops in loss: {rec.blocked}"

    def test_backward_no_blocked_ops(self) -> None:
        config = _make_full_model_config()
        model = CurveCopilotModel(config)
        inputs = _make_model_inputs(config=config)

        predictions, confidences = model(**inputs)
        prediction = predictions[:, 0, :]
        confidence = confidences[:, 0, :]
        target = torch.randn_like(prediction)
        confidence_targets = torch.rand_like(confidence)
        loss, _ = compute_loss(
            prediction, confidence, target, LossWeights(),
            confidence_targets, inputs["context_keyframes"],
        )

        with _OpRecorder() as rec:
            loss.backward()

        assert not rec.blocked, f"DML-incompatible ops in backward: {rec.blocked}"

    def test_optimizer_step_no_blocked_ops(self) -> None:
        config = _make_full_model_config()
        model = CurveCopilotModel(config)
        inputs = _make_model_inputs(config=config)

        optimizer = DmlAdamW(model.parameters(), lr=1e-3)

        predictions, confidences = model(**inputs)
        prediction = predictions[:, 0, :]
        confidence = confidences[:, 0, :]
        target = torch.randn_like(prediction)
        confidence_targets = torch.rand_like(confidence)
        loss, _ = compute_loss(
            prediction, confidence, target, LossWeights(),
            confidence_targets, inputs["context_keyframes"],
        )
        loss.backward()

        with _OpRecorder() as rec:
            optimizer.step()

        assert not rec.blocked, f"DML-incompatible ops in optimizer: {rec.blocked}"

    def test_full_training_step_no_blocked_ops(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="train")
        config = _make_train_config()
        config.training.batch_size = min(len(dataset), 16)
        config.output.log_every_steps = 999
        model = CurveCopilotModel(config.model)
        device = torch.device("cpu")

        steps_per_epoch = max(len(dataset) // config.training.batch_size, 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        with _OpRecorder() as rec:
            train_one_epoch(
                model, dataset, optimizer, scheduler, config, device, epoch=1,
            )

        dataset.close()
        assert not rec.blocked, f"DML-incompatible ops in training step: {rec.blocked}"

    def test_fft_runs_on_cpu_only(self) -> None:
        prediction_value = torch.randn(4)
        target_value = torch.randn(4)
        context = torch.randn(4, 8, 6)

        original_rfft = torch.fft.rfft
        fft_devices: list[str] = []

        def tracking_rfft(
            input: torch.Tensor, *args: object, **kwargs: object,
        ) -> torch.Tensor:
            fft_devices.append(str(input.device))
            return original_rfft(input, *args, **kwargs)

        torch.fft.rfft = tracking_rfft  # type: ignore[assignment]
        try:
            compute_frequency_loss(prediction_value, target_value, context)
        finally:
            torch.fft.rfft = original_rfft  # type: ignore[assignment]

        assert fft_devices, "FFT should be called at least once"
        for dev in fft_devices:
            assert dev == "cpu", f"FFT called on non-CPU device: {dev}"

    def test_no_unsupported_dtypes_in_forward(self) -> None:
        config = _make_full_model_config()
        model = CurveCopilotModel(config)
        model.eval()
        inputs = _make_model_inputs(config=config)

        predictions, confidences = model(**inputs)

        assert predictions.dtype not in DML_UNSUPPORTED_DTYPES
        assert confidences.dtype not in DML_UNSUPPORTED_DTYPES

    def test_no_unsupported_dtypes_in_loss(self) -> None:
        prediction = torch.randn(4, 6, requires_grad=True)
        confidence = torch.rand(4, 1, requires_grad=True)
        target = torch.randn(4, 6)
        weights = LossWeights()
        confidence_targets = torch.rand(4, 1)
        context = torch.randn(4, 8, 6)

        loss, _ = compute_loss(
            prediction, confidence, target, weights,
            confidence_targets, context,
        )

        assert loss.dtype not in DML_UNSUPPORTED_DTYPES
        assert not loss.is_complex()

        loss.backward()
        assert prediction.grad is not None
        assert prediction.grad.dtype not in DML_UNSUPPORTED_DTYPES

    def test_no_unsupported_dtypes_in_gradients(self) -> None:
        config = _make_full_model_config()
        model = CurveCopilotModel(config)
        inputs = _make_model_inputs(config=config)

        predictions, confidences = model(**inputs)
        prediction = predictions[:, 0, :]
        confidence = confidences[:, 0, :]
        target = torch.randn_like(prediction)
        confidence_targets = torch.rand_like(confidence)
        loss, _ = compute_loss(
            prediction, confidence, target, LossWeights(),
            confidence_targets, inputs["context_keyframes"],
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.dtype not in DML_UNSUPPORTED_DTYPES, (
                    f"Unsupported dtype {param.grad.dtype} in gradient of {name}"
                )

    def test_model_without_optional_features_no_blocked_ops(self) -> None:
        config = CurveCopilotConfig(
            d_model=32, n_heads=2, d_ff=64, n_layers=2, dropout=0.0,
            use_expert_mixing=False, use_pae=False,
            use_multi_resolution=False, use_phase_detection=False,
        )
        model = CurveCopilotModel(config)
        inputs = _make_model_inputs(config=config)
        del inputs["curve_window"]

        with _OpRecorder() as rec:
            predictions, confidences = model(**inputs)
            prediction = predictions[:, 0, :]
            confidence = confidences[:, 0, :]
            target = torch.randn_like(prediction)
            confidence_targets = torch.rand_like(confidence)
            loss, _ = compute_loss(
                prediction, confidence, target, LossWeights(),
                confidence_targets, inputs["context_keyframes"],
            )
            loss.backward()

        assert not rec.blocked, f"DML-incompatible ops in minimal model: {rec.blocked}"


@pytest.mark.unit
class TestValidate:
    def test_returns_val_loss(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = CurveCopilotDataset([hdf5_path], split="val")

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 64), num_workers=0,
        )

        config = _make_train_config()
        model = CurveCopilotModel(config.model)
        device = torch.device("cpu")

        metrics = validate(model, loader, config, device)

        dataset.close()
        assert "val/total" in metrics
        assert metrics["val/total"] > 0
