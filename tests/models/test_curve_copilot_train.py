from __future__ import annotations

from pathlib import Path

import gc

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
    compute_loss,
    create_optimizer_and_scheduler,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train,
    train_one_epoch,
    validate,
)
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
                data=np.stack([s.target_keyframe for s in samples]),
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
                "query_time",
                data=np.array([s.query_time for s in samples], dtype=np.float32),
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

        loss, metrics = compute_loss(prediction, confidence, target, weights, confidence_targets)

        assert loss.shape == ()
        assert loss.requires_grad
        assert "loss/total" in metrics
        assert "loss/value" in metrics
        assert "loss/tangent" in metrics

    def test_zero_loss_for_perfect_prediction(self) -> None:
        target = torch.tensor([[0.5, 1.0, 0.1, 0.2, 0.3, 0.4]])
        prediction = torch.tensor([[1.0, 0.1, 0.2, 0.3, 0.4, 100.0]])
        confidence = torch.ones(1, 1)
        confidence_targets = torch.ones(1, 1)

        _, metrics = compute_loss(prediction, confidence, target, LossWeights(), confidence_targets)

        assert metrics["loss/value"] < 1e-6
        assert metrics["loss/tangent"] < 1e-6
        assert metrics["loss/confidence"] < 1e-6


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
            metrics = train_one_epoch(
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
            metrics = train_one_epoch(
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
        save_checkpoint(model, optimizer, scheduler, epoch=1, metrics={"loss/val": 0.5},
                        path=checkpoint_path)

        assert checkpoint_path.exists()

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        assert loaded["epoch"] == 1
        assert loaded["metrics"]["loss/val"] == 0.5

        model2 = CurveCopilotModel(config)
        model2.load_state_dict(loaded["model_state_dict"])

        inputs = {
            "context_keyframes": torch.randn(1, 8, 6),
            "property_type": torch.zeros(1, dtype=torch.long),
            "topology_features": torch.randn(1, 6),
            "bone_name_tokens": torch.zeros(1, 32, dtype=torch.long),
            "query_time": torch.tensor([0.5]),
        }

        model.eval()
        model2.eval()
        with torch.no_grad():
            pred1, conf1 = model(**inputs)
            pred2, conf2 = model2(**inputs)

        assert torch.allclose(pred1, pred2, atol=1e-6)
        assert torch.allclose(conf1, conf2, atol=1e-6)


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
        assert config.data.val_split == "val"
        assert config.output.checkpoint_dir == "runs/curve_copilot"
        assert config.model.use_expert_mixing is True
        assert config.model.num_experts == 3
        assert config.model.use_pae is True
        assert config.model.pae_window_size == 64
        assert config.model.pae_latent_channels == 5
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
        assert len(optimizer_state_before["param_groups"]) == len(optimizer_state_after["param_groups"])

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
            metrics = train_one_epoch(
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
