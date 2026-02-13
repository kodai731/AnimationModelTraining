from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anim_ml.data.rig_data_generator import build_adjacency
from anim_ml.data.rig_dataset import RigPropagationDataset
from anim_ml.models.rig_propagation.model import RigPropagationConfig, RigPropagationModel
from anim_ml.models.rig_propagation.train import (
    LossWeights,
    TrainConfig,
    TrainingConfig,
    compute_loss,
    compute_rotation_loss,
    create_optimizer_and_scheduler,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train_one_epoch,
    validate,
)
from anim_ml.utils.skeleton import SMPL_22_PARENT_INDICES


def _create_test_hdf5(tmp_path: Path, num_samples: int = 32) -> Path:
    hdf5_path = tmp_path / "test_rig.h5"
    rng = np.random.default_rng(42)

    adjacency = build_adjacency(SMPL_22_PARENT_INDICES)

    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("adjacency", data=adjacency)

        for split_name in ("train", "val"):
            grp = f.create_group(split_name)

            joint_features = rng.standard_normal((num_samples, 22, 10)).astype(np.float32)
            joint_types = rng.integers(0, 13, (num_samples, 22)).astype(np.int64)

            target_deltas = rng.standard_normal((num_samples, 22, 4)).astype(np.float32)
            norms = np.linalg.norm(target_deltas, axis=-1, keepdims=True)
            target_deltas /= np.maximum(norms, 1e-8)

            confidence_targets = rng.choice([0.0, 1.0], size=(num_samples, 22)).astype(np.float32)

            grp.create_dataset("joint_features", data=joint_features)
            grp.create_dataset("joint_types", data=joint_types)
            grp.create_dataset("target_deltas", data=target_deltas)
            grp.create_dataset("confidence_targets", data=confidence_targets)

    return hdf5_path


def _make_small_config() -> RigPropagationConfig:
    return RigPropagationConfig(
        node_feature_dim=32,
        edge_feature_dim=8,
        hidden_dim=64,
        ffn_dim=128,
        num_message_passing_layers=2,
        dropout=0.0,
    )


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
        rotation_deltas = torch.randn(4, 22, 4, requires_grad=True)
        norms = torch.norm(rotation_deltas, dim=-1, keepdim=True).detach()
        rotation_deltas = rotation_deltas / norms.clamp(min=1e-8)

        confidence = torch.rand(4, 22, 1, requires_grad=True)
        target_deltas = torch.randn(4, 22, 4)
        target_norms = torch.norm(target_deltas, dim=-1, keepdim=True)
        target_deltas = target_deltas / target_norms.clamp(min=1e-8)
        confidence_targets = torch.randint(0, 2, (4, 22)).float()
        weights = LossWeights()

        loss, metrics = compute_loss(
            rotation_deltas, confidence, target_deltas, confidence_targets, weights,
        )

        assert loss.shape == ()
        assert "loss/total" in metrics
        assert "loss/rotation" in metrics
        assert "loss/confidence" in metrics

    def test_geodesic_zero_for_identical_quats(self) -> None:
        quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        loss = compute_rotation_loss(quat, quat)
        assert loss.item() < 1e-6


@pytest.mark.unit
class TestOptimizer:
    def test_warmup_starts_at_zero(self) -> None:
        model = RigPropagationModel(_make_small_config())
        config = TrainingConfig(warmup_epochs=5, epochs=50, learning_rate=1e-3)
        _, scheduler = create_optimizer_and_scheduler(model, config, steps_per_epoch=10)

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr < 1e-6

    def test_lr_increases_during_warmup(self) -> None:
        model = RigPropagationModel(_make_small_config())
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
        dataset = RigPropagationDataset([hdf5_path], split="train")
        adjacency = torch.from_numpy(dataset.adjacency)

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 32), shuffle=True, num_workers=0,
        )

        config = _make_train_config()
        config.training.epochs = 50
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999

        model = RigPropagationModel(config.model, adjacency=adjacency)
        device = torch.device("cpu")

        steps_per_epoch = max(len(loader), 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        first_loss = None
        last_loss = None
        for epoch in range(1, config.training.epochs + 1):
            metrics = train_one_epoch(model, loader, optimizer, scheduler, config, device, epoch)
            if first_loss is None:
                first_loss = metrics["loss/train"]
            last_loss = metrics["loss/train"]

        dataset.close()
        assert last_loss is not None
        assert first_loss is not None
        assert last_loss < first_loss


@pytest.mark.unit
class TestLossDecreases:
    def test_five_epochs(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="train")
        adjacency = torch.from_numpy(dataset.adjacency)

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 32), shuffle=True, num_workers=0,
        )

        config = _make_train_config()
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999
        model = RigPropagationModel(config.model, adjacency=adjacency)
        device = torch.device("cpu")

        steps_per_epoch = max(len(loader), 1)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, config.training, steps_per_epoch,
        )

        epoch_losses = []
        for epoch in range(1, 6):
            metrics = train_one_epoch(model, loader, optimizer, scheduler, config, device, epoch)
            epoch_losses.append(metrics["loss/train"])

        dataset.close()
        assert epoch_losses[-1] < epoch_losses[0]


@pytest.mark.unit
class TestCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        config = _make_small_config()
        model = RigPropagationModel(config)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model, TrainingConfig(), steps_per_epoch=10,
        )

        checkpoint_path = tmp_path / "test_ckpt.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch=1,
            metrics={"loss/val": 0.5}, path=checkpoint_path,
        )

        assert checkpoint_path.exists()

        loaded = load_checkpoint(checkpoint_path, torch.device("cpu"))
        assert loaded["epoch"] == 1
        assert loaded["metrics"]["loss/val"] == 0.5

        model2 = RigPropagationModel(config)
        model2.load_state_dict(loaded["model_state_dict"])

        inputs = {
            "joint_features": torch.randn(1, 22, 10),
            "joint_types": torch.zeros(1, 22, dtype=torch.long),
        }

        model.eval()
        model2.eval()
        with torch.no_grad():
            deltas1, conf1 = model(**inputs)
            deltas2, conf2 = model2(**inputs)

        torch.testing.assert_close(deltas1, deltas2, atol=1e-6, rtol=0)
        torch.testing.assert_close(conf1, conf2, atol=1e-6, rtol=0)


@pytest.mark.unit
class TestConfigLoading:
    def test_load_yaml(self) -> None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "rig_propagation.yaml"
        config = load_config(config_path)

        assert config.model.num_joints == 22
        assert config.model.node_feature_dim == 128
        assert config.model.num_message_passing_layers == 4
        assert config.training.batch_size == 128
        assert config.training.learning_rate == 1e-4
        assert config.training.loss_weights.rotation == 1.0
        assert config.training.loss_weights.confidence == 0.2
        assert config.data.val_split == "val"
        assert config.output.checkpoint_dir == "runs/rig_propagation"


@pytest.mark.unit
class TestValidate:
    def test_returns_val_loss(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="val")
        adjacency = torch.from_numpy(dataset.adjacency)

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 32), num_workers=0,
        )

        config = _make_train_config()
        model = RigPropagationModel(config.model, adjacency=adjacency)
        device = torch.device("cpu")

        metrics = validate(model, loader, config, device)

        dataset.close()
        assert "loss/val" in metrics
        assert metrics["loss/val"] > 0
