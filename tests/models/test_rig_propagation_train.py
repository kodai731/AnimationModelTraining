from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from anim_ml.data.rig_data_generator import MAX_EDGES, MAX_JOINTS
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

NUM_TEST_JOINTS = 20
PARENT_INDICES = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    9, 9, 9,
    10, 11, 12,
    14, 15,
    16, 17,
]


def _create_test_hdf5(tmp_path: Path, num_samples: int = 32) -> Path:
    hdf5_path = tmp_path / "test_rig.h5"
    rng = np.random.default_rng(42)

    src_list: list[int] = []
    tgt_list: list[int] = []
    for child, parent in enumerate(PARENT_INDICES):
        if parent == -1:
            continue
        src_list.extend([parent, child])
        tgt_list.extend([child, parent])
    num_real_edges = len(src_list)

    with h5py.File(hdf5_path, "w") as f:
        for split_name in ("train", "val"):
            grp = f.create_group(split_name)

            joint_features = np.zeros((num_samples, MAX_JOINTS, 9), dtype=np.float32)
            joint_features[:, :NUM_TEST_JOINTS, :] = rng.standard_normal(
                (num_samples, NUM_TEST_JOINTS, 9),
            ).astype(np.float32)

            topology_features = np.zeros((num_samples, MAX_JOINTS, 6), dtype=np.float32)
            topology_features[:, :NUM_TEST_JOINTS, :] = rng.standard_normal(
                (num_samples, NUM_TEST_JOINTS, 6),
            ).astype(np.float32)

            bone_name_tokens = np.zeros((num_samples, MAX_JOINTS, 32), dtype=np.int64)
            bone_name_tokens[:, :NUM_TEST_JOINTS, :] = rng.integers(
                0, 44, (num_samples, NUM_TEST_JOINTS, 32),
            ).astype(np.int64)

            joint_mask = np.zeros((num_samples, MAX_JOINTS), dtype=np.float32)
            joint_mask[:, :NUM_TEST_JOINTS] = 1.0

            target_deltas = np.zeros((num_samples, MAX_JOINTS, 4), dtype=np.float32)
            raw_deltas = rng.standard_normal((num_samples, NUM_TEST_JOINTS, 4)).astype(np.float32)
            norms = np.linalg.norm(raw_deltas, axis=-1, keepdims=True)
            target_deltas[:, :NUM_TEST_JOINTS, :] = raw_deltas / np.maximum(norms, 1e-8)

            confidence_targets = np.zeros((num_samples, MAX_JOINTS), dtype=np.float32)
            confidence_targets[:, :NUM_TEST_JOINTS] = rng.choice(
                [0.0, 1.0], size=(num_samples, NUM_TEST_JOINTS),
            ).astype(np.float32)

            source_indices = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)
            target_indices = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)
            edge_direction = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)
            edge_mask_arr = np.zeros((num_samples, MAX_EDGES), dtype=np.int64)

            for i in range(num_samples):
                source_indices[i, :num_real_edges] = src_list
                target_indices[i, :num_real_edges] = tgt_list
                for e in range(num_real_edges):
                    s, t = src_list[e], tgt_list[e]
                    edge_direction[i, e] = 0 if PARENT_INDICES[t] == s else 1
                edge_mask_arr[i, :num_real_edges] = 1

            grp.create_dataset("joint_features", data=joint_features)
            grp.create_dataset("topology_features", data=topology_features)
            grp.create_dataset("bone_name_tokens", data=bone_name_tokens)
            grp.create_dataset("joint_mask", data=joint_mask)
            grp.create_dataset("target_deltas", data=target_deltas)
            grp.create_dataset("confidence_targets", data=confidence_targets)
            grp.create_dataset("source_indices", data=source_indices)
            grp.create_dataset("target_indices", data=target_indices)
            grp.create_dataset("edge_direction", data=edge_direction)
            grp.create_dataset("edge_mask", data=edge_mask_arr)

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
        rotation_deltas = torch.randn(4, MAX_JOINTS, 4, requires_grad=True)
        norms = torch.norm(rotation_deltas, dim=-1, keepdim=True).detach()
        rotation_deltas = rotation_deltas / norms.clamp(min=1e-8)

        confidence = torch.rand(4, MAX_JOINTS, 1, requires_grad=True)
        target_deltas = torch.randn(4, MAX_JOINTS, 4)
        target_norms = torch.norm(target_deltas, dim=-1, keepdim=True)
        target_deltas = target_deltas / target_norms.clamp(min=1e-8)
        confidence_targets = torch.randint(0, 2, (4, MAX_JOINTS)).float()
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

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 32), shuffle=True, num_workers=0,
        )

        config = _make_train_config()
        config.training.epochs = 50
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999

        model = RigPropagationModel(config.model)
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

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 32), shuffle=True, num_workers=0,
        )

        config = _make_train_config()
        config.training.learning_rate = 5e-3
        config.output.log_every_steps = 999
        model = RigPropagationModel(config.model)
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


@pytest.mark.unit
class TestConfigLoading:
    def test_load_yaml(self) -> None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "rig_propagation.yaml"
        config = load_config(config_path)

        assert config.model.max_joints == 64
        assert config.model.max_edges == 126
        assert config.model.node_feature_dim == 128
        assert config.model.num_message_passing_layers == 4
        assert config.model.input_feature_dim == 9
        assert config.model.vocab_size == 64
        assert config.model.bone_context_dim == 64
        assert config.training.batch_size == 128
        assert config.training.learning_rate == 1e-4
        assert config.training.loss_weights.rotation == 1.0
        assert config.training.loss_weights.confidence == 0.2
        assert config.data.val_split == "val"
        assert config.output.checkpoint_dir == "runs/rig_propagation"
        assert len(config.data.train_files) >= 1
        assert all(f.endswith(".h5") for f in config.data.train_files)


@pytest.mark.unit
class TestValidate:
    def test_returns_val_loss(self, tmp_path: Path) -> None:
        hdf5_path = _create_test_hdf5(tmp_path)
        dataset = RigPropagationDataset([hdf5_path], split="val")

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset, batch_size=min(len(dataset), 32), num_workers=0,
        )

        config = _make_train_config()
        model = RigPropagationModel(config.model)
        device = torch.device("cpu")

        metrics = validate(model, loader, config, device)

        dataset.close()
        assert "loss/val" in metrics
        assert metrics["loss/val"] > 0
