from __future__ import annotations


def compute_topology_features(
    joint_names: list[str],
    parent_indices: list[int],
) -> dict[str, list[float]]:
    num_joints = len(joint_names)

    children: list[list[int]] = [[] for _ in range(num_joints)]
    for i, parent in enumerate(parent_indices):
        if parent >= 0:
            children[parent].append(i)

    depths = _compute_depths(parent_indices, num_joints)
    max_depth = max(depths) if depths else 1

    chain_lengths = _compute_chain_lengths_to_leaf(children, num_joints)
    max_chain = max(chain_lengths) if chain_lengths else 1

    max_children = max((len(c) for c in children), default=1)
    max_siblings = _compute_max_siblings(parent_indices, children, num_joints)

    result: dict[str, list[float]] = {}
    for i, name in enumerate(joint_names):
        parent = parent_indices[i]
        siblings = children[parent] if parent >= 0 else [i]
        sibling_index = siblings.index(i) if i in siblings else 0

        hierarchy_depth = depths[i] / max(max_depth, 1)
        child_count = len(children[i]) / max(max_children, 1)
        sibling_idx_norm = sibling_index / max(max_siblings - 1, 1) if max_siblings > 1 else 0.0
        chain_to_leaf = chain_lengths[i] / max(max_chain, 1)
        parent_child_count = len(children[parent]) / max(max_children, 1) if parent >= 0 else 0.0
        is_leaf = 1.0 if len(children[i]) == 0 else 0.0

        result[name] = [
            hierarchy_depth,
            child_count,
            sibling_idx_norm,
            chain_to_leaf,
            parent_child_count,
            is_leaf,
        ]

    return result


def _compute_depths(parent_indices: list[int], num_joints: int) -> list[int]:
    depths = [0] * num_joints
    for i in range(num_joints):
        d = 0
        current = i
        while parent_indices[current] >= 0:
            d += 1
            current = parent_indices[current]
        depths[i] = d
    return depths


def _compute_chain_lengths_to_leaf(
    children: list[list[int]],
    num_joints: int,
) -> list[int]:
    lengths = [0] * num_joints

    for i in reversed(range(num_joints)):
        if not children[i]:
            lengths[i] = 0
        else:
            lengths[i] = 1 + max(lengths[c] for c in children[i])

    return lengths


def _compute_max_siblings(
    parent_indices: list[int],
    children: list[list[int]],
    num_joints: int,
) -> int:
    max_sibs = 1
    for i in range(num_joints):
        parent = parent_indices[i]
        if parent >= 0:
            max_sibs = max(max_sibs, len(children[parent]))
    return max_sibs
