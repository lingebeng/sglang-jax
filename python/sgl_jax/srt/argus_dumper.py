"""Argus dump lifecycle for sglang-jax inference debugging.

Controlled via environment variables:
  ARGUS_DUMP_DIR          — output directory (required to activate)
  ARGUS_DUMP_SPEC         — layer spec, default "*:output"
  ARGUS_DUMP_MAX_REQUESTS — auto-finalize after N forward passes, default 1
"""

import logging
import os

logger = logging.getLogger(__name__)

_active = False
_step = 0
_mesh_shape = None
_mesh_coord = None
_global_rank = 0


def setup_argus_dump(global_rank=0, mesh_shape=None, mesh_coord=None):
    dump_dir = os.environ.get("ARGUS_DUMP_DIR")
    dump_spec = os.environ.get("ARGUS_DUMP_SPEC", "*:output")
    if not dump_dir:
        return

    # TP causes jax.debug.callback to fire once per device, each writing
    # the same shard names — allow overwrites instead of raising FATAL.
    os.environ.setdefault("ARGUS_DUPLICATE_POLICY", "WARNING")

    from argus.core.registry import DumpRegistry
    from argus.jax.saver import init_writer

    DumpRegistry.parse(dump_spec)
    init_writer(dump_dir, global_rank, mesh_shape=mesh_shape, mesh_coord=mesh_coord)

    global _active, _mesh_shape, _mesh_coord, _global_rank
    _active = True
    _mesh_shape = mesh_shape
    _mesh_coord = mesh_coord
    _global_rank = global_rank
    logger.info("Argus dump active: dir=%s spec=%s", dump_dir, dump_spec)


def advance_step(forward_batch=None):
    global _step
    _step += 1

    from argus.jax.saver import set_name_prefix

    parts = [f"step_{_step}"]
    if forward_batch is not None:
        parts.append(f"bs{forward_batch.batch_size}")
        parts.append(str(forward_batch.forward_mode.name))
    set_name_prefix("/".join(parts))


def finalize_argus_dump():
    global _active
    if not _active:
        return

    import yaml

    from argus.jax.saver import _require_writer

    writer = _require_writer()
    writer.finalize()
    metadata = {
        "schema_version": 1,
        "framework": "jax",
        "global_rank": _global_rank,
        "mesh": {
            "shape": _mesh_shape or {"dp": 1, "tp": 1},
            "coord": _mesh_coord or {"dp": 0, "tp": 0},
        },
        "dtype_map": writer.dtype_map,
        "tensor_layout": {},
    }
    with open(f"{writer.rank_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    logger.info("Argus dump finalized: %s", writer.rank_dir)
    _active = False


def is_argus_active():
    return _active
