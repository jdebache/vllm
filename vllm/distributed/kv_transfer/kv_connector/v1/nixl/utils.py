# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import errno
import time
from collections.abc import Iterator
from typing import Any

import regex as re
import zmq

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

logger = init_logger(__name__)

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())

# Bind retry policy for the NIXL side-channel ROUTER socket. The port is
# published to remote peers (decoders connect to it explicitly), so on
# EADDRINUSE we cannot fall back to a different port and must wait for the
# kernel to release the previous binding -- e.g. a TCP TIME_WAIT leftover
# from a previously-crashed engine on the same host/port. ~5 minutes total
# is well past the typical TIME_WAIT window.
_ZMQ_BIND_MAX_ATTEMPTS = 60
_ZMQ_BIND_RETRY_BACKOFF_S = 5.0


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket.

    For ROUTER sockets (which ``bind``), retries on ``EADDRINUSE`` to tolerate
    a stale listener (e.g. one still in TCP ``TIME_WAIT`` after a crashed
    engine on the same host/port). REQ sockets ``connect`` and are not
    retried here -- callers handle connect-time failures themselves.
    """

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    bind = socket_type == zmq.ROUTER
    max_attempts = _ZMQ_BIND_MAX_ATTEMPTS if bind else 1

    ctx: zmq.Context | None = None
    sock: zmq.Socket | None = None
    try:
        for attempt in range(1, max_attempts + 1):
            ctx = zmq.Context()  # type: ignore[attr-defined]
            try:
                sock = make_zmq_socket(
                    ctx=ctx,
                    path=addr,
                    socket_type=socket_type,
                    bind=bind,
                )
                break
            except zmq.error.ZMQError as e:
                # Drop the half-initialized context before retrying so we
                # don't leak a context per failed attempt.
                ctx.destroy(linger=0)
                ctx = None
                if (
                    not bind
                    or e.errno != errno.EADDRINUSE
                    or attempt == max_attempts
                ):
                    raise
                logger.warning(
                    "ZMQ bind to %s failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs.",
                    addr,
                    attempt,
                    max_attempts,
                    e,
                    _ZMQ_BIND_RETRY_BACKOFF_S,
                )
                time.sleep(_ZMQ_BIND_RETRY_BACKOFF_S)
        assert sock is not None
        yield sock
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)
