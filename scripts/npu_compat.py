"""
Ascend NPU compatibility layer.
- Monkey-patches torch.cuda → torch.npu
- Patches numpy 1.x ↔ jax 0.11 incompatibilities
- Wraps flax/orbax import failures

MUST be imported before any openpi module.
"""
import logging
import sys
import types

_log = logging.getLogger(__name__)

# ─── numpy 1.x compat for jax 0.11.x ─────────────────────────────
import numpy as np
if np.__version__.startswith('1.'):
    if not hasattr(np.dtypes, 'StringDType'):
        class _StringDType:
            _name = 'StringDType'
            def __init__(self, *a, **kw): pass
            def __repr__(self): return 'StringDType()'
            def __eq__(self, o): return type(o).__name__ == '_StringDType' or isinstance(o, type(self))
            def __hash__(self): return hash('StringDType')
            kind = 'T'
            char = 'T'
        np.dtypes.StringDType = _StringDType

    # Patch jax.errors.JaxRuntimeError (missing in older jax or needed by orbax)
    try:
        import jax.errors
        if not hasattr(jax.errors, 'JaxRuntimeError'):
            jax.errors.JaxRuntimeError = RuntimeError
    except Exception:
        pass

# ─── torch.cuda → torch.npu ──────────────────────────────────────
import torch
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except ImportError:
    _HAS_NPU = False

if _HAS_NPU:
    torch.cuda = torch.npu  # type: ignore[assignment]
    _log.info('[npu_compat] torch.cuda → torch.npu (%d devices)', torch.npu.device_count())
    if not hasattr(torch.backends, 'cudnn') or not hasattr(torch.backends.cudnn, 'benchmark'):
        class _Stub:
            benchmark = False; allow_tf32 = False
            def __setattr__(self, k, v): pass
        torch.backends.cudnn = _Stub()
    if not hasattr(torch.backends.cuda, 'matmul'):
        class _MStub:
            allow_tf32 = False
            def __setattr__(self, k, v): pass
        torch.backends.cuda.matmul = _MStub()
else:
    _log.warning('torch_npu not available')
