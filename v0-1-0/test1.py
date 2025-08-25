# sanity_tensor_backend.py
import numpy as np
import torch as th

from znet.autograd.tensor import Tensor  # your rewritten torch-backed Tensor

def expected_default_device():
    if th.cuda.is_available():
        return th.device("cuda:0")
    if th.backends.mps.is_available() and th.backends.mps.is_built():
        return th.device("mps")
    return th.device("cpu")

def check(msg, cond):
    print(f"[{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise AssertionError(msg)

def main():
    print("=== Torch backend sanity check ===")
    print(f"torch version: {th.__version__}")
    print(f"cuda available: {th.cuda.is_available()}")
    print(f"mps available: {th.backends.mps.is_available() and th.backends.mps.is_built()}")

    # 1) Construction & defaults
    t = Tensor([1, 2, 3], dtype=np.float32)  # default auto-device
    print(f"default device picked: {t.device}")
    check("data is torch.Tensor", isinstance(t.data, th.Tensor))
    check("torch autograd OFF", (t.data.requires_grad is False) and (t.data.grad_fn is None))
    check("dtype is float32", t.dtype == th.float32)
    check("shape is (3,)", t.shape == (3,))
    check("default device logic", t.device == expected_default_device())

    # 2) Construct from a torch tensor that *has* requires_grad=True (must detach)
    base = th.tensor([10.0, 20.0], requires_grad=True)
    t2 = Tensor(base)
    print("OK")
    check("constructor detaches from requires_grad=True",
          t2.data.requires_grad is False and t2.data.grad_fn is None)

    # 3) numpy() roundtrip (no autograd, moves to CPU internally)
    arr = t.numpy()
    check("numpy() returns ndarray", isinstance(arr, np.ndarray))
    check("numpy() content match", np.allclose(arr, np.array([1, 2, 3], dtype=np.float32)))

    # 4) .item() for scalar
    s = Tensor(3.5, dtype="float32")
    check("item() returns float", isinstance(s.item(), float))
    check("item() value", abs(s.item() - 3.5) < 1e-6)

    # 5) dtype mappings
    check("dtype string 'float64'", Tensor([1], dtype="float64").dtype == th.float64)
    check("dtype numpy float16", Tensor([1], dtype=np.float16).dtype == th.float16)
    check("dtype torch.float16", Tensor([1], dtype=th.float16).dtype == th.float16)

    # 6) Device transfers (.to, .to_, .cpu/.cuda/.mps)
    # start on whatever default is; then hop around where possible
    t_cpu = t.cpu()
    check(".cpu() moves to CPU", t_cpu.device.type == "cpu")

    if th.cuda.is_available():
        t_cuda = t.to("cuda")
        check(".to('cuda') works", t_cuda.device.type == "cuda")
        t_cuda_inplace = t.to_("cuda")
        check(".to_('cuda') in-place works", t.device.type == "cuda")
        # move back to CPU for later tests
        t.to_("cpu")
        check("moved back to CPU", t.device.type == "cpu")

    if th.backends.mps.is_available() and th.backends.mps.is_built():
        t_mps = t.to("mps")
        check(".to('mps') works", t_mps.device.type == "mps")
        t_mps_inplace = t.to_("mps")
        check(".to_('mps') in-place works", t.device.type == "mps")
        # move back to CPU for later tests
        t.to_("cpu")
        check("moved back to CPU", t.device.type == "cpu")

    # 7) .to(copy=...) aliasing behavior when dtype/device unchanged
    x = Tensor([1, 2, 3], dtype="float32").to("cpu")  # ensure CPU for in-place ops
    y_same = x.to(device=x.device, dtype=x.dtype)      # may alias storage
    y_copy = x.to(device=x.device, dtype=x.dtype, copy=True)  # forced clone

    # mutate y_same; x should change if it's an alias
    y_same.data.add_(1)
    check("to() without copy may alias", th.allclose(x.data, th.tensor([2., 3., 4.])))

    # mutate y_copy; x should NOT change
    before = x.data.clone()
    y_copy.data.add_(10)
    check("to(copy=True) clones", th.allclose(x.data, before))

    print("=== All sanity checks passed ✅ ===")

if __name__ == "__main__":
    main()
