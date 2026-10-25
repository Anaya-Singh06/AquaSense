"""Run BEFORE the full pipeline. Exits non-zero unless PyTorch really runs on the NVIDIA GPU."""
import platform, sys
import torch
print("python", platform.python_version(), "| torch", torch.__version__, "| torch CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FAIL: CUDA unavailable (CPU-only torch build or driver problem). Do not start training.")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(8, 30, 7, device="cuda")
net = torch.nn.LSTM(7, 16, 2, batch_first=True).to("cuda")
y, _ = net(x)
assert x.is_cuda and y.is_cuda
print("forward pass OK on", y.device, "shape", tuple(y.shape))
