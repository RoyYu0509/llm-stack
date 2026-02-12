import torch
import torch.nn as nn

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 1024
IN_DIM = 1024
HIDDEN = 4096
OUT_DIM = 1024


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x):
        with torch.autograd.profiler.record_function("MLP_forward"):
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
        return x


def run_once(model, optim):
    x = torch.randn(BATCH, IN_DIM, device=DEVICE)
    target = torch.zeros(BATCH, OUT_DIM, device=DEVICE)

    with torch.autograd.profiler.record_function("forward"):
        out = model(x)
        loss = (out - target).pow(2).mean()

    with torch.autograd.profiler.record_function("backward"):
        loss.backward()

    with torch.autograd.profiler.record_function("optimizer_step"):
        optim.step()
        optim.zero_grad(set_to_none=True)


def main():
    print("Using device:", DEVICE)
    model = MLP().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # warmup
    for _ in range(5):
        run_once(model, optim)
        torch.mps.synchronize()

    # actual profiling run (Instruments will capture this)
    for _ in range(1000):
        run_once(model, optim)
        torch.mps.synchronize()


if __name__ == "__main__":
    print("Using device:", DEVICE)
    main()
