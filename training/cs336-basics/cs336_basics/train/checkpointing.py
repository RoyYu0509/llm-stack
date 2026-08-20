import torch
import os
import wandb

def save_checkpoint(model, optimizer, iteration, out): 
    """
    Save the states of `model`, `optimizer` and `iteration` into
    one file. 

    Inputs:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    torch.save(
        {"model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration},
        f = out
    )

def load_checkpoint(src, model, optimizer):
    """
    Load a checkpoint from cs336_basics, then recover the model and optimizer 
    states from that checkpoint.

    Inputs:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
    """
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iter"]



def save_checkpoint_and_log(model, optimizer, iteration, out, wandb_run):
    """Save to local & Log to WanDB"""
    # 1) Save locally
    save_checkpoint(model, optimizer, iteration, out)

    # 2) Wrap in an artifact
    artifact = wandb.Artifact(
        name="transformer-lm",   # logical name of this model family
        type="model",
        metadata={"iter": iteration},
    )
    artifact.add_file(out, name=os.path.basename(out))

    # 3) Log artifact with aliases
    wandb_run.log_artifact(
        artifact,
        aliases=[f"iter-{iteration}", "latest"],  # "latest" will keep moving
    )
