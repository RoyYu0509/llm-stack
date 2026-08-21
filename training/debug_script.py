import torch
import torch.nn as nn
from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed
import os
import sys

# Mock torch.distributed calls
import torch.distributed as dist
dist.is_initialized = lambda: True
dist.get_rank = lambda: 0
dist.all_reduce = lambda *args, **kwargs: None
class DistWork:
    def wait(self): pass
dist.all_reduce = lambda x, **kw: DistWork()

# Need to set CUDA device for parameters
if torch.cuda.is_available():
    torch.cuda.set_device(0)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # fc1: 10*10 = 10params
        self.fc1 = nn.Linear(10, 10, bias=False)
        # fc2: 10*1 = 10params
        self.fc2 = nn.Linear(10, 1, bias=False)

def test_bucket_construction():
    print("Testing Bucket Construction...")
    model = SimpleModel()
    if torch.cuda.is_available():
        model.cuda()
    
    # Enable debug logging
    os.environ['DEBUG_DDP'] = '1'
    
    bucket_size_mb = 1.0 # Large enough for 1 bucket
    
    try:
        ddp_model = DDPOverlapBucketed(model, bucket_size_mb)
        
        # Verify flatten_grad order
        print("\n--- Flatten Grad Order ---")
        param_list = list(model.parameters())

        # params[0] is fc1.weight (100)
        # params[1] is fc2.weight (10)
        
        # Let's inspect the `buckets`
        for i, bucket in enumerate(ddp_model.buckets):
            print(f"Bucket {i}: Bound {bucket.bound}")
            
            # Check what parameters are mapped to what bounds
            for p, g in bucket.para_tensor_dict.items():
                size = p.numel()
                # Find index of p in param_list
                idx = -1
                for k, existing_p in enumerate(param_list):
                    if existing_p is p:
                        idx = k
                        break
                print(f"  Contains param index {idx} (size {size})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bucket_construction()
