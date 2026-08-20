import asyncio, time, torch

async def ticker():
    for i in range(20):
        print(f"tick {i} at {time.time():.2f}")
        await asyncio.sleep(0.05)

async def fake_inference_v1():
    """你的方案:sleep(0) + 直接 .item()"""
    x = torch.randn(4096, 4096, device='mps')
    for _ in range(10):
        y = x @ x  # enqueue heavy GPU work
        await asyncio.sleep(0)
        _ = y.sum().item()  # block

async def fake_inference_v2():
    """to_thread 方案"""
    x = torch.randn(4096, 4096, device='mps')
    for _ in range(10):
        y = x @ x
        _ = await asyncio.to_thread(y.sum().item)

async def main():
    await asyncio.gather(ticker(), fake_inference_v1())
    # 看 ticker 在 v1 期间几乎完全停 tick
    
    # await asyncio.gather(ticker(), fake_inference_v2())
    # 看 ticker 在 v2 期间持续 tick


if __name__ == "__main__":
    asyncio.run(main())
