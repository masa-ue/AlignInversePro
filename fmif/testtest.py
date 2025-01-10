import torch

# Define your function
def gpu_function(x, y):
    return x * y

# Create data for parallel operations
inputs = [
    (torch.tensor([1.0], device='cuda'), torch.tensor([2.0], device='cuda')),
    (torch.tensor([3.0], device='cuda'), torch.tensor([4.0], device='cuda')),
]

# Create CUDA streams
streams = [torch.cuda.Stream() for _ in range(len(inputs))]

# Launch operations in parallel
results = []
for i, (x, y) in enumerate(inputs):
    with torch.cuda.stream(streams[i]):
        results.append(gpu_function(x, y))

# Synchronize all streams
torch.cuda.synchronize()

# Gather results
results = [result.item() for result in results]
print("Results:", results)