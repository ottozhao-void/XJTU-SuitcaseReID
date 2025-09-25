#!/usr/bin/env python
"""
Test script to verify that the NCCL timeout fixes work correctly.
This script will run a minimal distributed training setup to test synchronization.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    test_distributed_fix.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist

# Set environment variables early
os.environ.setdefault("NCCL_TIMEOUT", "300")  # 5 minutes
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1000000")

def setup_distributed():
    """Initialize distributed training"""
    if not dist.is_available():
        print("Distributed training not available")
        return False
        
    # Get rank information
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Initializing process group: rank={local_rank}, world_size={world_size}")
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with timeout
    timeout = torch.distributed.distributed_c10d.timedelta(minutes=5)
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        timeout=timeout
    )
    
    print(f"Rank {local_rank}: Process group initialized successfully")
    return True

def test_collective_operations():
    """Test NCCL collective operations that were causing timeouts"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}: Testing collective operations...")
    
    # Test barrier synchronization
    print(f"Rank {rank}: Testing barrier...")
    dist.barrier()
    print(f"Rank {rank}: Barrier passed")
    
    # Test all_gather (the operation that was timing out)
    print(f"Rank {rank}: Testing all_gather...")
    
    # Create a tensor similar to what was causing issues
    # The error showed NumelIn=1271808, so let's test with a similar size
    tensor_size = 1271808 // world_size  # Divide by world size for testing
    test_tensor = torch.randn(tensor_size, device=f'cuda:{rank}')
    
    # Gather tensors from all processes
    gather_list = [torch.empty_like(test_tensor) for _ in range(world_size)]
    
    start_time = time.time()
    dist.all_gather(gather_list, test_tensor)
    end_time = time.time()
    
    print(f"Rank {rank}: All_gather completed in {end_time - start_time:.2f} seconds")
    
    # Final barrier
    dist.barrier()
    print(f"Rank {rank}: All tests completed successfully!")

def cleanup():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"Process group destroyed")

def main():
    print("Starting distributed training test...")
    
    try:
        if not setup_distributed():
            print("Failed to setup distributed training")
            return
            
        test_collective_operations()
        
        print("All distributed tests passed successfully!")
        
    except Exception as e:
        print(f"Error during distributed test: {e}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    main()