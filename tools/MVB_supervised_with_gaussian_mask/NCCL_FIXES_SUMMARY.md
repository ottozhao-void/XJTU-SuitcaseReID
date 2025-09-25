# NCCL Timeout Error Fixes - Summary

## Problem Analysis
The error logs showed NCCL ALLGATHER operations timing out after 600 seconds (10 minutes). This indicates distributed training processes were not synchronizing properly during collective operations.

## Root Causes Identified
1. **Incorrect distributed initialization**: Using deprecated `device_id` parameter
2. **Missing NCCL configuration**: No timeout settings or debug flags
3. **Poor gradient synchronization**: Default DistributedDataParallel settings
4. **Resource contention**: High batch sizes and worker counts
5. **Missing error handling**: No graceful degradation on failures

## Fixes Applied

### 1. Fixed Distributed Initialization (`openunreid/utils/dist_utils.py`)
- Removed deprecated `device_id` parameter from `init_process_group`
- Added proper timeout configuration (5-10 minutes)
- Added NCCL environment variables for stability:
  - `NCCL_TIMEOUT`: Set to 300 seconds (5 minutes)
  - `NCCL_BLOCKING_WAIT`: Enable blocking wait
  - `NCCL_DEBUG`: Enable debug logging
  - `TORCH_NCCL_TRACE_BUFFER_SIZE`: Enable flight recorder for debugging

### 2. Enhanced NCCL Configuration (Training Script)
- Added comprehensive NCCL environment setup at startup
- Disabled potentially problematic features:
  - `NCCL_IB_DISABLE=1`: Disable InfiniBand
  - `NCCL_P2P_DISABLE=1`: Disable peer-to-peer communication
- Disabled `torch.autograd.set_detect_anomaly()` to prevent hangs

### 3. Improved DistributedDataParallel Settings
- Added `find_unused_parameters=True` for unused gradient handling
- Added `broadcast_buffers=True` for batch norm synchronization
- Added `bucket_cap_mb=25` for better memory management
- Added `gradient_as_bucket_view=True` for efficient gradients

### 4. Reduced Resource Contention
- Reduced batch size from 32 to 16 per GPU (training)
- Reduced batch size from 64 to 32 per GPU (testing)
- Reduced data loader workers from 4 to 2 per GPU

### 5. Added Error Handling and Synchronization
- Added try-catch blocks around distributed operations
- Added explicit synchronization barriers at key points
- Added proper error handling in training step
- Added non-blocking CUDA transfers

### 6. Created Test Script
- `test_distributed_fix.py`: Verifies NCCL collective operations work
- Tests the exact type of operation that was failing (ALLGATHER)
- Provides diagnostic information about distributed setup

## Usage Instructions

### Testing the Fixes
```bash
# Test basic distributed functionality
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nproc_per_node=4 \
tools/MVB_supervised_with_gaussian_mask/test_distributed_fix.py
```

### Running Training with Fixes
```bash
# Run training with fixed distributed setup
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nproc_per_node=4 \
tools/MVB_supervised_with_gaussian_mask/strong_baseline_mvb_with_gaussian.py \
--config tools/MVB_supervised_with_gaussian_mask/mvb_config.yaml \
--launcher pytorch
```

## Expected Results
- No more 10-minute NCCL timeouts
- Better error messages and debugging information
- More stable distributed training
- Faster detection of actual issues (within 5 minutes instead of 10)

## Monitoring and Debugging
- Check NCCL debug output for communication issues
- Monitor GPU memory usage (reduced batch sizes)
- Watch for synchronization messages in logs
- Use flight recorder traces if issues persist

## Additional Recommendations
1. Ensure all GPUs are on the same node for optimal performance
2. Check network connectivity if using multi-node training
3. Monitor system resources (CPU, memory, GPU memory)
4. Consider using `NCCL_SOCKET_IFNAME` to specify network interface if needed

## Rollback Instructions
If issues persist, you can:
1. Revert changes in `dist_utils.py`
2. Remove NCCL environment variables
3. Restore original batch sizes in config
4. Use single-GPU training as fallback