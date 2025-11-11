#!/bin/bash

set -ex

# Check for macOS and adjust Ray configuration
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected: Starting Ray without vLLM-specific settings"
    echo "   VLLM_USE_V1=1 is not needed on macOS"
    ray stop -v --force --grace-period 60
    ps aux
    env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 ray start --head --dashboard-host=0.0.0.0
else
    echo "🚀 Linux/NVIDIA system detected: Starting Ray with vLLM support"
    ray stop -v --force --grace-period 60
    ps aux
    env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0
fi
