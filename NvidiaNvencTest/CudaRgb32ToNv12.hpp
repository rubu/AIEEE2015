#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

void CudaRgba32ToNv12(CUstream Stream, CUdeviceptr pRgb32Frame, CUdeviceptr pNv12Frame, size_t nNv12Pitch, unsigned int nImageWidth, unsigned int nImageHeight);