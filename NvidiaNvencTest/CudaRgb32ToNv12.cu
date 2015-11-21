#include "CudaRgb32ToNv12.hpp"

#include <cstdio>

__device__ __forceinline__ void RGBToY(const unsigned char r, const unsigned char g, const unsigned char b, unsigned char& y)
{
	y = ((65 * r + 128 * g + 24 * b + 128) >> 8) + 16;
	//y = static_cast<unsigned char>(((int)(30 * r) + (int)(59 * g) + (int)(11 * b)) / 100);
}

__device__ __forceinline__ void RGBToYUV(const unsigned char r, const unsigned char g, const unsigned char b, unsigned char& y, unsigned char& u, unsigned char& v)
{
	RGBToY(r, g, b, y);
	u = ((-37 * r - 74 * g + 112 * b + 128) >> 8) + 128;
	//u = static_cast<unsigned char>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
	v = ((112 * r - 93 * g - 18 * b + 128) >> 8) + 128;
	//v = static_cast<unsigned char>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
}

__global__ void Rgba32ToNv12(CUdeviceptr pRgb32Frame, CUdeviceptr pNv12Frame, size_t nNv12Pitch, unsigned int nImageWidth, unsigned int nImageHeight)
{
	const int nPositionX = (blockIdx.x*blockDim.x + threadIdx.x) * 2;
	const int nPositionY = (blockIdx.y*blockDim.y + threadIdx.y) * 2;
	
	if (nPositionY >= nImageHeight || nPositionX >= nImageWidth)
	{
		// FIXME: calculate the block & grid size correctly
		//printf("thread %d:%d done\n", nPositionX, nPositionY);
		return;
	}
	uchar4* pPixel = reinterpret_cast<uchar4*>(pRgb32Frame)+ nImageWidth * nPositionY + nPositionX;
	unsigned char* pYPlane = reinterpret_cast<unsigned char*>(pNv12Frame) + nNv12Pitch * nPositionY + nPositionX;
	unsigned char* pUVPlane = reinterpret_cast<unsigned char*>(pNv12Frame) + nNv12Pitch * nImageHeight + (nNv12Pitch * nPositionY) / 2 + nPositionX;
	unsigned char nY, nU, nV;

	RGBToY(pPixel->x, pPixel->y, pPixel->z, nY);
	*(pYPlane) =  nY;

	RGBToY((pPixel + 1)->x, (pPixel + 1)->y, (pPixel + 1)->z, nY);
	*(pYPlane + 1) = nY;

	RGBToY((pPixel + nImageWidth)->x, (pPixel + nImageWidth)->y, (pPixel + nImageWidth)->z, nY);
	*(pYPlane + nNv12Pitch) = nY;

	RGBToYUV((pPixel + nImageWidth + 1)->x, (pPixel + nImageWidth + 1)->y, (pPixel + nImageWidth + 1)->z, nY, nU, nV);
	*(pYPlane + nNv12Pitch + 1) = nY;
	*pUVPlane = nU;
	*(pUVPlane + 1) = nV;
	//printf("thread %d:%d done\n", nPositionX, nPositionY);
}

static inline unsigned int DivideAndRoundUp(unsigned int nNumerator, unsigned int nDenominator)
{
	return (nNumerator + nDenominator - 1) / nDenominator;
}

void CudaRgba32ToNv12(CUstream Stream, CUdeviceptr pRgb32Frame, CUdeviceptr pNv12Frame, size_t nNv12Pitch, unsigned int nImageWidth, unsigned int nImageHeight)
{
	const dim3 Block(16, 16);
	const dim3 Grid(DivideAndRoundUp(nImageWidth, 32), DivideAndRoundUp(nImageHeight, 32));
	Rgba32ToNv12 << < Grid, Block, 0, Stream >> >(pRgb32Frame, pNv12Frame, nNv12Pitch, nImageWidth, nImageHeight);
}