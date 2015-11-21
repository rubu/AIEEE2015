#include "..\Shared\TestRun.hpp"
#include "nvEncodeAPI.h"
#include "cuda.h"

#include <cstdlib>
#include <string>
#include <map>

#define CHECK_CUDA_DRV_STATUS(x) { CUresult result = (x); if(result != CUDA_SUCCESS) throw std::runtime_error(std::string("CUDA error ").append(std::to_string(result))); }
#define CHECK_NVENC_STATUS(x) { NVENCSTATUS result = (x); if (result != NV_ENC_SUCCESS) throw std::runtime_error(std::string("NVENC error ").append(std::to_string(result))); }
//#define ASYNCHRONOUS

#include "CudaRgb32ToNv12.hpp"

#pragma comment(lib, "cuda.lib")

extern "C"
{
#include <libswscale\swscale.h>
}

#pragma comment(lib, "swscale.lib")

class CNvidiaNvencCodecContext : public CCodecContextBase
{
public:
	CNvidiaNvencCodecContext(unsigned int nWidth, unsigned int nHeight, unsigned int nFps, bool bSaveOutputToFile, unsigned int nBitrate, bool bUsePageLockedIntermediateBuffer, bool bUseSwscaleInsteadOfCuda) : CCodecContextBase(nWidth, nHeight, nFps, bSaveOutputToFile), m_nBitrate(nBitrate), m_bUsePageLockedIntermediateBuffer(bUsePageLockedIntermediateBuffer), m_bUseSwscaleInsteadOfCuda(bUseSwscaleInsteadOfCuda)
	{
	}

	unsigned int GetBitrate() const
	{
		return m_nBitrate;
	}

	bool GetUsePageLockedIntermediateBuffer()
	{
		return m_bUsePageLockedIntermediateBuffer;
	}

	bool GetUseSwscaleInsteadOfCuda()
	{
		return m_bUseSwscaleInsteadOfCuda;
	}

private:
	unsigned int m_nBitrate;
	bool m_bUsePageLockedIntermediateBuffer;
	bool m_bUseSwscaleInsteadOfCuda;
};

class CNvidiaNvencCodec
{
public:
	typedef NVENCSTATUS(NVENCAPI* PNVENCODEAPICREATEINSTANCE)(NV_ENCODE_API_FUNCTION_LIST *);

	CNvidiaNvencCodec(DWORD nCodecInstanceId, const CCodecContextBase& CodecContext) : m_NvidiaNvencCodecContext(static_cast<const CNvidiaNvencCodecContext&>(CodecContext)), m_hNvEncodeAPI64(LoadLibraryA("nvEncodeAPI64.dll"))
	{
		PNVENCODEAPICREATEINSTANCE pNvEncodeAPICreateInstance = reinterpret_cast<PNVENCODEAPICREATEINSTANCE>(GetProcAddress(m_hNvEncodeAPI64, "NvEncodeAPICreateInstance"));
		memset(&m_FunctionList, 0, sizeof(m_FunctionList));
		m_FunctionList.version = NV_ENCODE_API_FUNCTION_LIST_VER;
		NVENCSTATUS nStatus = pNvEncodeAPICreateInstance(&m_FunctionList);
		CHECK_CUDA_DRV_STATUS(cuCtxCreate(&m_Context, 0, 0));
		if (m_NvidiaNvencCodecContext.GetUseSwscaleInsteadOfCuda())
		{
			CHECK_CUDA_DRV_STATUS(cuMemAlloc(&m_pNv12Buffer, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 3 / 2));
			m_nNv12BufferPitch = m_NvidiaNvencCodecContext.GetWidth();
			CHECK_CUDA_DRV_STATUS(cuMemAllocHost(&m_pPageLockedNv12Buffer, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 3 / 2));
			m_pNv12Planes[0] = reinterpret_cast<unsigned char*>(m_pPageLockedNv12Buffer);
			m_pNv12Planes[1] = reinterpret_cast<unsigned char*>(m_pPageLockedNv12Buffer) + m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight();
			m_pNv12Strides[0] = m_NvidiaNvencCodecContext.GetWidth();
			m_pNv12Strides[1] = m_NvidiaNvencCodecContext.GetWidth();
			m_SwscaleContext = sws_getContext(m_NvidiaNvencCodecContext.GetWidth(), m_NvidiaNvencCodecContext.GetHeight(), AV_PIX_FMT_BGR32, m_NvidiaNvencCodecContext.GetWidth(), m_NvidiaNvencCodecContext.GetHeight(), AV_PIX_FMT_NV12, 0, 0, 0, 0);
		}
		else
		{
			CHECK_CUDA_DRV_STATUS(cuMemAllocPitch(&m_pNv12Buffer, &m_nNv12BufferPitch, m_NvidiaNvencCodecContext.GetWidth(), m_NvidiaNvencCodecContext.GetHeight() * 3 / 2, 16));
			if (m_NvidiaNvencCodecContext.GetUsePageLockedIntermediateBuffer())
			{
				CHECK_CUDA_DRV_STATUS(cuMemAllocHost(&m_pPageLockedRgb32Buffer, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 4));
			}
			CHECK_CUDA_DRV_STATUS(cuMemAlloc(&m_pRgb32Buffer, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 4));
		}
		CHECK_CUDA_DRV_STATUS(cuStreamCreate(&m_Stream, 0));
		NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS SessionParameters;
		memset(&SessionParameters, 0, sizeof(SessionParameters));
		SessionParameters.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
		SessionParameters.apiVersion = NVENCAPI_VERSION;
		SessionParameters.device = m_Context;
		SessionParameters.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
		nStatus = m_FunctionList.nvEncOpenEncodeSessionEx(&SessionParameters, &m_pEncoder);
		m_PictureParameters.version = NV_ENC_PIC_PARAMS_VER;
		auto PresetGuid = NV_ENC_PRESET_HP_GUID;
		NV_ENC_PRESET_CONFIG PresetConfiguration = { NV_ENC_PRESET_CONFIG_VER, 0 };
		PresetConfiguration.presetCfg.version = NV_ENC_CONFIG_VER;
		CHECK_NVENC_STATUS(m_FunctionList.nvEncGetEncodePresetConfig(m_pEncoder, NV_ENC_CODEC_H264_GUID, PresetGuid, &PresetConfiguration));
		NV_ENC_CONFIG EncoderConfiguration = { NV_ENC_CONFIG_VER, 0 };
		EncoderConfiguration = PresetConfiguration.presetCfg;
		EncoderConfiguration.gopLength = NVENC_INFINITE_GOPLENGTH;
		EncoderConfiguration.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;
		EncoderConfiguration.frameIntervalP = 1; // No B frames
		EncoderConfiguration.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
		EncoderConfiguration.encodeCodecConfig.h264Config.idrPeriod = m_NvidiaNvencCodecContext.GetFrameCount();
		EncoderConfiguration.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
		EncoderConfiguration.encodeCodecConfig.h264Config.sliceMode = 0;
		EncoderConfiguration.encodeCodecConfig.h264Config.sliceModeData = 0;
		NV_ENC_INITIALIZE_PARAMS InitializationParameters = { NV_ENC_INITIALIZE_PARAMS_VER, 0 };
		InitializationParameters.encodeGUID = NV_ENC_CODEC_H264_GUID;
		InitializationParameters.presetGUID = PresetGuid;
		InitializationParameters.frameRateNum = m_NvidiaNvencCodecContext.GetFps();
		InitializationParameters.frameRateDen = 1;
#ifdef ASYNCHRONOUS
		InitializationParameters.enableEncodeAsync = 1;
#else
		InitializationParameters.enableEncodeAsync = 0;
#endif
		InitializationParameters.enablePTD = 1; // Let the encoder decide the picture type
		InitializationParameters.reportSliceOffsets = 0;
		InitializationParameters.maxEncodeWidth = m_NvidiaNvencCodecContext.GetWidth();
		InitializationParameters.maxEncodeHeight = m_NvidiaNvencCodecContext.GetHeight();
		InitializationParameters.encodeConfig = &EncoderConfiguration;
		InitializationParameters.encodeWidth = m_NvidiaNvencCodecContext.GetWidth();
		InitializationParameters.encodeHeight = m_NvidiaNvencCodecContext.GetHeight();
		InitializationParameters.darWidth = 16;
		InitializationParameters.darHeight = 9;
		CHECK_NVENC_STATUS(m_FunctionList.nvEncInitializeEncoder(m_pEncoder, &InitializationParameters));
		// Picture parameters that are known ahead of encoding
		m_PictureParameters = { NV_ENC_PIC_PARAMS_VER, 0 };
		m_PictureParameters.codecPicParams.h264PicParams.sliceMode = 0;
		m_PictureParameters.codecPicParams.h264PicParams.sliceModeData = 0;
		m_PictureParameters.inputWidth = m_NvidiaNvencCodecContext.GetWidth();
		m_PictureParameters.inputHeight = m_NvidiaNvencCodecContext.GetHeight();
		m_PictureParameters.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
		m_PictureParameters.inputPitch = static_cast<uint32_t>(m_nNv12BufferPitch);
		m_PictureParameters.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
#ifdef ASYNCHRONOUS
		m_hCompletionEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		m_EventParameters = { NV_ENC_EVENT_PARAMS_VER, 0 };
		m_EventParameters.completionEvent = m_hCompletionEvent;
		CHECK_NVENC_STATUS(m_FunctionList.nvEncRegisterAsyncEvent(m_pEncoder, &m_EventParameters));
		m_PictureParameters.completionEvent = m_hCompletionEvent;
#endif
		// Register CUDA input pointer
		NV_ENC_REGISTER_RESOURCE RegisterResource = { NV_ENC_REGISTER_RESOURCE_VER, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, m_NvidiaNvencCodecContext.GetWidth(), m_NvidiaNvencCodecContext.GetHeight(), static_cast<uint32_t>(m_nNv12BufferPitch), 0, reinterpret_cast<void*>(m_pNv12Buffer), NULL, NV_ENC_BUFFER_FORMAT_NV12_PL };
		CHECK_NVENC_STATUS(m_FunctionList.nvEncRegisterResource(m_pEncoder, &RegisterResource));
		NV_ENC_MAP_INPUT_RESOURCE MapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER, 0, 0, RegisterResource.registeredResource };
		m_pRegisteredResource = RegisterResource.registeredResource;
		CHECK_NVENC_STATUS(m_FunctionList.nvEncMapInputResource(m_pEncoder, &MapInputResource));
		m_PictureParameters.inputBuffer = MapInputResource.mappedResource;
		// Create output bitstream buffer
		m_nOutputBitstreamSize = 2 * 1024 * 1024;
		NV_ENC_CREATE_BITSTREAM_BUFFER CreateBitstreamBuffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER, m_nOutputBitstreamSize, NV_ENC_MEMORY_HEAP_AUTOSELECT, 0 };
		CHECK_NVENC_STATUS(m_FunctionList.nvEncCreateBitstreamBuffer(m_pEncoder, &CreateBitstreamBuffer));
		m_pOutputBitstream = CreateBitstreamBuffer.bitstreamBuffer;
		m_PictureParameters.outputBitstream = m_pOutputBitstream;
		if (m_NvidiaNvencCodecContext.GetSaveOutputToFile())
		{
			char pOutputFilename[MAX_PATH];
			sprintf_s(pOutputFilename, "nvenc-%d.h264", nCodecInstanceId);
			if (fopen_s(&m_pOutputFile, pOutputFilename, "wb") != 0)
			{
				throw std::runtime_error(std::string("could not open ").append(pOutputFilename).append(" for writing!"));
			}
		}
	}

	~CNvidiaNvencCodec()
	{
		if (m_NvidiaNvencCodecContext.GetSaveOutputToFile())
		{
			fclose(m_pOutputFile);
		}
#ifdef ASYNCHRONOUS
		CHECK_NVENC_STATUS(m_FunctionList.nvEncUnregisterAsyncEvent(m_pEncoder, &m_EventParameters));
#endif
		CHECK_NVENC_STATUS(m_FunctionList.nvEncUnmapInputResource(m_pEncoder, m_PictureParameters.inputBuffer));
		CHECK_NVENC_STATUS(m_FunctionList.nvEncUnregisterResource(m_pEncoder, m_pRegisteredResource));
		CHECK_NVENC_STATUS(m_FunctionList.nvEncDestroyBitstreamBuffer(m_pEncoder, m_pOutputBitstream));
		CHECK_NVENC_STATUS(m_FunctionList.nvEncDestroyEncoder(m_pEncoder));
		CHECK_CUDA_DRV_STATUS(cuStreamDestroy(m_Stream));
		if (m_NvidiaNvencCodecContext.GetUseSwscaleInsteadOfCuda())
		{
			cudaFree(m_pPageLockedNv12Buffer);
		}
		else
		{
			cudaFree(m_pPageLockedRgb32Buffer);
			CHECK_CUDA_DRV_STATUS(cuMemFree(m_pRgb32Buffer));
		}
		CHECK_CUDA_DRV_STATUS(cuMemFree(m_pNv12Buffer));
		CHECK_CUDA_DRV_STATUS(cuCtxDestroy(m_Context));
		FreeModule(m_hNvEncodeAPI64);
	}

	void Initialize()
	{
		cuCtxSetCurrent(m_Context);
		// Encode a dummy frame since it seems that some initialization is done on the first encoding
		m_PictureParameters.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
#ifdef ASYNCHRONOUS
		// Sanity check
		_ASSERT(WaitForSingleObject(m_PictureParameters.completionEvent, 0) == WAIT_TIMEOUT);
#endif
		CHECK_NVENC_STATUS(m_FunctionList.nvEncEncodePicture(m_pEncoder, &m_PictureParameters));
#ifdef ASYNCHRONOUS
		DWORD nWaitResult = WaitForSingleObject(m_PictureParameters.completionEvent, INFINITE);
		// Sanity check
		_ASSERT(nWaitResult == WAIT_OBJECT_0);
#endif
		NV_ENC_LOCK_BITSTREAM LockBitstream = { NV_ENC_LOCK_BITSTREAM_VER, 0 };
		LockBitstream.sliceOffsets = NULL;
		LockBitstream.outputBitstream = m_PictureParameters.outputBitstream;
		CHECK_NVENC_STATUS(m_FunctionList.nvEncLockBitstream(m_pEncoder, &LockBitstream));
		CHECK_NVENC_STATUS(m_FunctionList.nvEncUnlockBitstream(m_pEncoder, LockBitstream.outputBitstream));
	}

	unsigned int Encode(unsigned char* pFrame, bool bKeyframe)
	{
		unsigned int nSize = 0;
		if (m_NvidiaNvencCodecContext.GetUseSwscaleInsteadOfCuda())
		{
			uint8_t * pRgb32Planes[1] = { pFrame };
			int pRgb32Linesizes[1] = { m_NvidiaNvencCodecContext.GetWidth() * 4 };
			if (sws_scale(m_SwscaleContext, pRgb32Planes, pRgb32Linesizes, 0, m_NvidiaNvencCodecContext.GetHeight(), m_pNv12Planes, m_pNv12Strides) != m_NvidiaNvencCodecContext.GetHeight())
			{
				throw std::runtime_error("sws_scale failed!");
			}
			CHECK_CUDA_DRV_STATUS(cuMemcpyHtoD(m_pNv12Buffer, m_pPageLockedNv12Buffer, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 3 / 2));
		}
		else
		{
			if (m_NvidiaNvencCodecContext.GetUsePageLockedIntermediateBuffer())
			{
				memcpy(m_pPageLockedRgb32Buffer, pFrame, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 4);
				CHECK_CUDA_DRV_STATUS(cuMemcpyHtoDAsync(m_pRgb32Buffer, m_pPageLockedRgb32Buffer, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 4, m_Stream));
				CudaRgba32ToNv12(m_Stream, m_pRgb32Buffer, m_pNv12Buffer, m_nNv12BufferPitch, m_NvidiaNvencCodecContext.GetWidth(), m_NvidiaNvencCodecContext.GetHeight());
			}
			else
			{
				CHECK_CUDA_DRV_STATUS(cuMemcpyHtoD(m_pRgb32Buffer, pFrame, m_NvidiaNvencCodecContext.GetWidth() * m_NvidiaNvencCodecContext.GetHeight() * 4));
			}
			cudaStreamSynchronize(m_Stream);
		}
		m_PictureParameters.encodePicFlags = bKeyframe ? NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS : 0;
#ifdef ASYNCHRONOUS
		// Sanity check
		_ASSERT(WaitForSingleObject(m_PictureParameters.completionEvent, 0) == WAIT_TIMEOUT);
#endif
		CHECK_NVENC_STATUS(m_FunctionList.nvEncEncodePicture(m_pEncoder, &m_PictureParameters));
#ifdef ASYNCHRONOUS
		DWORD nWaitResult = WaitForSingleObject(m_PictureParameters.completionEvent, INFINITE);
		// Sanity check
		_ASSERT(nWaitResult == WAIT_OBJECT_0);
#endif
		NV_ENC_LOCK_BITSTREAM LockBitstream = { NV_ENC_LOCK_BITSTREAM_VER, 0 };
		LockBitstream.sliceOffsets = NULL;
		LockBitstream.outputBitstream = m_PictureParameters.outputBitstream;
		CHECK_NVENC_STATUS(m_FunctionList.nvEncLockBitstream(m_pEncoder, &LockBitstream));
		nSize = LockBitstream.bitstreamSizeInBytes;
		if (m_NvidiaNvencCodecContext.GetSaveOutputToFile())
		{
			fwrite(LockBitstream.bitstreamBufferPtr, 1, LockBitstream.bitstreamSizeInBytes, m_pOutputFile);
		}
		CHECK_NVENC_STATUS(m_FunctionList.nvEncUnlockBitstream(m_pEncoder, LockBitstream.outputBitstream));
		return nSize;
	}

private:
	CNvidiaNvencCodecContext m_NvidiaNvencCodecContext;
	HMODULE m_hNvEncodeAPI64;
	CUcontext m_Context;
	NV_ENCODE_API_FUNCTION_LIST m_FunctionList;
	void* m_pEncoder = nullptr;
	NV_ENC_PIC_PARAMS m_PictureParameters;
	CUstream m_Stream;
	CUdeviceptr m_pRgb32Buffer;
	size_t m_nRgb32BufferPitch;
	CUdeviceptr m_pNv12Buffer;
	size_t m_nNv12BufferPitch;
	NV_ENC_REGISTERED_PTR m_pRegisteredResource;
	NV_ENC_OUTPUT_PTR m_pOutputBitstream;
	uint32_t m_nOutputBitstreamSize;
	FILE* m_pOutputFile = nullptr;
	void* m_pPageLockedRgb32Buffer = nullptr;
	void* m_pPageLockedNv12Buffer = nullptr;
	SwsContext* m_SwscaleContext;
	uint8_t* m_pNv12Planes[2];
	int m_pNv12Strides[2];
#ifdef ASYNCHRONOUS
	HANDLE m_hCompletionEvent = NULL;
	NV_ENC_EVENT_PARAMS m_EventParameters;
#endif
};

int main(int argc, char** argv)
{
	try
	{
		if (argc != 8)
		{
			std::cout << "Usage: NvidiaNvencTest.exe <input filename> <width> <height> <threads> <fps> <bitrate per stream> <save output to file>" << std::endl;
			return EXIT_FAILURE;
		}
		cuInit(0);
		unsigned int nWidth = std::stoi(argv[2]), nHeight = std::stoi(argv[3]), nThreads = std::stoi(argv[4]), nFps = std::stoi(argv[5]), nBitrate = std::stoi(argv[6]);
		bool bSaveOutputToFile = strcmp("true", argv[7]) == 0;
		std::cout << "Input " << argv[1] << " - " << nWidth << "x" << nHeight << ", threads: " << nThreads << ", fps: " << nFps << ", bitrate: " << nBitrate << " bit/s per stream, save output to file: " << bSaveOutputToFile << std::endl;
		CTestRun<CNvidiaNvencCodec> TestRun(argv[1], nThreads, CNvidiaNvencCodecContext(nWidth, nHeight, nFps, bSaveOutputToFile, nBitrate, true, false));
	}
	catch (std::exception& Exception)
	{
		std::cout << "Error: " << Exception.what() << std::endl;
	}
	return EXIT_SUCCESS;
}