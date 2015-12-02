#include <cstdlib>
#include <iostream>

#include "..\Shared\TestRun.hpp"
#include "..\Shared\CodecContextBase.hpp"

#include "turbojpeg.h"

#pragma comment(lib, "turbojpeg.lib")

#include <string>

#define CHECK_LIBJPEG(x) { if ((x) == false) { throw std::runtime_error(std::string("libjpeg-turbo error ").append(tjGetErrorStr())); } }

class CLibjegTurboCodecContext : public CCodecContextBase
{
public:
	CLibjegTurboCodecContext(unsigned int nWidth, unsigned int nHeight, unsigned int nFps, bool bSaveOutputToFile, unsigned int nQuality) : CCodecContextBase(nWidth, nHeight, nFps, bSaveOutputToFile), m_nQuality(nQuality)
	{
	}

	unsigned int GetQuality()
	{
		return m_nQuality;
	}

private:
	unsigned int m_nQuality;
};

class CLibjpegTurboCodec
{
public:
	CLibjpegTurboCodec(DWORD nCodecContextId, const CCodecContextBase& CodecContext) : m_LibjpegTurboCodecContext(static_cast<const CLibjegTurboCodecContext&>(CodecContext)), m_nCodecContextId(nCodecContextId), m_Encoder(tjInitTransform()), m_nJpegSize(tjBufSize(m_LibjpegTurboCodecContext.GetWidth(), m_LibjpegTurboCodecContext.GetHeight(), TJSAMP_420))
	{
		CHECK_LIBJPEG(m_Encoder != NULL);
		m_pJpegFrame.reset(new unsigned char[m_nJpegSize]);
	}

	void Initialize()
	{
	}

	unsigned int Encode(unsigned char* pImage, bool)
	{
		unsigned long nJpegSize = m_nJpegSize;
		unsigned char* pJpeg = m_pJpegFrame.get();
		CHECK_LIBJPEG(tjCompress2(m_Encoder, pImage, m_LibjpegTurboCodecContext.GetWidth(), m_LibjpegTurboCodecContext.GetWidth() * 4, m_LibjpegTurboCodecContext.GetHeight(), TJPF_RGBA, &pJpeg, &nJpegSize, TJSAMP_420, m_LibjpegTurboCodecContext.GetQuality(), TJFLAG_NOREALLOC) == 0);
		if (m_LibjpegTurboCodecContext.GetSaveOutputToFile())
		{
			char pFilename[MAX_PATH];
			sprintf_s(pFilename, "libjpeg-turbo-%d-%d.jpg", m_nCodecContextId, m_nFrameIndex++);
			FILE* pOutputFile = nullptr;
			if (fopen_s(&pOutputFile, pFilename, "wb") == 0)
			{
				fwrite(m_pJpegFrame.get(), 1, nJpegSize, pOutputFile);
				fclose(pOutputFile);
			}
		}
		return nJpegSize;
	}

private:
	CLibjegTurboCodecContext m_LibjpegTurboCodecContext;
	tjhandle m_Encoder;
	std::unique_ptr<unsigned char[]> m_pJpegFrame;
	DWORD m_nCodecContextId;
	unsigned int m_nFrameIndex = 0;
	unsigned long m_nJpegSize;
};

int main(int argc, char** argv)
{
	try
	{
		if (argc != 8)
		{
			std::cout << "Usage: NvidiaNvencTest.exe <input filename> <width> <height> <threads> <fps> <quality> <save output to file>" << std::endl;
			return EXIT_FAILURE;
		}
		unsigned int nWidth = std::stoi(argv[2]), nHeight = std::stoi(argv[3]), nThreads = std::stoi(argv[4]), nFps = std::stoi(argv[5]), nQuality = std::stoi(argv[7]);
		bool bSaveOutputToFile = strcmp("true", argv[6]) == 0;
		std::cout << "Input " << argv[1] << " - " << nWidth << "x" << nHeight << ", threads: " << nThreads << ", fps: " << nFps << ", quality: " << nQuality << ", save output to file: " << bSaveOutputToFile << std::endl;
		CCpuUsageMonitor CpuUsageMonitor(L"LibjpegTurboTest");
		CTestRun<CLibjpegTurboCodec> TestRun(argv[1], nThreads, CLibjegTurboCodecContext(nWidth, nHeight, nFps, bSaveOutputToFile, nQuality), CpuUsageMonitor);
	}
	catch (std::exception& Exception)
	{
		std::cout << "Error: " << Exception.what() << std::endl;
	}
	return EXIT_SUCCESS;
}