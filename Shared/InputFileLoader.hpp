#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Windows.h>
#include <sys/stat.h>
#include <sys/types.h>

class CInputFileLoader
{
public:
	CInputFileLoader(const char* pInputFilename, unsigned int nWidth, unsigned int nHeight)
	{
		struct _stat64 FileStatistics;
		if (_stat64(pInputFilename, &FileStatistics) != 0)
		{
			throw std::runtime_error(std::string("could not obtain size of the input file ").append(pInputFilename));
		}
		const unsigned int nFrameSize = nWidth * nHeight * 4;
		if (FileStatistics.st_size % nFrameSize != 0)
		{
			throw std::runtime_error(std::string("the input file ").append(pInputFilename).append(" does not contain an even number of ").append(std::to_string(nWidth)).append("x").append(std::to_string(nHeight)).append(" RGB32 frames"));
		}

		m_nFrameCount = static_cast<unsigned int>(FileStatistics.st_size / nFrameSize);
		FILE* pInputFile = nullptr;
		if (fopen_s(&pInputFile, pInputFilename, "rb") == 0)
		{
			unsigned int nFrameIndex = 0;
			while (nFrameIndex++ < m_nFrameCount)
			{
				std::unique_ptr<unsigned char[]> pFrame(new unsigned char[nFrameSize]);
				if (fread(pFrame.get(), 1, nFrameSize, pInputFile) != nFrameSize)
				{
					throw std::runtime_error(std::string("could not read a full frame from the input file ").append(pInputFilename));
				}
				m_Frames.push_back(std::move(pFrame));
			}
		}
		else
		{
			throw std::runtime_error(std::string("could not open the input file ").append(pInputFilename));
		}
	}

	const std::vector<std::unique_ptr<unsigned char[]>>& GetFrames()
	{
		return m_Frames;
	}

	unsigned int GetFrameCount()
	{
		return m_nFrameCount;
	}

private:
	unsigned int m_nFrameCount = 0;
	std::vector<std::unique_ptr<unsigned char[]>> m_Frames;
};