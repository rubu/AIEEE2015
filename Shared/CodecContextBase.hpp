#pragma once

class CCodecContextBase
{
public:
	CCodecContextBase(unsigned int nWidth, unsigned int nHeight, unsigned int nFps, unsigned int nFrameCount) : m_nWidth(nWidth), m_nHeight(nHeight), m_nFps(nFps), m_nFrameCount(nFrameCount)
	{
	}

	unsigned int GetWidth() const
	{
		return m_nWidth;
	}

	unsigned int GetHeight() const
	{
		return m_nHeight;
	}

	unsigned int GetFps() const
	{
		return m_nFps;
	}

	unsigned int GetFrameCount() const
	{
		return m_nFrameCount;
	}
private:
	unsigned int m_nWidth;
	unsigned int m_nHeight;
	unsigned int m_nFps;
	unsigned int m_nFrameCount;
};