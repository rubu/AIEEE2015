#include <cstdlib>

#include "..\Shared\TestRun.hpp"
#include "..\Shared\CodecContextBase.hpp"

extern "C"
{
#include <libavcodec\avcodec.h>
#include <libavutil\imgutils.h>
#include <libswscale\swscale.h>
}

#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "swscale.lib")

#include <string>
#include <map>

class CFFmpegException : public std::runtime_error
{
public:
	CFFmpegException(int nError) : std::runtime_error(std::string("libav error: ").append(av_make_error_string(m_pError, AV_ERROR_MAX_STRING_SIZE, nError)))
	{
	}

private:
	char m_pError[AV_ERROR_MAX_STRING_SIZE];
};

#define CHECK_FFMPEG(x) { int nError = (x); if (nError < 0) throw CFFmpegException(nError); }

class CX264CodecContext : public CCodecContextBase
{
public:
	CX264CodecContext(unsigned int nWidth, unsigned int nHeight, unsigned int nFps, bool bSaveOutputToFile, unsigned int nBitrate, std::string sPreset) : CCodecContextBase(nWidth, nHeight, nFps, bSaveOutputToFile), m_sPreset(sPreset), m_nBitrate(nBitrate)
	{
	}

	unsigned int GetBitrate() const
	{
		return m_nBitrate;
	}

	std::string GetPreset() const
	{
		return m_sPreset;
	}

private:
	std::string m_sPreset;
	unsigned int m_nBitrate;
};

class CX264Codec
{
	class CFFmpegCodecContextDeleter
	{
	public:
		void operator()(AVCodecContext* pAVCodecContext)
		{
			avcodec_free_context(&pAVCodecContext);
		}
	};

public:
	CX264Codec(DWORD nCodecContextId, const CCodecContextBase& CodecContext) : m_X264CodecContext(static_cast<const CX264CodecContext&>(CodecContext)), m_Context(sws_getContext(m_X264CodecContext.GetWidth(), m_X264CodecContext.GetHeight(), AV_PIX_FMT_RGBA, m_X264CodecContext.GetWidth(), m_X264CodecContext.GetHeight(), AV_PIX_FMT_YUV420P, 0, 0, 0, 0)),
		m_pFrame(av_frame_alloc()), m_pYuvFrame(new unsigned char[m_X264CodecContext.GetWidth() * m_X264CodecContext.GetHeight() * 3 / 2])
	{
		auto pAVCodec = avcodec_find_encoder(AVCodecID::AV_CODEC_ID_H264);
		m_pAVCodecContext.reset(avcodec_alloc_context3(pAVCodec));
		AVDictionary* pOptions = nullptr;
		std::map<std::string, std::string> Options;
		Options["threads"] = "1";
		Options["preset"] = m_X264CodecContext.GetPreset();
		Options["tune"] = "zerolatency";
		Options["bitrate"] = std::to_string(m_X264CodecContext.GetBitrate() / 1000).append("k");
		for (auto& Option : Options)
		{
			av_dict_set(&pOptions, Option.first.c_str(), Option.second.c_str(), 0);
		}
		m_pAVCodecContext->pix_fmt = AV_PIX_FMT_YUV420P;
		m_pAVCodecContext->width = m_X264CodecContext.GetWidth();
		m_pAVCodecContext->height = m_X264CodecContext.GetHeight();
		m_pAVCodecContext->gop_size = m_X264CodecContext.GetFrameCount();
		m_pAVCodecContext->time_base.num = 1;
		m_pAVCodecContext->time_base.den = m_X264CodecContext.GetFps();
		m_pAVCodecContext->field_order = AV_FIELD_PROGRESSIVE;
		CHECK_FFMPEG(avcodec_open2(m_pAVCodecContext.get(), nullptr, &pOptions));
		av_packet_from_data(&m_Packet, reinterpret_cast<uint8_t*>(av_malloc(2 * 1024 * 1024)), 2 * 1024 * 1024);
		m_pYuvPlanes[0] = m_pYuvFrame.get();
		m_pYuvPlanes[1] = m_pYuvFrame.get() + m_X264CodecContext.GetWidth() * m_X264CodecContext.GetHeight();
		m_pYuvPlanes[2] = m_pYuvFrame.get() + m_X264CodecContext.GetWidth() * m_X264CodecContext.GetHeight() + m_X264CodecContext.GetWidth() * m_X264CodecContext.GetHeight() / 4;
		m_nYuvStrides[0] = m_X264CodecContext.GetWidth();
		m_nYuvStrides[1] = m_X264CodecContext.GetWidth() / 2;
		m_nYuvStrides[2] = m_X264CodecContext.GetWidth() / 2;
		m_pFrame->format = AV_PIX_FMT_YUV420P;
		m_pFrame->width = m_X264CodecContext.GetWidth();
		m_pFrame->height = m_X264CodecContext.GetHeight();
		if (m_X264CodecContext.GetSaveOutputToFile())
		{
			char pFilename[MAX_PATH];
			sprintf_s(pFilename, MAX_PATH, "x264-%d.h264", nCodecContextId);
			if (fopen_s(&m_pOutputFile, pFilename, "wb") != 0)
			{
				throw std::runtime_error(std::string("could not open output file ").append(pFilename));
			}
		}
	}

	~CX264Codec()
	{
		if (m_X264CodecContext.GetSaveOutputToFile())
		{
			fclose(m_pOutputFile);
		}
		av_packet_unref(&m_Packet);
		av_frame_free(&m_pFrame);
	}

	void Initialize()
	{
	}

	unsigned int Encode(unsigned char* pFrame, bool bKeyframe)
	{
		uint8_t * pImage[1] = { pFrame }; // RGB32 have one plane
		int nLinesize[1] = { m_pAVCodecContext->width * 4 };
		if (sws_scale(m_Context, pImage, nLinesize, 0, m_X264CodecContext.GetHeight(), m_pYuvPlanes, m_nYuvStrides) != m_X264CodecContext.GetHeight())
		{
			throw std::runtime_error("sws_scale failed");
		}
		m_pFrame->pict_type = bKeyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
		CHECK_FFMPEG(av_image_fill_arrays(m_pFrame->data, m_pFrame->linesize, m_pYuvFrame.get(), static_cast<AVPixelFormat>(m_pFrame->format), m_X264CodecContext.GetWidth(), m_X264CodecContext.GetHeight(), 1));
		int nGotPacketPtr;
		CHECK_FFMPEG((avcodec_encode_video2(m_pAVCodecContext.get(), &m_Packet, m_pFrame, &nGotPacketPtr) == 0 && nGotPacketPtr));
		unsigned int nSize = m_Packet.size;
		if (m_X264CodecContext.GetSaveOutputToFile())
		{
			fwrite(m_Packet.data, 1, m_Packet.size, m_pOutputFile);
		}
		m_Packet.size = 2 * 1024 * 1024;
		return nSize;
	}

private:
	CX264CodecContext m_X264CodecContext;
	SwsContext* m_Context;
	std::unique_ptr<AVCodecContext, CFFmpegCodecContextDeleter> m_pAVCodecContext;
	AVPacket m_Packet;
	unsigned int m_nPacketIndex = 0;
	std::unique_ptr<unsigned char[]> m_pYuvFrame;
	uint8_t* m_pYuvPlanes[3];
	int m_nYuvStrides[3];
	AVFrame* m_pFrame;
	FILE* m_pOutputFile = nullptr;
};

int main(int argc, char** argv)
{
	try
	{
		if (argc != 9)
		{
			std::cout << "Usage: NvidiaNvencTest.exe <input filename> <width> <height> <threads> <fps> <bitrate per stream> <preset> <save output to file>" << std::endl;
			return EXIT_FAILURE;
		}
		avcodec_register_all();
		av_log_set_level(AV_LOG_ERROR);
		unsigned int nWidth = std::stoi(argv[2]), nHeight = std::stoi(argv[3]), nThreads = std::stoi(argv[4]), nFps = std::stoi(argv[5]), nBitrate = std::stoi(argv[6]);
		bool bSaveOutputToFile = strcmp("true", argv[8]) == 0;
		std::cout << "Input " << argv[1] << " - " << nWidth << "x" << nHeight << ", threads: " << nThreads << ", fps: " << nFps << ", bitrate: " << nBitrate << " bit/s per stream, save output to file: " << bSaveOutputToFile << ", preset: " << argv[7] << std::endl;
		CTestRun<CX264Codec> TestRun(argv[1], nThreads, CX264CodecContext(nWidth, nHeight, nFps, bSaveOutputToFile, nBitrate, argv[7]));
	}
	catch (std::exception& Exception)
	{
		std::cout << "Error: " << Exception.what() << std::endl;
	}
	return EXIT_SUCCESS;
}