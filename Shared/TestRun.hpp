#pragma once

#include "CodecContextBase.hpp"
#include "InputFileLoader.hpp"

#include <Windows.h>

#include <vector>
#include <deque>
#include <iostream>
#include <memory>
#include <iomanip> 

template<typename _Codec>
class CTestRun
{
public:
	class CTestContext
	{
	public:
		CTestContext(const CTestContext& TestContext) = delete;

		CTestContext(size_t nFrameCount, unsigned int nContextId, unsigned int nFps, CCodecContextBase& CodecContext) : m_Codec(nContextId, CodecContext), m_nContextId(nContextId), m_hNewFrameEvent(CreateEvent(NULL, FALSE, FALSE, NULL)), m_hCodecInitializedEvent(CreateEvent(NULL, TRUE, FALSE, NULL))
		{
			InitializeCriticalSection(&m_CriticalSection);
			m_FrameSizes.resize(nFrameCount);
		}

		~CTestContext()
		{
			if (m_hNewFrameEvent != NULL)
			{
				CloseHandle(m_hNewFrameEvent);
				m_hNewFrameEvent = NULL;
			}
		}

		unsigned char* GetNewFrame(unsigned int& nFrameIndex)
		{
			unsigned char* pFrame = nullptr;
			WaitForSingleObject(m_hNewFrameEvent, INFINITE);
			EnterCriticalSection(&m_CriticalSection);
			pFrame = m_pCurrentFrame;
			nFrameIndex = m_nFrameCount - 1;
			m_pCurrentFrame = nullptr;
			LeaveCriticalSection(&m_CriticalSection);
			return pFrame;
		}

		void PushFrame(unsigned char* pFrame)
		{
			EnterCriticalSection(&m_CriticalSection);
			if (m_pCurrentFrame != nullptr)
			{
				m_nFramesDropped += 1;
			}
			m_nFrameCount++;
			m_pCurrentFrame = pFrame;
			LeaveCriticalSection(&m_CriticalSection);
			SetEvent(m_hNewFrameEvent);
		}

		void SetFrameSize(unsigned int nFrameIndex, unsigned int nSize)
		{
			EnterCriticalSection(&m_CriticalSection);
			m_FrameSizes[nFrameIndex] = nSize;
			LeaveCriticalSection(&m_CriticalSection);
		}

		void GetStatistics(size_t& nTotalSize, size_t& nFramesDropped)
		{
			for (auto& FrameSize : m_FrameSizes)
			{
				nTotalSize += FrameSize;
			}
			nFramesDropped += m_nFramesDropped;
		}

		void Run()
		{
			m_Codec.Initialize();
			SetEvent(m_hCodecInitializedEvent);
			unsigned char* pFrame = nullptr;
			bool bFirstFrame = true;
			unsigned int nFrameIndex = 0;
			while ((pFrame = GetNewFrame(nFrameIndex)) != nullptr)
			{
				unsigned int nSize = m_Codec.Encode(pFrame, bFirstFrame);
				bFirstFrame = false;
				SetFrameSize(nFrameIndex, nSize);
			}
		}

		void WaitForCodec()
		{
			WaitForSingleObject(m_hCodecInitializedEvent, INFINITE);
		}

	private:
		CRITICAL_SECTION m_CriticalSection;
		unsigned char* m_pCurrentFrame = nullptr;
		HANDLE m_hNewFrameEvent;
		size_t m_nFramesDropped = 0;
		unsigned int  m_nFrameCount = 0;
		std::vector<unsigned int> m_FrameSizes;
		_Codec m_Codec;
		unsigned int m_nContextId;
		HANDLE m_hCodecInitializedEvent;
	};

	CTestRun(const char* pInputFilename, unsigned int nThreadCount, CCodecContextBase& CodecContext) 
	{
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
		CInputFileLoader InputFileLoader(pInputFilename, CodecContext.GetWidth(), CodecContext.GetHeight());
		CodecContext.SetFrameCount(InputFileLoader.GetFrameCount());
		std::cout << "Input file preloaded!" << std::endl;
		DWORD nThreadId = 0;
		for (unsigned int nThreadIndex = 0; nThreadIndex < nThreadCount; ++nThreadIndex)
		{
			m_TestContexts.push_back(std::move(std::unique_ptr<CTestContext>(new CTestContext(CodecContext.GetFrameCount(), nThreadIndex, CodecContext.GetFps(), CodecContext))));
			auto pTestContext = m_TestContexts.back().get();
			HANDLE hThread = CreateThread(nullptr, 0, &CTestRun::ThreadProc, pTestContext, 0, &nThreadId);
			pTestContext->WaitForCodec();
			m_Threads.push_back(hThread);
		}
		HANDLE hTimer = CreateWaitableTimer(NULL, FALSE, NULL);
		LARGE_INTEGER DueTime = { 0 };
		SetWaitableTimer(hTimer, &DueTime, 1000 / CodecContext.GetFps(), nullptr, nullptr, FALSE);
		LARGE_INTEGER Start, End, Frequency;
		SYSTEM_INFO SystemInfo;
		GetSystemInfo(&SystemInfo);
		HANDLE hCurrentProcess = GetCurrentProcess();
		FILETIME CreationFileTime, ExitFileTime, StartUserFileTime, StartKerneFileTime, StopUserFileTime, StopKernelFileTime;
		QueryPerformanceCounter(&Start);
		GetProcessTimes(hCurrentProcess, &CreationFileTime, &ExitFileTime, &StartKerneFileTime, &StartUserFileTime);
		const auto& Frames = InputFileLoader.GetFrames();
		for (const auto& Frame : Frames)
		{
			WaitForSingleObject(hTimer, INFINITE);
			for (auto& TestContext : m_TestContexts)
			{
				TestContext->PushFrame(Frame.get());
			}
		}
		WaitForSingleObject(hTimer, INFINITE);
		for (auto& TestContext : m_TestContexts)
		{
			TestContext->PushFrame(nullptr);
		}
		QueryPerformanceCounter(&End);
		GetProcessTimes(hCurrentProcess, &CreationFileTime, &ExitFileTime, &StopKernelFileTime, &StopUserFileTime);
		WaitForMultipleObjects(static_cast<DWORD>(m_Threads.size()), &m_Threads[0], TRUE, INFINITE);
		QueryPerformanceFrequency(&Frequency);
		auto fConsumedTime = (End.QuadPart - Start.QuadPart) * 1.0f / (Frequency.QuadPart);
		ULARGE_INTEGER StartUserTime{ StartUserFileTime.dwLowDateTime, StartUserFileTime.dwHighDateTime },
			StopUserTime{ StopUserFileTime.dwLowDateTime, StopUserFileTime.dwHighDateTime },
			StartKernelTime{ StartKerneFileTime.dwLowDateTime, StartKerneFileTime.dwHighDateTime },
			StopKernelTime{ StopKernelFileTime.dwLowDateTime, StopKernelFileTime.dwHighDateTime };
		auto fConsumedCpuTime = ((StopUserTime.QuadPart - StartUserTime.QuadPart) + (StopKernelTime.QuadPart - StartKernelTime.QuadPart)) / (10000000.0);
		unsigned int nTestContextIndex = 0;
		size_t nTotalSize = 0;
		size_t nTotalDroppedFrames = 0;
		for (auto &TestContext : m_TestContexts)
		{
			TestContext->GetStatistics(nTotalSize, nTotalDroppedFrames);
		}
		std::cout << "Elapsed time: Consumed CPU: Frames total: Frames dropped: Duration: Total size: Total Bitrate:" << std::endl;
		const double fDuration = (Frames.size() * 1.0) / CodecContext.GetFps();
		std::cout << std::setw(13) << fConsumedTime << " " << std::setw(13) << fConsumedCpuTime * 100.0 / (fConsumedTime * SystemInfo.dwNumberOfProcessors) << " " <<
			std::setw(13) << Frames.size() * nThreadCount << " " << std::setw(15) << nTotalDroppedFrames << " " << std::setw(9) << fDuration << " " << std::setw(11) << nTotalSize << " " << std::fixed << (nTotalSize * 8.0) / fDuration;
		CancelWaitableTimer(hTimer);
		CloseHandle(hTimer);
	}
	
	~CTestRun()
	{
		for (auto& Thread : m_Threads)
		{
			CloseHandle(Thread);
		}
	}

	static DWORD WINAPI ThreadProc(LPVOID pParameter)
	{
		try
		{
			CTestContext* pTestContext = reinterpret_cast<CTestContext*>(pParameter);
			pTestContext->Run();
		}
		catch (std::exception& Exception)
		{
			std::cout << "Error: " << Exception.what() << std::endl;
		}
		return 0;
	}

private:
	std::vector<std::unique_ptr<CTestContext>> m_TestContexts;
	std::vector<HANDLE> m_Threads;
	HANDLE m_hTerminationEvent;
};