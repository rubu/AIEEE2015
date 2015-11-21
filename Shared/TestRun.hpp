#pragma once

#include <Windows.h>

#include <vector>
#include <deque>
#include <iostream>
#include <memory>

template<typename _Codec>
class CTestRun
{
public:
	class CTestContext
	{
	public:
		CTestContext(const CTestContext& TestContext) = delete;

		CTestContext(size_t nFrameCount, unsigned int nContextId, unsigned int nFps, const typename _Codec::CContext& CodecContext) : m_Codec(CodecContext), m_nContextId(nContextId), m_hNewFrameEvent(CreateEvent(NULL, FALSE, FALSE, NULL)), m_fDuration(nFrameCount * 1.0 / nFps), m_hCodecInitializedEvent(CreateEvent(NULL, TRUE, FALSE, NULL))
		{
			InitializeCriticalSection(&m_CriticalSection);
			m_FrameStatistics.resize(nFrameCount);
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

		void SetFrameStatistics(unsigned int nFrameIndex, LONGLONG nElapsedTime, unsigned int nSize)
		{
			EnterCriticalSection(&m_CriticalSection);
			m_FrameStatistics[nFrameIndex] = std::make_pair(nElapsedTime, nSize);
			LeaveCriticalSection(&m_CriticalSection);
		}

		void PrintStatistics(double& fTotalBitrate, size_t& nFramesDropped)
		{
			std::cout << "Frames dropped: " << m_nFramesDropped << std::endl;
			unsigned int nFrameIndex = 0;
			LARGE_INTEGER PerformanceFrequency;
			QueryPerformanceFrequency(&PerformanceFrequency);
			size_t nTotalBitrate = 0;
			for (auto& FrameStatistics : m_FrameStatistics)
			{
				std::cout << "\t" << nFrameIndex++ << ": elapsed time - " << FrameStatistics.first * 1.0 / PerformanceFrequency.QuadPart << ", size - " << FrameStatistics.second << std::endl;
				nTotalBitrate += FrameStatistics.second;
			}
			double fTestRunBitrate = 8.0 * nTotalBitrate / m_fDuration;
			fTotalBitrate += fTestRunBitrate;
			nFramesDropped += m_nFramesDropped;
			std::cout << "Total bitrate: " << std::fixed << fTestRunBitrate << " bits/s" << std::endl;
		}

		void Run()
		{
			m_Codec.Initialize();
			SetEvent(m_hCodecInitializedEvent);
			unsigned char* pFrame = nullptr;
			LARGE_INTEGER Start, Stop;
			bool bFirstFrame = true;
			unsigned int nFrameIndex = 0;
			while ((pFrame = GetNewFrame(nFrameIndex)) != nullptr)
			{
				QueryPerformanceCounter(&Start);
				unsigned int nSize = m_Codec.Encode(pFrame, bFirstFrame);
				bFirstFrame = false;
				QueryPerformanceCounter(&Stop);
				SetFrameStatistics(nFrameIndex, Stop.QuadPart - Start.QuadPart, nSize);
			}
		}

		void WaitForCodec()
		{
			WaitForSingleObject(m_hCodecInitializedEvent, INFINITE);
		}

		void Store()
		{
			m_Codec.Store(m_nContextId);
		}

	private:
		CRITICAL_SECTION m_CriticalSection;
		unsigned char* m_pCurrentFrame = nullptr;
		HANDLE m_hNewFrameEvent;
		size_t m_nFramesDropped = 0;
		unsigned int  m_nFrameCount = 0;
		std::vector<std::pair<LONGLONG, unsigned int>> m_FrameStatistics;
		_Codec m_Codec;
		unsigned int m_nContextId;
		double m_fDuration;
		HANDLE m_hCodecInitializedEvent;
	};

	CTestRun(const std::unique_ptr<unsigned char[]>& pInput, unsigned int nThreadCount, const typename _Codec::CContext& CodecContext) 
	{
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
		QueryPerformanceCounter(&Start);
		FILETIME CreationFileTime, ExitFileTime, StartUserFileTime, StartKerneFileTime, StopUserFileTime, StopKernelFileTime;
		GetProcessTimes(hCurrentProcess, &CreationFileTime, &ExitFileTime, &StartKerneFileTime, &StartUserFileTime);
		unsigned int nFrameIndex = 0;
		while (nFrameIndex < CodecContext.GetFrameCount())
		{
			WaitForSingleObject(hTimer, INFINITE);
			for (auto& TestContext : m_TestContexts)
			{
				TestContext->PushFrame(pInput.get() + nFrameIndex * CodecContext.GetWidth() * CodecContext.GetHeight() * 4);
			}
		}
		WaitForSingleObject(hTimer, INFINITE);
		for (auto& TestContext : m_TestContexts)
		{
			TestContext->PushFrame(nullptr);
		}
		WaitForMultipleObjects(static_cast<DWORD>(m_Threads.size()), &m_Threads[0], TRUE, INFINITE);
		GetProcessTimes(hCurrentProcess, &CreationFileTime, &ExitFileTime, &StopKernelFileTime, &StopUserFileTime);
		QueryPerformanceCounter(&End);
		QueryPerformanceFrequency(&Frequency);
		auto fConsumedTime = (End.QuadPart - Start.QuadPart) * 1.0f / (Frequency.QuadPart);
		ULARGE_INTEGER StartUserTime{ StartUserFileTime.dwLowDateTime, StartUserFileTime.dwHighDateTime },
			StopUserTime{ StopUserFileTime.dwLowDateTime, StopUserFileTime.dwHighDateTime },
			StartKernelTime{ StartKerneFileTime.dwLowDateTime, StartKerneFileTime.dwHighDateTime },
			StopKernelTime{ StopKernelFileTime.dwLowDateTime, StopKernelFileTime.dwHighDateTime };
		auto fConsumedCpuTime = ((StopUserTime.QuadPart - StartUserTime.QuadPart) + (StopKernelTime.QuadPart - StartKernelTime.QuadPart)) / (10000000.0);
		std::cout << "Elapsed time: " << fConsumedTime << " , consumed CPU: " << fConsumedCpuTime * 100.0 / (fConsumedTime * SystemInfo.dwNumberOfProcessors) << std::endl;
		unsigned int nTestContextIndex = 0;
		double fTotalBitrate = 0.0;
		size_t nTotalDroppedFrames = 0;
		for (auto &TestContext : m_TestContexts)
		{
			std::cout << nTestContextIndex++ << ": " << std::endl;
			TestContext->PrintStatistics(fTotalBitrate, nTotalDroppedFrames);
			TestContext->Store();
			std::cout << std::endl;
		}
		std::cout << "Total bitrate for all streams: " << std::fixed << fTotalBitrate << ", total frames/dropped: " << CodecContext.GetFrameCount() * nThreadCount << "/" << nTotalDroppedFrames << std::endl;
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