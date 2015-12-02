#pragma once

#include "CodecContextBase.hpp"
#include "InputFileLoader.hpp"

#include <Windows.h>

#include <vector>
#include <deque>
#include <iostream>
#include <memory>
#include <iomanip> 
#include <pdh.h>
#include <strsafe.h>

#pragma comment(lib, "pdh.lib")

ULARGE_INTEGER FileTimeToLargeInteger(const FILETIME& FileTime)
{
	ULARGE_INTEGER LargeInteger;
	LargeInteger.LowPart = FileTime.dwLowDateTime;
	LargeInteger.HighPart = FileTime.dwHighDateTime;
	return LargeInteger;
}

class CSystemTime
{
public:
	CSystemTime()
	{
		FILETIME SystemTime;
		GetSystemTimeAsFileTime(&SystemTime);
		m_SystemTime = FileTimeToLargeInteger(SystemTime);
	}

	ULONGLONG GetSystemTime()
	{
		return m_SystemTime.QuadPart;
	}

private:
	ULARGE_INTEGER m_SystemTime;
};

class CSystemTimes
{
public:
	CSystemTimes()
	{
		FILETIME IdleTime, UserTime, KernelTime;
		GetSystemTimes(&IdleTime, &UserTime, &KernelTime);
		m_IdleTime = FileTimeToLargeInteger(IdleTime);
		m_UserTime = FileTimeToLargeInteger(UserTime);
		m_KernelTime = FileTimeToLargeInteger(KernelTime);
	}

	ULONGLONG GetUserTime()
	{
		return m_UserTime.QuadPart;
	}

	ULONGLONG GetKernelTime()
	{
		return m_KernelTime.QuadPart;
	}

	ULONGLONG GetIdleTime()
	{
		return m_IdleTime.QuadPart;
	}

public:
	ULARGE_INTEGER m_IdleTime;
	ULARGE_INTEGER m_UserTime;
	ULARGE_INTEGER m_KernelTime;
};

class CProcessTimes
{
public:
	CProcessTimes()
	{
		FILETIME CreationTime, ExitTime, KernelTime, UserTime;
		GetProcessTimes(GetCurrentProcess(), &CreationTime, &ExitTime, &KernelTime, &UserTime);
		m_UserTime = FileTimeToLargeInteger(UserTime);
		m_KernelTime = FileTimeToLargeInteger(KernelTime);
	}

	ULONGLONG GetUserTime()
	{
		return m_UserTime.QuadPart;
	}

	ULONGLONG GetKernelTime()
	{
		return m_KernelTime.QuadPart;
	}
public:
	ULARGE_INTEGER m_UserTime;
	ULARGE_INTEGER m_KernelTime;
};

class CCpuUsageMonitor
{
public:
	CCpuUsageMonitor(const wchar_t* pProcessName)
	{
		
		GetSystemInfo(&m_SystemInfo);
		auto nStatus = PdhOpenQuery(NULL, NULL, &m_hPdhQuery);
		_ASSERT(nStatus == ERROR_SUCCESS);
		nStatus = PdhAddCounter(m_hPdhQuery, L"\\Processor(_Total)\\% Processor Time", NULL, &m_hPdhCpuUsageCounter);
		_ASSERT(nStatus == ERROR_SUCCESS);
		wchar_t pCounterPath[PDH_MAX_COUNTER_PATH];
		StringCbPrintf(pCounterPath, PDH_MAX_COUNTER_PATH, L"\\Process(%s)\\%% Processor Time", pProcessName);
		nStatus = PdhAddCounter(m_hPdhQuery, pCounterPath, NULL, &m_hPhdProcessCpuUsageCounter);
		_ASSERT(nStatus == ERROR_SUCCESS);
	}

	~CCpuUsageMonitor()
	{
		PdhCloseQuery(&m_hPdhQuery);
	}

	void CollectSample()
	{
		auto nStatus = PdhCollectQueryData(m_hPdhQuery);
		_ASSERT(nStatus == ERROR_SUCCESS);
	}

	double GetCpuUsage()
	{
		DWORD nType;
		PDH_FMT_COUNTERVALUE CounterValue;
		auto nStatus = PdhGetFormattedCounterValue(m_hPdhCpuUsageCounter, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &nType, &CounterValue);
		_ASSERT(nStatus == ERROR_SUCCESS);
		return CounterValue.doubleValue;
	}

	double GetProcessCpuUsage()
	{
		DWORD nType;
		PDH_FMT_COUNTERVALUE CounterValue;
		auto nStatus = PdhGetFormattedCounterValue(m_hPhdProcessCpuUsageCounter, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &nType, &CounterValue);
		_ASSERT(nStatus == ERROR_SUCCESS);
		return CounterValue.doubleValue / m_SystemInfo.dwNumberOfProcessors;
	}

private:
	SYSTEM_INFO m_SystemInfo;
	HANDLE m_hPdhQuery;
	HANDLE m_hPdhCpuUsageCounter;
	HANDLE m_hPhdProcessCpuUsageCounter;
};

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

	CTestRun(const char* pInputFilename, unsigned int nThreadCount, CCodecContextBase& CodecContext, CCpuUsageMonitor& CpuUsageMonitor)
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
		CSystemTime StartSystemTime;
		CpuUsageMonitor.CollectSample();
		CProcessTimes StartProcessTimes;
		CSystemTimes StartSystemTimes;
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
		CpuUsageMonitor.CollectSample();
		CSystemTime StopSystemTime;
		CProcessTimes StopProcessTimes;
		CSystemTimes StopSystemTimes;
		WaitForMultipleObjects(static_cast<DWORD>(m_Threads.size()), &m_Threads[0], TRUE, INFINITE);
		unsigned int nTestContextIndex = 0;
		size_t nTotalSize = 0;
		size_t nTotalDroppedFrames = 0;
		for (auto &TestContext : m_TestContexts)
		{
			TestContext->GetStatistics(nTotalSize, nTotalDroppedFrames);
		}
		SYSTEM_INFO SystemInfo;
		GetSystemInfo(&SystemInfo);
		double fCpuUsageFromMethodOne = ((StopProcessTimes.GetUserTime() - StartProcessTimes.GetUserTime()) + (StopProcessTimes.GetKernelTime() - StartProcessTimes.GetKernelTime())) * 100.0 / (SystemInfo.dwNumberOfProcessors * (StopSystemTime.GetSystemTime() - StartSystemTime.GetSystemTime())),
			fCpuUsageFromMethodTwo = ((StopProcessTimes.GetUserTime() - StartProcessTimes.GetUserTime()) + (StopProcessTimes.GetKernelTime() - StartProcessTimes.GetKernelTime())) * 100.0 / ((StopSystemTimes.GetKernelTime() - StartSystemTimes.GetKernelTime()) + (StopSystemTimes.GetUserTime() - StartSystemTimes.GetUserTime()));
		std::cout << "Consumed CPU (GetProcessTimes() / Wall Time)" << fCpuUsageFromMethodTwo << std::endl;
		std::cout << "Consumed CPU (GetProcessTimes() / GetSystemTimes())" << fCpuUsageFromMethodOne << std::endl;
		std::cout << "Elapsed time: Consumed CPU: Frames total: Frames dropped: Duration: Total size: Total Bitrate:" << std::endl;
		const double fDuration = (Frames.size() * 1.0) / CodecContext.GetFps();
		std::cout << std::setw(13) << (StopSystemTime.GetSystemTime() - StartSystemTime.GetSystemTime()) / 10000000.0 << " " << std::setw(13) << CpuUsageMonitor.GetProcessCpuUsage() << " " <<
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