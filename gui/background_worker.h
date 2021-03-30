#pragma once


#include <ThreadPool.h> //https://github.com/mtrebi/thread-pool
#include <cassert>
#include <iostream> //Only for logging, can be replaced by something else

/**
 * \brief Improved background worker that reuses the thread.
 * Can only be allocated on the heap.
 * Source: https://gist.github.com/shamanDevel/215376f8ce74a57c695a0885c5384319
 */
class BackgroundWorker
{
public:
	/// The task that is executed in background
	typedef std::function<void(BackgroundWorker*)> task;

private:
	ThreadPool pool_;
	mutable std::future<void> future_;
	std::atomic<bool> interrupted_;
	std::mutex statusMutex_;
	std::string status_;

public:
	BackgroundWorker(const BackgroundWorker&) = delete;
	void operator=(const BackgroundWorker&) = delete;

	BackgroundWorker()
		: pool_(1)
	{
		pool_.init();
		interrupted_.store(false);
	}
	~BackgroundWorker()
	{
		pool_.shutdown();
	}

	///Launches the task t in a background thread
	void launch(const task& t)
	{
		assert(isDone());
		interrupted_.store(false);
		status_ = "";
		future_ = pool_.submit(t, this);
	}

	///Sets the status message, called from inside the background task
	void setStatus(const std::string& message)
	{
		std::lock_guard<std::mutex> lock(statusMutex_);
		status_ = message;
	}
	///Reads the status message, called from the main thread
	const std::string& getStatus()
	{
		std::lock_guard<std::mutex> lock(statusMutex_);
		return status_;
	}

	///Interrupts the background task, called from the main thread
	void interrupt() { interrupted_.store(true); }
	///Checks if the task is interrupted, called from the background task
	bool isInterrupted() const { return interrupted_; }

	///Tests if the background task is done
	bool isDone() const
	{
		if (!future_.valid()) return true;
		if (future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
		{
			try {
				future_.get(); //done
			}
			catch (std::exception& ex) {
				//Replace by your own exception handling
				std::cerr << "Exception in the background thread! " << ex.what() << std::endl;
			}
			catch (...) {
				//Replace by your own exception handling
				std::cerr << "Unknown exception in the background thread!" << std::endl;
			}
			return true;
		}
		return false;
	}
	/**
	 * \brief Waits until the task is done
	 */
	void wait()
	{
		if (!future_.valid()) return;
		try {
			//future_.get(); //done
			future_.wait();
		}
		catch (std::exception& ex) {
			//Replace by your own exception handling
			std::cerr << "Exception in the background thread! " << ex.what() << std::endl;
		}
		catch (...) {
			//Replace by your own exception handling
			std::cerr << "Unknown exception in the background thread!" << std::endl;
		}
	}

	template< class Clock, class Duration >
	std::future_status wait(const std::chrono::time_point<Clock, Duration>& timeout_time)
	{
		if (!future_.valid()) return std::future_status::ready;
		try {
			//future_.get(); //done
			return future_.wait_until(timeout_time);
		}
		catch (std::exception& ex) {
			//Replace by your own exception handling
			std::cerr << "Exception in the background thread! " << ex.what() << std::endl;
			return std::future_status::ready;
		}
		catch (...) {
			//Replace by your own exception handling
			std::cerr << "Unknown exception in the background thread!" << std::endl;
			return std::future_status::ready;
		}
	}
};