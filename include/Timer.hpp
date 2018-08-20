#ifndef TIMER_HPP__
#define TIMER_HPP__

#include <map>
#include <chrono>
#include <iostream>


class Timer {
public:
	static void PrintTiming();
	static void AddCategory(std::string str);
	static void StartTiming(const std::string cat, const std::string str);
	static void StopTiming(const std::string cat, const std::string str);

private:
	static std::map<std::string, std::map<std::string, std::chrono::steady_clock::time_point>> mTable;
	static std::map<std::string, std::map<std::string, int>> mDuration;
};

#endif
