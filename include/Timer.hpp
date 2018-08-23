#ifndef TIMER_HPP__
#define TIMER_HPP__

#include <map>
#include <chrono>
#include <iostream>


class Timer {
public:
	static void Print();
	static void Enable();
	static void Disable();
	static void AddCategory(std::string str);
	static void Start(const std::string cat, const std::string str);
	static void Stop(const std::string cat, const std::string str);

private:
	static bool bEnabled;
	static std::map<std::string, std::map<std::string, std::chrono::high_resolution_clock::time_point>> mTable;
	static std::map<std::string, std::map<std::string, int>> mDuration;
};

#endif
