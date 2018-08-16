#ifndef TIMER_HPP__
#define TIMER_HPP__

#include <map>
#include <chrono>
#include <iostream>

using namespace std;

class Timer {
public:
	static void PrintTiming();
	static void AddCategory(string str);
	static void StartTiming(const string cat, const string str);
	static void StopTiming(const string cat, const string str);

private:
	static map<string, map<string, chrono::steady_clock::time_point>> mTable;
	static map<string, map<string, int>> mDuration;
};

#endif
