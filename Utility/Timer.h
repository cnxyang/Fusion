#ifndef TIMER_H__
#define TIMER_H__

#include <map>
#include <chrono>
#include <iostream>

class Timer {
public:

	static void Enable() {
//		bEnabled = true;
	}

	static void Disable(){
//		bEnabled = false;
	}

	static void AddCategory(std::string str)  {
//		mTable[str] = std::map<std::string, std::chrono::high_resolution_clock::time_point>();
//		mDuration[str] = std::map<std::string, int>();
	}

	static void Start(const std::string cat, const std::string str) {
//		if(!bEnabled)
//			return;
//
//		if (!mTable.count(cat))
//			AddCategory(cat);
//
//		auto& iter = mTable[cat];
//		iter[str] = std::chrono::high_resolution_clock::now();
	}

	static void Stop(const std::string cat, const std::string str) {
//		if(!bEnabled)
//			return;
//
//		if (!mTable.count(cat) || !mTable[cat].count(str))
//			return;
//
//		auto t = std::chrono::high_resolution_clock::now();
//		auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t - mTable[cat][str]);
//
//		mDuration[cat][str] = dt.count();
	}

	static void Print() {
//		if(!bEnabled)
//			return;
//
//		int counter = 1;
//		std::map<std::string, std::map<std::string, int>>::iterator iter = mDuration.begin();
//		std::map<std::string, std::map<std::string, int>>::iterator lend = mDuration.end();
//
//		std::cout << "------------------------------------------" << std::endl;
//		std::cout << "Timer Readings: " << std::endl;
//		for (; iter != lend; ++iter) {
//
//			int total_time = 0;
//			int inner_counter = 1;
//
//			std::map<std::string, int>::iterator inner_iter = mDuration[iter->first].begin();
//			std::map<std::string, int>::iterator inner_lend = mDuration[iter->first].end();
//			for (; inner_iter != inner_lend; ++inner_iter) {
//				total_time += inner_iter->second;
//			}
//
//			std::string unit;
//			if(total_time >= 1000) {
//				total_time /= 1000;
//				unit = " ms";
//			}
//			else
//				unit = " us";
//
//			std::cout << counter++
//					  << ". " << iter->first << " : "
//					  << total_time << unit << std::endl;
//
//			inner_iter = mDuration[iter->first].begin();
//			for (; inner_iter != inner_lend; ++inner_iter) {
//				int time = inner_iter->second;
//				if (time >= 1000) {
//					time /= 1000;
//					unit = " ms";
//				} else
//					unit = " us";
//
//				std::cout << "\t"
//						  << inner_counter++ << ". "
//						  << inner_iter->first
//						  << " : "
//						  << time
//						  << unit << std::endl;
//			}
//		}
	}

private:
//	static bool bEnabled;
//	static std::map<std::string, std::map<std::string, std::chrono::high_resolution_clock::time_point>> mTable;
//	static std::map<std::string, std::map<std::string, int>> mDuration;
};

#endif
