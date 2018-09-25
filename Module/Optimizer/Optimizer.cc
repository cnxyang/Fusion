#include "Optimizer.h"

Optimizer::Optimizer() : map(NULL), noKeyFrames(0) {

}

void Optimizer::run() {

	while(1) {

		if(map->HasNewKF()) {

		}

		std::this_thread::sleep_for(std::chrono::milliseconds(3000));
	}
}

void Optimizer::LocalBA() {

}

void Optimizer::GlobalBA() {

}

void Optimizer::SetMap(Mapping * map_) {

	map = map_;
}
