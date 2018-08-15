#include "Map.hpp"
#include "Tracking.hpp"

class Viewer {
public:
	Viewer();
	void Spin();

private:
	Map* mpMap;
};
