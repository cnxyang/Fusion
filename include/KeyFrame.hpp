#include "Frame.hpp"

using namespace std;
using Eigen::Vector2d;

class KeyFrame {
public:
	KeyFrame(const Frame& F);

public:
	int mKFId;
	static int nextId;
	std::vector<Eigen::Vector2d> mObservations;
};
