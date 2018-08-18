#include "rs.hpp"
#include "opencv.hpp"

class RealSense {
public:

	RealSense(int cols, int rows);
	bool Poll_for_frame(cv::Mat& imRGB, cv::Mat& imD);

	float fx;
	float fy;
	float cx;
	float cy;

private:
	int mcols, mrows;
	rs2::pipeline pipe;
	rs2::frameset frames;
	rs2::frame depth;
	rs2::frame color;
	rs2::pipeline_profile selection;
};
