#include <opencv2/opencv.hpp>
#include "SlamSystem.h"
#include "interface/dataset_interface.h"

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("Must provide path to the dataset as parameters!\n");
		exit(-1);
	}

	std::string data_path = std::string(argv[1]);
	TUMDatasetInterface interface(data_path);
	interface.load_association_file("association.txt");
	interface.load_ground_truth("groundtruth.txt");

	Eigen::Matrix3f K;
	K << 535.4f, 0.f, 320.1f,
		 0.f, 539.2f, 247.6f,
		 0.f, 0.f, 1.f;

	SlamSystem sys(640, 480, K);
	sys.load_groundtruth(interface.get_groundtruth());

	cv::Mat image, depth;
	cv::Mat image_float, depth_float;
	cv::Mat intensity;

	while(interface.read_next_images(image, depth))
	{
		int id = interface.get_current_id();
		double ts = interface.get_current_timestamp();
		cvtColor(image, image, cv::COLOR_BGR2RGB);
		depth.convertTo(depth_float, CV_32FC1, 1.f / 5000.f);
		sys.trackFrame(image, depth_float, id, ts);
	}

	cv::waitKey(1000);
	sys.build_full_trajectory();
	interface.save_full_trajectory(sys.full_trajectory, "result.txt");
}
