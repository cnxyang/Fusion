#ifndef DATASET_INTERFACE
#define DATASET_INTERFACE

#include <iostream>
#include <opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>

class TUMDatasetWrapper
{
public:

	TUMDatasetWrapper(std::string dir);
	void load_association_file(std::string file_name);
	void load_ground_truth(std::string file_name);
	bool read_next_images(cv::Mat& image, cv::Mat& depth);
	std::vector<Sophus::SE3d> get_groundtruth() const;
	double get_current_timestamp() const;
	unsigned int get_current_id() const;
	void save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory, std::string file_name) const;
	int find_closest_index(std::vector<double> list, double time) const;
	Sophus::SE3d get_starting_pose() const;

private:

	unsigned int id;
	std::string base_dir;
	std::vector<double> time_stamp;
	std::vector<std::string> image_list;
	std::vector<std::string> depth_list;
	std::vector<double> time_stamp_gt;
	std::vector<Sophus::SE3d> ground_truth;
};

#endif
