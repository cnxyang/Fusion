#ifndef DATASET_INTERFACE
#define DATASET_INTERFACE

#include <iostream>
#include <opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>

class TUMDatasetInterface
{
public:

	TUMDatasetInterface(std::string dir);

	~TUMDatasetInterface();

	/** ESSENTIAL: Load data from the association file */
	void load_association_file(std::string file_name);

	/** ESSENTIAL: Load ground truth data from file system */
	void load_ground_truth(std::string file_name);

	/** ESSENTIAL: Read the next pair of images. return false if there is none */
	bool read_next_images(cv::Mat& image, cv::Mat& depth);

	/** MUTATOR: Return the list of all ground truth poses */
	std::vector<Sophus::SE3d> get_groundtruth() const;

	/** MUTATOR: Return the time stamp of the current frame */
	double get_current_timestamp() const;

	/** MUTATOR: Return the id of the current frame */
	unsigned int get_current_id() const;

	/** ADVANCED: Save the full camera trajectory to the file system */
	void save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory, std::string file_name) const;

private:

	unsigned int id;
	std::string base_dir;
	std::vector<double> time_stamp;
	std::vector<std::string> image_list;
	std::vector<std::string> depth_list;
	std::vector<Sophus::SE3d> gt_list;
};

#endif
