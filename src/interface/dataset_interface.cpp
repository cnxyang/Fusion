#include "dataset_interface.h"

TUMDatasetWrapper::TUMDatasetWrapper(std::string dir) : id(0), base_dir(dir)
{
	if(base_dir.back() != '/')
		base_dir += '/';
}

void TUMDatasetWrapper::load_association_file(std::string file_name)
{
	std::ifstream file;
	file.open(base_dir + file_name, std::ios_base::in);

	double ts;
	std::string name_depth, name_image;

	while(file >> ts >> name_image >> ts >> name_depth)
	{
		image_list.push_back(name_image);
		depth_list.push_back(name_depth);
		time_stamp.push_back(ts);
	}

	file.close();

	printf("Total of %lu Images Loaded.\n", depth_list.size());
}

int TUMDatasetWrapper::find_closest_index(std::vector<double> list, double time) const
{
	int idx = -1;
	double min_val = std::numeric_limits<double>::max();
	for(int i = 0; i < list.size(); ++i)
	{
		double d = std::abs(list[i] - time);
		if(d < min_val)
		{
			idx = i;
			min_val = d;
		}
	}
	return idx;
}

void TUMDatasetWrapper::load_ground_truth(std::string file_name)
{
	if(time_stamp.size() == 0)
	{
		printf("Please load images first!\n");
		return;
	}

	double ts;
	double tx, ty, tz, qx, qy, qz, qw;

	std::ifstream file;
	file.open(base_dir + file_name);

	for(int i = 0; i < 3; ++i)
	{
		std::string line;
		std::getline(file, line);
	}

	std::vector<double> ts_gt;
	std::vector<Sophus::SE3d> vgt;
	while(file >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
	{
		Eigen::Quaterniond q(qw, qx, qy, qz);
		q.normalize();
		auto r = q.toRotationMatrix();
		auto t = Eigen::Vector3d(tx, ty, tz);
		Sophus::SE3d gt(r, t);

		ts_gt.push_back(ts);
		vgt.push_back(gt);
	}
	std::cout << vgt[0].matrix() << std::endl;

	for(int i = 0; i < time_stamp.size(); ++i)
	{
		double time = time_stamp[i];
		int idx = find_closest_index(ts_gt, time);
		time_stamp_gt.push_back(ts_gt[idx]);
		ground_truth.push_back(vgt[idx]);
	}

	file.close();
	printf("Total of %lu Ground Truth Data Loaded.\n", ground_truth.size());
}

bool TUMDatasetWrapper::read_next_images(cv::Mat& image, cv::Mat& depth)
{
	if(id >= image_list.size())
		return false;

	std::string fullpath_image = base_dir + image_list[id];
	std::string fullpath_depth = base_dir + depth_list[id];

	image = cv::imread(fullpath_image, cv::IMREAD_UNCHANGED);
	depth = cv::imread(fullpath_depth, cv::IMREAD_UNCHANGED);

	id++;
	return true;
}

void TUMDatasetWrapper::save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory, std::string file_name) const
{
	std::ofstream file;
	std::string file_path = base_dir + file_name;
	file.open(file_path, std::ios_base::out);

	for(int i = 0; i < full_trajectory.size(); ++i)
	{
		if(i >= time_stamp.size())
			break;

		double ts = time_stamp[i];
		Sophus::SE3d& curr = full_trajectory[i];
		Eigen::Vector3d t = curr.translation();
		Eigen::Quaterniond q(curr.rotationMatrix());

		file << std::fixed
			 << std::setprecision(4)
		     << ts << " "
			 << t(0) << " "
			 << t(1) << " "
			 << t(2) << " "
			 << q.x() << " "
			 << q.y() << " "
			 << q.z() << " "
			 << q.w() << std::endl;
	}

	file.close();
}

std::vector<Sophus::SE3d> TUMDatasetWrapper::get_groundtruth() const
{
	return ground_truth;
}

double TUMDatasetWrapper::get_current_timestamp() const
{
	return time_stamp[id - 1];
}

unsigned int TUMDatasetWrapper::get_current_id() const
{
	return id - 1;
}

Sophus::SE3d TUMDatasetWrapper::get_starting_pose() const
{
	double time = time_stamp[0];
	int idx = find_closest_index(time_stamp_gt, time);
	return ground_truth[idx];
}
