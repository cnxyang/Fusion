#include "dataset_interface.h"

TUMDatasetInterface::TUMDatasetInterface(std::string dir) : id(0), base_dir(dir)
{
	if(base_dir.back() != '/')
	{
		base_dir += '/';
	}
}

TUMDatasetInterface::~TUMDatasetInterface()
{

}

void TUMDatasetInterface::load_association_file(std::string file_name)
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

void TUMDatasetInterface::load_ground_truth(std::string file_name)
{
	double ts;
	double tx, ty, tz, qx, qy, qz, qw;

	std::ifstream file;
	file.open(base_dir + file_name);

	while(file >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
	{
		Eigen::Quaterniond q(qw, qx, qy, qz);
		q.normalize();
		auto r = q.toRotationMatrix();
		auto t = Eigen::Vector3d(tx, ty, tz);
		Sophus::SE3d gt(r, t);
		gt_list.push_back(gt);
	}

	file.close();

	printf("Total of %lu Ground truth data Loaded.\n", gt_list.size());
}

bool TUMDatasetInterface::read_next_images(cv::Mat& image, cv::Mat& depth)
{
	if(id == image_list.size())
		return false;

	std::string fullpath_image = base_dir + image_list[id];
	std::string fullpath_depth = base_dir + depth_list[id];

	image = cv::imread(fullpath_image, cv::IMREAD_UNCHANGED);
	depth = cv::imread(fullpath_depth, cv::IMREAD_UNCHANGED);

	id++;
	return true;
}

std::vector<Sophus::SE3d> TUMDatasetInterface::get_groundtruth() const
{
	return gt_list;
}

double TUMDatasetInterface::get_current_timestamp() const
{
	return time_stamp[id - 1];
}

unsigned int TUMDatasetInterface::get_current_id() const
{
	return id - 1;
}

void TUMDatasetInterface::save_full_trajectory(std::vector<Sophus::SE3d> full_trajectory, std::string file_name) const
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
