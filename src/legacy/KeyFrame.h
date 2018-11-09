//#ifndef KEY_FRAME_HPP__
//#define KEY_FRAME_HPP__
//
//#include "Frame.h"
//#include "MapPoint.h"
//#include "Utilities/VectorMath.h"
//#include "Utilities/DeviceArray.h"
//
//#include <Eigen/Dense>
//#include <opencv.hpp>
//
//class Frame;
//struct MapPoint;
//
//struct KeyFrame {
//
//	KeyFrame();
//
//	KeyFrame(Frame * f);
//
//	Eigen::Matrix3f Rotation() const;
//
//	Eigen::Vector3f Translation() const;
//
//	Matrix3f GpuRotation() const;
//
//	Matrix3f GpuInvRotation() const;
//
//	float3 GpuTranslation() const;
//
//	Eigen::Vector3f GetWorldPoint(int i) const;
//
//	void ComputePoseChange();
//
//	int N;
//	unsigned long frameId;
//
//	Eigen::Matrix4f pose;
//	Eigen::Matrix4f newPose;
//
//	cv::cuda::GpuMat descriptors;
//	std::vector<float4> pointNormal;
//	std::vector<cv::KeyPoint> keyPoints;
//	std::vector<int> observations;
//
//	mutable std::vector<bool> outliers;
//	mutable std::vector<int> keyIndex;
//	mutable std::vector<Eigen::Vector3f> mapPoints;
//	mutable std::vector<MapPoint *> pt3d;
//
//	DeviceArray2D<unsigned char> image;
//	DeviceArray2D<float> depth;
//
//	std::vector<Frame *> subFrames;
//
//	std::vector<KeyFrame *> visGraph;
//
//	static int nextId;
//	float poseChanged;
//};
//
//#endif