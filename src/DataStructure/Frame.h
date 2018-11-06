#pragma once
#include "opencv.hpp"
#include "SophusUtil.h"
#include "PoseStruct.h"
#include <unordered_set>

class KeyPointStruct;

class Frame
{
public:

	static const int NUM_PYRS = 3;
	Frame(cv::Mat & image, cv::Mat & depth, int id, Eigen::Matrix3f K, double timeStamp);
	void initialize(int w, int h, int id, Eigen::Matrix3f & K, double timeStamp);

	/** Returns the unique frame id. */
	inline int id() const;
	/** Returns the frame's image width. */
	inline int width(int level = 0) const;
	/** Returns the frame's image height. */
	inline int height(int level = 0) const;
	/** Returns the frame's total number of pixels */
	inline int totalPixel(int level = 0) const;
	/** Returns the frame's intrinsics matrix. */
	inline const Eigen::Matrix3f & K(int level = 0) const;
	/** Returns the frame's inverse intrinsics matrix. */
	inline const Eigen::Matrix3f & KInv(int level = 0) const;
	/** Returns K(0, 0). */
	inline float getfx(int level = 0) const;
	/** Returns K(1, 1). */
	inline float getfy(int level = 0) const;
	/** Returns K(0, 2). */
	inline float getcx(int level = 0) const;
	/** Returns K(1, 2). */
	inline float getcy(int level = 0) const;
	/** Returns KInv(0, 0). */
	inline float fxInv(int level = 0) const;
	/** Returns KInv(1, 1). */
	inline float fyInv(int level = 0) const;
	/** Returns KInv(0, 2). */
	inline float cxInv(int level = 0) const;
	/** Returns KInv(1, 2). */
	inline float cyInv(int level = 0) const;
	/** Returns the frame's recording timestamp. */
	inline double timeStamp() const;

	inline SE3& pose();
	inline bool hasTrackingParent() const;
	inline Frame* getTrackingParent() const;
	PoseStruct * poseStruct;
	KeyPointStruct* keyPoints;

	struct Data
	{
		int id;
		int width[NUM_PYRS], height[NUM_PYRS];
		float fx[NUM_PYRS], fy[NUM_PYRS];
		float cx[NUM_PYRS], cy[NUM_PYRS];
		float fxInv[NUM_PYRS], fyInv[NUM_PYRS];
		float cxInv[NUM_PYRS], cyInv[NUM_PYRS];
		double timeStamp;

		Eigen::Matrix3f K[NUM_PYRS];
		Eigen::Matrix3f KInv[NUM_PYRS];

		cv::Mat image;
		cv::Mat depth;

	} data;

	std::unordered_set<Frame*, std::hash<Frame*>> neighbors;
};

inline SE3& Frame::pose()
{
	return poseStruct->camToWorld;
}

inline int Frame::id() const
{
	return data.id;
}

inline int Frame::width(int level) const
{
	return data.width[level];
}

inline int Frame::height(int level) const
{
	return data.height[level];
}

inline int Frame::totalPixel(int level) const
{
	return data.width[level] * data.height[level];
}

inline const Eigen::Matrix3f& Frame::K(int level) const
{
	return data.K[level];
}

inline const Eigen::Matrix3f& Frame::KInv(int level) const
{
	return data.KInv[level];
}

inline float Frame::getfx(int level) const
{
	return data.fx[level];
}

inline float Frame::getfy(int level) const
{
	return data.fy[level];
}

inline float Frame::getcx(int level) const
{
	return data.cx[level];
}

inline float Frame::getcy(int level) const
{
	return data.cy[level];
}

inline float Frame::fxInv(int level) const
{
	return data.fxInv[level];
}

inline float Frame::fyInv(int level) const
{
	return data.fyInv[level];
}

inline float Frame::cxInv(int level) const
{
	return data.cxInv[level];
}

inline float Frame::cyInv(int level) const
{
	return data.cyInv[level];
}

inline double Frame::timeStamp() const
{
	return data.timeStamp;
}

inline bool Frame::hasTrackingParent() const
{
	return (poseStruct->parentPose != 0);
}

inline Frame* Frame::getTrackingParent() const
{
	return poseStruct->parentPose->frame;
}
