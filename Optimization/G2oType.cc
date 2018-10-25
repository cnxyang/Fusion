#include "G2oType.h"
#include <g2o/core/factory.h>
#include <g2o/stuff/macros.h>

G2O_USE_TYPE_GROUP(sba);

G2O_REGISTER_TYPE_GROUP(se3sophus);

G2O_REGISTER_TYPE(VERTEX_SE3_SOPHUS:EXPMAP, VertexSE3);
G2O_REGISTER_TYPE(EDGE_SE3_SOPHUS:EXPMAP, EdgeSE3);

VertexSE3::VertexSE3() : g2o::BaseVertex<6, Sophus::SE3d>() {
	_marginalized = false;
}

bool VertexSE3::write(std::ostream & os) const {
	// TODO
	assert(false);
	return false;
}

bool VertexSE3::read(std::istream & is) {
	// TODO
	assert(false);
	return false;
}

EdgeSE3::EdgeSE3() : g2o::BaseBinaryEdge<6, Sophus::SE3d, VertexSE3, VertexSE3>() {
}

bool EdgeSE3::write(std::ostream& os) const {
	assert(false);
	return false;
}

bool EdgeSE3::read(std::istream& is) {
	assert(false);
	return false;
}
