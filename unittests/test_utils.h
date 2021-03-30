#pragma once

#include <Eigen/Core>
#include <iomanip>

#include "renderer_settings.cuh"

typedef Eigen::Matrix<real_t, 3, 1, 0> Vector3r;
typedef Eigen::Matrix<real_t, 4, 1, 0> Vector4r;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, 1, 0> VectorXr;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, 0> MatrixXr;

namespace std
{
	inline std::ostream& operator<<(std::ostream& o, const real4& v)
	{
		o << std::fixed << std::setw(5) << std::setprecision(3)
			<< v.x << "," << v.y << "," << v.z << "," << v.w;
		return o;
	}
}

inline Vector3r toEigen(const real3& v)
{
	return Vector3r(v.x, v.y, v.z);
}
inline Vector4r toEigen(const real4& v)
{
	return Vector4r(v.x, v.y, v.z, v.w);
}
inline real3 fromEigen3(const VectorXr& v)
{
	CHECK(v.size() == 3);
	return make_real3(v[0], v[1], v[2]);
}
inline real4 fromEigen4(const VectorXr& v)
{
	CHECK(v.size() == 4);
	return make_real4(v[0], v[1], v[2], v[3]);
}

struct empty {};
