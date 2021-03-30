#include "volume.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>

/**
 * Source:
 * Real-Time Ray-Tracing of Implicit Surfaces on the GPU, Singh, Narayanan, 2007
 */

BEGIN_RENDERER_NAMESPACE

namespace {
	static inline float sqr(float s) { return s * s; }
	static inline float cb(float s) { return s * s * s; }

	template <typename K1, typename K2, typename V>
	V get_or(const  std::unordered_map <K1, V>& m, const K2& key, const V& defval) {
		auto it = m.find(key);
		if (it == m.end()) {
			return defval;
		}
		else {
			return it->second;
		}
	}
	
	struct ImplicitEquation_MARSCHNER_LOBB
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			double fM = get_or(params, "fM", 6.0f);
			double alpha = get_or(params, "alpha", 0.25f);
			double r = sqrt(x * x + y * y);
			double pr = cos(2 * M_PI * fM * cos(M_PI * r / 2));
			double num = (1 - sin(M_PI * z / 2)) + alpha * (1 + pr);
			double denom = 2 * (1 + alpha);
			return num / denom;
		}
	};

	struct ImplicitEquation_CUBE
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float scale = get_or(params, "scale", 0.5f);
			float d = std::sqrt(
				sqr(std::max(0.0f, std::abs(x) - scale)) +
				sqr(std::max(0.0f, std::abs(y) - scale)) +
				sqr(std::max(0.0f, std::abs(z) - scale)));
			return 1 - d;
		}
	};

	struct ImplicitEquation_SPHERE
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			return 1 - std::sqrt(sqr(x) + sqr(y) + sqr(z));
		}
	};

	struct ImplicitEquation_INVERSE_SPHERE
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			return std::sqrt(sqr(x) + sqr(y) + sqr(z));
		}
	};

	static inline float implicit2Density(float i)
	{
		//implicit functions are at i==0.
		//Transform to 0.5 and clamp at 0 and 1
		return std::max(0.0f, std::min(1.0f, -i + 0.5f));
	}

	struct ImplicitEquation_DING_DONG
	{
		static constexpr float boxMin = -2;
		static constexpr float boxMax = +2;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			return implicit2Density(x * x + y * y - z * (1 - z * z));
		}
	};

	struct ImplicitEquation_ENDRASS //octic
	{
		static constexpr float boxMin = -2;
		static constexpr float boxMax = +2;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float a = sqr(x + y) - 2;
			float b = sqr(x - y) - 2;
			float c = -4 * (1 - sqrtf(2));
			float d = 8 * (2 - sqrtf(2)) * z * z + 2 * (2 - 7 * sqrtf(2)) * (x * x + y * y);
			float e = -16 * sqr(sqr(z)) + 8 * (1 + 2 * sqrtf(2)) * sqr(z) - 1 + 12 * sqrtf(2);
			return 0.5f+(64*(x*x-1)*(y*y-1)*a*b - sqr(c+d+e));
		}
	};

	struct ImplicitEquation_BARTH //sextic
	{
		static constexpr float boxMin = -1.5f;
		static constexpr float boxMax = +1.5f;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			z += 0.5f;
			float phi = (1 + sqrtf(5)) / 2;
			float x2 = x * x, y2 = y * y, z2 = z * 2, phi2 = phi * phi;
			return 0.5f+(
				4*(phi2*x2-y2)*(phi2*y2-z2)*(phi2*z2-x2) - (1+2*phi)*sqr(x2+y2+z2-1));
		}
	};

	struct ImplicitEquation_HEART //sextic
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float x2 = x * x, y2 = y * y, z2 = z * 2;
			return implicit2Density(cb(2*x2+2*y2+z2-1) - 0.1f*x2*z2*z - y2*z2*z);
		}
	};

	struct ImplicitEquation_KLEINE //sextic
	{
		static constexpr float boxMin = -5;
		static constexpr float boxMax = +5;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float x2 = x * x, y2 = y * y, z2 = z * 2;
			return 0.5f + -(
				(x2+y2+z2+2*y-1)*sqr(x2+y2+z2-2*y-1)-8*z2+16*x*y*(x2+y2+z2-2*y-1));
		}
	};

	struct ImplicitEquation_CASSINI //quartic
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float a = get_or(params, "a", 0.25f);
			return implicit2Density(
				(sqr(x+a)+y*y)*(sqr(x-a)+y*y) - z*z);
		}
	};

	struct ImplicitEquation_STEINER
	{
		static constexpr float boxMin = -0.5f;
		static constexpr float boxMax = +0.5f;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float x2 = x * x, y2 = y * y, z2 = z * 2;
			return implicit2Density(
				x2*y2+x2*z2+y2*z2-2*x*y*z);
		}
	};

	struct ImplicitEquation_CROSS_CAP
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float x2 = x * x, y2 = y * y, z2 = z * 2;
			return implicit2Density(
				4*x2*(x2+y2*z2+z)+y2*(y2+z2-1));
		}
	};

	struct ImplicitEquation_KUMMER
	{
		static constexpr float boxMin = -2;
		static constexpr float boxMax = +2;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float x2 = x * x, y2 = y * y, z2 = z * 2;
			return implicit2Density(
				x2*x2+y2*y2+z2*z2-x2-y2-z2-x2*y2-y2*z2-z2*x2+1);
		}
	};

	struct ImplicitEquation_BLOBBY
	{
		static constexpr float boxMin = -2;
		static constexpr float boxMax = +2;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float x2 = x * x, y2 = y * y, z2 = z * 2;
			return implicit2Density(
				x2+y2*z2+sin(4*x)-cos(4*y)+sin(4*z)-1);
		}
	};

	struct ImplicitEquation_TUBE
	{
		static constexpr float boxMin = -1;
		static constexpr float boxMax = +1;
		static float eval(float x, float y, float z, const std::unordered_map<std::string, float>& params)
		{
			float r = sqrtf(y * y + z * z);
			return ((1 - (r * cb(0.9f - 0.5f * cosf(7 * x)))) - 0.9f) * 10;
		}
	};

}
	
std::unique_ptr<renderer::Volume> renderer::Volume::createImplicitDataset(int resolution, ImplicitEquation equation,
	const std::unordered_map<std::string, float>& params)
{
	float boxMin, boxMax;
	ImplicitFunction_t f;

	// NOLINT(cppcoreguidelines-macro-usage)
#define CASE(name) \
	case ImplicitEquation::name:		\
		boxMin = ImplicitEquation_ ## name::boxMin;		\
		boxMax = ImplicitEquation_ ## name::boxMax;		\
		f = [&params](float3 v) {return ImplicitEquation_ ## name ::eval(v.x, v.y, v.z, params); };	\
		break
	
	switch (equation)
	{
		CASE(MARSCHNER_LOBB);
		CASE(CUBE);
		CASE(SPHERE);
		CASE(INVERSE_SPHERE);
		CASE(DING_DONG);
		CASE(ENDRASS);
		CASE(BARTH);
		CASE(HEART);
		CASE(KLEINE);
		CASE(CASSINI);
		CASE(STEINER);
		CASE(CROSS_CAP);
		CASE(KUMMER);
		CASE(BLOBBY);
		CASE(TUBE);
	default:
		return {};
	}
	return createSyntheticDataset(resolution, boxMin, boxMax, f);
}

END_RENDERER_NAMESPACE
