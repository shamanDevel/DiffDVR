#include <enoki/stl.h>

#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/spectrum.h>
#include <mitsuba/render/texture3d.h>

#if defined(MTS_ENABLE_AUTODIFF)
#include <optix.h>
#endif

NAMESPACE_BEGIN(mitsuba)

/**
 * Purely absorptive heterogeneous medium.
 */
class HeterogeneousAbsorptiveMedium final : public Medium {
public:
    HeterogeneousAbsorptiveMedium(const Properties &props) : Medium(props) {
        m_sigma_t       = props.spectrum("sigma_t", 1.0f);
        m_density_scale = props.float_("density_scale", 1.f);
        m_density_tex   = props.texture3d("density", 1.0f);

        bool use_raymarching = props.bool_("use_raymarching", true);
        if (!use_raymarching)
            NotImplementedError("HeterogeneousAbsorptiveMedium without ray marching.");
    }

    template <typename Interaction3, typename Value = typename Interaction3::Value>
    Value density_impl(const Interaction3 &it, mask_t<Value> active) const {
        // TODO: Support spectrally varying density here
        return m_density_scale * m_density_tex->eval(it, active).x();
    }

    /// Returns (step size, max steps)
    MTS_INLINE auto get_step_size() const {
        auto density_bbox = m_density_tex->bbox();
        Vector3f diag = density_bbox.max - density_bbox.min;
        Float diag_norm = norm(diag);
        Vector3f res  = enoki::max(Vector3f(m_density_tex->resolution()), 3);

        Float step_size = hmin(diag * rcp(res));
        int max_steps = diag_norm / step_size + 1;
        return std::make_pair(step_size, max_steps);
    }

    template <
        typename Ray, typename Point3 = typename Ray::Point, typename Value = typename Ray::Value,
        typename Point2 = typename mitsuba::Point<Value, 2>, typename Spectrum = Spectrum<Value>,
        typename SurfaceInteraction3 = SurfaceInteraction<Point3>,
        typename Index               = uint_array_t<Value>>
    MTS_INLINE auto sample_distance_impl(const Scene *scene, const Ray &ray, const Point2 &sample,
                                         Sampler *sampler, Index index,
                                         mask_t<Value> active) const {
        SurfaceInteraction3 si = scene->ray_intersect(ray, active);
        auto [mi, throughput]  = sample_distance_with_si(si, ray, sample, sampler, index, active);
        masked(si.t, mi.is_valid()) = math::Infinity;
        return std::make_tuple(si, mi, throughput);
    }

    template <typename Ray, typename Point3 = typename Ray::Point,
              typename Value = typename Ray::Value, typename Spectrum = Spectrum<Value>,
              typename Point2              = typename mitsuba::Point<Value, 2>,
              typename SurfaceInteraction3 = SurfaceInteraction<Point3>,
              typename Index               = uint_array_t<Value>>
    MTS_INLINE auto sample_distance_with_si_impl(const SurfaceInteraction3 &si, const Ray &ray,
                                                 const Point2 &sample, Sampler * /*sampler*/,
                                                 Index /*index*/, mask_t<Value> active) const {
        using Vector3            = mitsuba::Vector<Value, 3>;
        using Frame              = Frame<Vector3>;
        using Mask               = mask_t<Value>;
        using Interaction3       = mitsuba::Interaction<Point3>;
        using MediumInteraction3 = mitsuba::MediumInteraction<Point3>;

        auto [step_size, max_steps] = get_step_size();
        Float half_step_size = 0.5f * step_size;

        Value mean_sigma_t                    = enoki::mean(m_sigma_t->eval(ray.wavelengths));
        auto [aabb_its, aabb_mint, aabb_maxt] = m_density_tex->bbox().ray_intersect(ray);
        Value mint                            = enoki::max(ray.mint, aabb_mint);
        Value maxt                            = enoki::min(si.t, enoki::min(ray.maxt, aabb_maxt));
        active &= aabb_its;

        Value desired_density = -enoki::log(1 - sample.x());

        // March along the given ray direction up to reaching:
        // - `si.t` or the bounding box,
        // - or accumulating at least `desired_density`. In that case,
        //   the particle is absorbed.
        Interaction3 lookup_it(mint, ray.time, ray.wavelengths, ray(mint));

        Value t_a                = mint;
        lookup_it.p              = ray(t_a);
        Value f_a                = density(lookup_it, active) * mean_sigma_t;
        Value integrated_density = zero<Value>();
        Mask reached_density     = false;

        // Jitter: Perform first step outside of loop
        Value t_b         = t_a + sample.y() * step_size;
        lookup_it.p       = ray(t_b);
        Value f_b         = density(lookup_it, active) * mean_sigma_t;
        Value new_density = 0.5f * (t_b - t_a) * (f_a + f_b);
        reached_density |= active && (new_density >= desired_density);
        // reached_density |= (new_density >= desired_density);
        active &= !(active && (reached_density || (t_b > maxt)));
        masked(integrated_density, active) = new_density;
        masked(t_a, active)                = t_b;
        masked(f_a, active)                = f_b;
        mint                               = t_b;

        for (int i = 1; i < max_steps; ++i) {
            masked(t_b, active) = fmadd(i, step_size, mint);
            lookup_it.p         = ray(t_b);
            masked(f_b, active) = density(lookup_it, active) * mean_sigma_t;

            Value new_density = fmadd(half_step_size, f_a + f_b, integrated_density);

            reached_density |= active && (new_density >= desired_density);
            // reached_density |= (new_density >= desired_density);
            active &= !reached_density && (t_b <= maxt);

            masked(integrated_density, active) = new_density;
            masked(t_a, active)                = t_b;
            masked(f_a, active)                = f_b;
        }

        // Solve quadratic equation to get exact intersection location
        Value a = 0.5f * (f_b - f_a);
        Value b = f_a;
        Value c = (integrated_density - desired_density) * rcp(t_b - t_a);
        auto [has_solution, solution1, solution2] = math::solve_quadratic(a, b, c);
        // Assert(none(active && reached_density && !has_solution));
        Value interp = select(solution1 >= 0.f && solution1 <= 1.f, solution1, solution2);
        Value sampled_t = t_a + (t_b - t_a) * interp;
        // Assert(none(active && reached_density && (interp < 0.f || interp > 1.f)));
        // Value f_c = (1 - interp) * f_a + interp * f_b;

        // Update integrated_density for rays which do not interact with the medium
        Mask escaped =
            (!reached_density && (t_b > maxt)) || (reached_density && (sampled_t > maxt));
        lookup_it.p = ray(maxt - math::Epsilon);
        masked(integrated_density, escaped) +=
            0.5f * (f_a + density(lookup_it, active) * mean_sigma_t);

        // Record medium interaction if the generated "t" is within the range of valid "t"
        Mask valid_mi = reached_density && (sampled_t <= maxt);
        // Assert(none(valid_mi && f_c <= 0.f));

        // Create mi for points that didn't escape the medium
        auto mi = zero<MediumInteraction3>();
        mi.sh_frame    = Frame(ray.d);
        mi.time        = ray.time;
        mi.wavelengths = ray.wavelengths;
        mi.t           = select(valid_mi, detach(sampled_t), math::Infinity);
        masked(integrated_density, valid_mi) = desired_density;
        masked(mi.p, valid_mi) = ray(mi.t);
        masked(mi.medium, valid_mi) = this_pointer<Value>();


        // Compute transmittance
        // TODO: support spectrally-varying absorption better
        // Spectrum sigma_t_color = m_sigma_t->eval(ray.wavelengths);
        Spectrum tr            = exp(-integrated_density);
        masked(integrated_density, valid_mi) = desired_density;
        masked(mi.t, valid_mi) = detach(sampled_t);
        masked(mi.p, valid_mi) = ray(mi.t);
        masked(mi.medium, valid_mi) = this_pointer<Value>();

        Spectrum throughput = select(tr > 0.f, tr / detach(tr), 1.0f);
        // Absorbed rays.
        masked(throughput, valid_mi || !escaped) = 0.f;

        return std::make_pair(mi, throughput);
    }


    template <typename Ray, typename Value = typename Ray::Value>
    MTS_INLINE Value integrate_density(const Ray &ray, mask_t<Value> active) const {
        using Mask         = mask_t<Value>;
        using Interaction3 = mitsuba::Interaction<Point<Value, 3>>;

        Value mean_sigma_t = enoki::mean(m_sigma_t->eval(ray.wavelengths));

        // We must assume the ray is well-adjusted.
        Value maxt = ray.maxt;
        Value mint = ray.mint;
        active &= maxt > mint;

        Interaction3 lookup_it;
        lookup_it.p           = ray(mint);
        lookup_it.wavelengths = ray.wavelengths;
        lookup_it.time        = ray.time;

        auto [step_size, max_steps] = get_step_size();
        Value t_a                   = mint;
        lookup_it.p                 = ray(t_a);
        Value f_a                   = density(lookup_it, active) * mean_sigma_t;

        Value result = zero<Value>();
        for (int i = 1; i < max_steps; ++i) {
            Value t_b                 = fmadd(i, step_size, mint);
            Mask reached_maxt         = t_b >= maxt;
            masked(t_b, reached_maxt) = maxt;
            lookup_it.p               = ray(t_b);
            Value f_b                 = density(lookup_it, active) * mean_sigma_t;

            // result += select(active, 0.5f * (f_a + f_b) * (t_b - t_a), 0.f);
            result += 0.5f * (f_a + f_b) * (t_b - t_a);
            active &= !reached_maxt;
            t_a = t_b;
            f_a = f_b;
        }
        return result;
    }

    template <typename Ray, typename Point3 = typename Ray::Point,
              typename Value = typename Ray::Value>
    MTS_INLINE auto eval_transmittance_impl(const Ray &ray, Sampler * /*unused*/,
                                            uint_array_t<Value> /*unused*/,
                                            mask_t<Value> active) const {
        Value integrated_density = integrate_density(ray, active);
        return Spectrum<Value>(enoki::exp(-integrated_density));
    }

    template <typename Ray3, typename Value = typename Ray3::Value>
    Value pdf_distance_impl(Ray3 ray, Value t, mask_t<Value> active) const {
        // Adjust ray to make sure that we integrate along the desired distance `t`
        // We assume that `t` starts from the given ray's origin.
        ray.mint = 0;
        ray.maxt = t;
        auto integrated_density = integrate_density(ray, active);
        return exp(-integrated_density);
    }

    template <typename MediumInteraction3, typename Value = typename MediumInteraction3::Value>
    MTS_INLINE auto eval_scattering_impl(const MediumInteraction3 & /*mi*/,
                                         mask_t<Value> /*active*/) const {
        // No scattering
        return Spectrum<Value>(0.f) + 0.f;  // Workaround
    }

    template <typename Interaction3, typename Value = typename Interaction3::Value>
    MTS_INLINE auto eval_albedo_impl(const Interaction3 & /*it*/, mask_t<Value> /*active*/) const {
        // No scattering -> zero albedo
        return Spectrum<Value>(0.f) + 0.f;  // Workaround
    }

    // bool is_homogeneous() const override { return true; }
    bool is_homogeneous() const override { return false; }

    std::vector<ref<Object>> children() override {
        if (m_sigma_t->id().empty())
            m_sigma_t->set_id(this->id() + "_sigma_t");
        if (m_density_tex->id().empty())
            m_density_tex->set_id(this->id() + "_density");
        return { m_sigma_t.get(), m_density_tex.get() };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HeterogeneousAbsorptiveMedium[" << std::endl
            << "  m_sigma_t = " << string::indent(m_sigma_t) << std::endl
            << "  m_density = " << string::indent(m_density_tex) << std::endl
            << "]";
        return oss.str();
    }

#if defined(MTS_ENABLE_AUTODIFF)
    void set_labels(const std::string &prefix) const override {
        Medium::set_labels(prefix);

        m_sigma_t->set_labels(prefix + ".m_sigma_t");
        m_density_tex->set_labels(prefix + ".m_density_tex");
    }

    void set_globals(const std::string &prefix, const RTcontext &context,
                     const RTmaterial &material) const override {
        Medium::set_globals(prefix, context, material);
        m_sigma_t->set_globals(prefix + "_m_sigma_t", context, material);
        m_density_tex->set_globals(prefix + "_m_density_tex", context, material);


        // Extra variables to help with re-implementation
        RTvariable var = get_or_create_optix_var(prefix + "_step_size", context, material);
        rtVariableSet1f(var, get_step_size().first);

        auto rgb_sigma_t = m_sigma_t->eval(Spectrumf(1.f));
        var = get_or_create_optix_var(prefix + "_rgb_sigma_t", context, material);
        rtVariableSet3f(var, rgb_sigma_t.x(), rgb_sigma_t.y(), rgb_sigma_t.z());
    }
#endif

    MTS_IMPLEMENT_MEDIUM_ALL()
    MTS_DECLARE_CLASS()
private:
    ref<ContinuousSpectrum> m_sigma_t;
    ref<Texture3D> m_density_tex;
    Float m_density_scale;
};

MTS_IMPLEMENT_CLASS(HeterogeneousAbsorptiveMedium, Medium)
MTS_EXPORT_PLUGIN(HeterogeneousAbsorptiveMedium, "Homogeneous absorptive medium")

NAMESPACE_END(mitsuba)
