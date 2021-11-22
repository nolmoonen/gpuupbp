// Copyright (C) 2021, Nol Moonen
// Copyright (C) 2014, Petr Vevoda, Martin Sik (http://cgg.mff.cuni.cz/~sik/),
// Tomas Davidovic (http://www.davidovic.cz),
// Iliyan Georgiev (http://www.iliyan.com/),
// Jaroslav Krivanek (http://cgg.mff.cuni.cz/~jaroslav/)
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// (The above is MIT License: http://en.wikipedia.origin/wiki/MIT_License)

#ifndef KERNEL_BSDF_CUH
#define KERNEL_BSDF_CUH

#include "phase_function.cuh"
#include "scene.cuh"

#include <cuda_runtime.h>

/**
 * BSDF definition
 *
 * One of important conventions is prefixing direction with world_ when
 * are in world coordinates and with local_ when they are in local frame.
 *
 * Another important convention is the suffix _fix and _gen.
 * For PDF computation, we need to know which direction is given (_fix),
 * and which is the generated (_gen) direction. This is important even
 * when simply evaluating BSDF.
 * In BPT, we call evaluate() when directly connecting to light/camera.
 * This gives us both directions required for evaluating BSDF.
 * However, for MIS we also need to know probabilities of having sampled
 * this path via BSDF sampling, and we need that for both possible directions.
 * The _fix/_gen convention (along with _for and _rev for PDF) clearly
 * establishes which PDF is which.
 *
 * Used both for events on surface as well as in medium. Depending on
 * constructor parameters BSDF knows, if it is on surface or in medium.
 * Then each of its methods begin with if and continue with appropriate code.
 * Surface events are handled directly in this class, for scattering in media
 * phase_function methods are called.
 *
 * One important distinction between the thesis and the code is that the PDFs
 * returned are w.r.t. the solid angle measure, which equals the non-projected
 * solid angle measure used in the thesis pr_W times the factor D. Hence, in
 * some places in the updating of the recursive quantities a D factor will
 * appear missing, but it is in fact included in the directional sampling PDF.*/

__forceinline__ __device__ bool is_valid(const BSDF& bsdf) { return (bsdf.flags & BSDF::VALID) != 0; }

__forceinline__ __device__ bool is_delta(const BSDF& bsdf) { return (bsdf.flags & BSDF::DELTA) != 0; }

__forceinline__ __device__ bool is_in_medium(const BSDF& bsdf) { return (bsdf.flags & BSDF::IN_MEDIUM) != 0; }

__forceinline__ __device__ bool is_from_light(const BSDF& bsdf) { return (bsdf.flags & BSDF::DIR_FROM_LIGHT) != 0; }

__forceinline__ __device__ bool is_on_surface(const BSDF& bsdf) { return !is_in_medium(bsdf); }

__forceinline__ __device__ bool is_from_camera(const BSDF& bsdf) { return !is_from_light(bsdf); }

__forceinline__ __device__ BSDF::SrfData* srf(BSDF& bsdf)
{
    GPU_ASSERT(is_on_surface(bsdf));
    return &bsdf.data.srf_data;
}

__forceinline__ __device__ const BSDF::SrfData* srf(const BSDF& bsdf)
{
    GPU_ASSERT(is_on_surface(bsdf));
    return &bsdf.data.srf_data;
}

__forceinline__ __device__ BSDF::MedData* med(BSDF& bsdf)
{
    GPU_ASSERT(!is_on_surface(bsdf));
    return &bsdf.data.med_data;
}

__forceinline__ __device__ const BSDF::MedData* med(const BSDF& bsdf)
{
    GPU_ASSERT(!is_on_surface(bsdf));
    return &bsdf.data.med_data;
}

/// Cosine of angle between the incoming direction and the shading normal
/// for a surface.
__forceinline__ __device__ float cos_theta_fix(const BSDF& bsdf) { return srf(bsdf)->local_dir_fix.z; }

/// Attenuation factor D.
/// On surface, cosine of angle between the incoming direction and the normal.
/// In medium, one.
__forceinline__ __device__ float d_factor(const BSDF& bsdf)
{
    return is_on_surface(bsdf) ? fabsf(cos_theta_fix(bsdf)) : 1.f;
}

/// Incoming (fixed) direction, in world space.
__forceinline__ __device__ float3 world_dir_fix(const BSDF& bsdf)
{
    return is_on_surface(bsdf) ? bsdf.frame.to_world(srf(bsdf)->local_dir_fix) : med(bsdf)->world_dir_fix;
}

////////////////////////////////////////////////////////////////////////////////
// Albedo methods
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float albedo_diffuse(BSDF& bsdf, const Material& material)
{
    return luminance(material.diffuse_reflectance);
}

__forceinline__ __device__ float albedo_phong(BSDF& bsdf, const Material& material)
{
    return luminance(material.phong_reflectance);
}

__forceinline__ __device__ float albedo_reflect(BSDF& bsdf, const Material& material)
{
    return luminance(material.mirror_reflectance);
}

__forceinline__ __device__ float albedo_refract(BSDF& bsdf, const Material& material)
{
    // refractive albedo is 1 only for materials with IOR
    return material.ior > 0.f ? 1.f : 0.f;
}

__forceinline__ __device__ void get_component_probabilities(BSDF& bsdf, const Material& material)
{
    GPU_ASSERT(is_on_surface(bsdf));
    srf(bsdf)->reflect_coeff = fresnel_dielectric(srf(bsdf)->local_dir_fix.z, srf(bsdf)->ior);

    const float al_diffuse = albedo_diffuse(bsdf, material);
    const float al_phong = albedo_phong(bsdf, material);
    const float al_reflect = srf(bsdf)->reflect_coeff * albedo_reflect(bsdf, material);
    const float al_refract = (1.f - srf(bsdf)->reflect_coeff) * albedo_refract(bsdf, material);

    const float total_albedo = al_diffuse + al_phong + al_reflect + al_refract;

    if (total_albedo < 1e-9f) {
        srf(bsdf)->diff_prob = 0.f;
        srf(bsdf)->phong_prob = 0.f;
        srf(bsdf)->refl_prob = 0.f;
        srf(bsdf)->refr_prob = 0.f;
        bsdf.continuation_prob = 0.f;
    } else {
        srf(bsdf)->diff_prob = al_diffuse / total_albedo;
        srf(bsdf)->phong_prob = al_phong / total_albedo;
        srf(bsdf)->refl_prob = al_reflect / total_albedo;
        srf(bsdf)->refr_prob = al_refract / total_albedo;

        float3 tmp = material.diffuse_reflectance + material.phong_reflectance +
                     srf(bsdf)->reflect_coeff * material.mirror_reflectance;
        float max_tmp = fmaxf(tmp.x, fmaxf(tmp.y, tmp.z));

        // the continuation probability is max component from reflectance.
        // that way the weight of sample will never rise.
        // luminance is another very valid option.
        bsdf.continuation_prob = max_tmp + (1.f - srf(bsdf)->reflect_coeff);
        bsdf.continuation_prob = fminf(1.f, fmaxf(0.f, bsdf.continuation_prob));
    }
}

__forceinline__ __device__ void bsdf_setup(BSDF& bsdf,
                                           const Ray& ray,
                                           const Intersection& isect,
                                           const Scene& scene,
                                           const BSDF::BSDFDirection direction = BSDF::FROM_CAMERA,
                                           const float ior = -1.f)
{
    bsdf.flags = 0;

    if (direction == BSDF::FROM_LIGHT) bsdf.flags |= BSDF::DIR_FROM_LIGHT;

    if (isect.is_on_surface()) {
        const Material& mat = get_material(scene, isect.mat_id);
        srf(bsdf)->mat_ptr = &mat;
        bsdf.frame.set_from_z(isect.shading_normal);
        srf(bsdf)->local_dir_fix = bsdf.frame.to_local(-ray.direction);

        srf(bsdf)->geometry_normal = bsdf.frame.to_local(isect.normal);
        srf(bsdf)->cos_theta_fix_geom = dot(srf(bsdf)->local_dir_fix, srf(bsdf)->geometry_normal);

        // Reject rays that are too parallel with tangent plane
        if (fabsf(cos_theta_fix(bsdf)) < EPS_COSINE || fabsf(srf(bsdf)->cos_theta_fix_geom) < EPS_COSINE) {
            bsdf.flags = 0;
            return;
        }

        srf(bsdf)->ior = ior;

        get_component_probabilities(bsdf, mat);
        if (srf(bsdf)->diff_prob == 0.f && srf(bsdf)->phong_prob == 0.f) {
            bsdf.flags |= bsdf.DELTA;
        }
    } else {
        bsdf.flags |= bsdf.IN_MEDIUM;
        med(bsdf)->med_ptr = get_medium_ptr(scene, isect.med_id);
        med(bsdf)->world_dir_fix = -ray.direction;
        bsdf.frame.set_from_z(-med(bsdf)->world_dir_fix);
        bsdf.continuation_prob = med(bsdf)->med_ptr->continuation_prob;
        med(bsdf)->mean_cosine = med(bsdf)->med_ptr->mean_cosine;
        med(bsdf)->scatter_coef = med(bsdf)->med_ptr->scattering_coeff;
        med(bsdf)->scatter_coef_is_pos = is_black_or_negative(med(bsdf)->med_ptr->scattering_coeff) ? false : true;
    }

    bsdf.flags |= BSDF::VALID;
}

////////////////////////////////////////////////////////////////////////////////
// Evaluation methods
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float3 evaluate_diffuse(const BSDF& bsdf,
                                                   const Material& material,
                                                   const float3& local_dir_gen,
                                                   float* for_pdf_w = NULL,
                                                   float* rev_pdf_w = NULL)
{
    if (srf(bsdf)->diff_prob == 0) return make_float3(0.f);

    if (srf(bsdf)->local_dir_fix.z < EPS_COSINE || local_dir_gen.z < EPS_COSINE) {
        return make_float3(0.f);
    }

    if (for_pdf_w) {
        *for_pdf_w += srf(bsdf)->diff_prob * fmaxf(0.f, local_dir_gen.z * INV_PI_F);
    }

    if (rev_pdf_w) {
        *rev_pdf_w += srf(bsdf)->diff_prob * fmaxf(0.f, srf(bsdf)->local_dir_fix.z * INV_PI_F);
    }

    return material.diffuse_reflectance * INV_PI_F;
}

__forceinline__ __device__ float3 evaluate_phong(const BSDF& bsdf,
                                                 const Material& material,
                                                 const float3& local_dir_gen,
                                                 float* for_pdf_w = NULL,
                                                 float* rev_pdf_w = NULL)
{
    if (srf(bsdf)->phong_prob == 0) return make_float3(0.f);

    if (srf(bsdf)->local_dir_fix.z < EPS_COSINE || local_dir_gen.z < EPS_COSINE) {
        return make_float3(0.f);
    }

    // assumes this is never called when rejectShadingCos(local_dir_gen.z) is true
    const float3 refl_local_dir_in = reflect_local(srf(bsdf)->local_dir_fix);
    const float dot_r_wi = dot(refl_local_dir_in, local_dir_gen);

    if (dot_r_wi <= EPS_PHONG) return make_float3(0.f);

    float pow_ = powf(dot_r_wi, material.phong_exponent);
    if (!is_positive(pow_)) return make_float3(0.f);

    if (for_pdf_w || rev_pdf_w) {
        // the sampling is symmetric
        const float pdf_w = srf(bsdf)->phong_prob *
                            power_cos_hemisphere_pdf_w(refl_local_dir_in, local_dir_gen, material.phong_exponent);

        if (for_pdf_w) *for_pdf_w += pdf_w;

        if (rev_pdf_w) *rev_pdf_w += pdf_w;
    }

    const float3 rho = material.phong_reflectance * (material.phong_exponent + 2.f) * .5f * INV_PI_F;

    return rho * pow_;
}

////////////////////////////////////////////////////////////////////////////////
// PDF methods
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ void
pdf_diffuse(const BSDF& bsdf, const float3& local_dir_gen, float* for_pdf_w = NULL, float* rev_pdf_w = NULL)
{
    if (srf(bsdf)->diff_prob == 0) return;

    if (for_pdf_w) {
        *for_pdf_w += srf(bsdf)->diff_prob * fmaxf(0.f, local_dir_gen.z * INV_PI_F);
    }

    if (rev_pdf_w) {
        *rev_pdf_w += srf(bsdf)->diff_prob * fmaxf(0.f, srf(bsdf)->local_dir_fix.z * INV_PI_F);
    }
}

__forceinline__ __device__ void pdf_phong(const BSDF& bsdf,
                                          const Material& material,
                                          const float3& local_dir_gen,
                                          float* for_pdf_w = NULL,
                                          float* rev_pdf_w = NULL)
{
    if (srf(bsdf)->phong_prob == 0) return;

    // assumes this is never called when rejectShadingCos(local_dir_gen.z) is true
    const float3 refl_local_dir_in = reflect_local(srf(bsdf)->local_dir_fix);
    const float dot_r_wi = dot(refl_local_dir_in, local_dir_gen);

    if (dot_r_wi <= EPS_PHONG) return;

    if (for_pdf_w || rev_pdf_w) {
        // the sampling is symmetric
        const float pdf_w = power_cos_hemisphere_pdf_w(refl_local_dir_in, local_dir_gen, material.phong_exponent) *
                            srf(bsdf)->phong_prob;

        if (for_pdf_w) *for_pdf_w += pdf_w;

        if (rev_pdf_w) *rev_pdf_w += pdf_w;
    }
}

/// Given a direction, evaluates BSDF
/// Returns value of BSDF, as well as cosine for the
/// world_dir_gen direction.
/// Can return probability (w.r.t. solid angle W),
/// of having sampled world_dir_gen given local_dir_fix (for_pdf_w),
/// and of having sampled local_dir_fix given world_dir_gen (sin_theta).
__forceinline__ __device__ float3 evaluate(const BSDF& bsdf,
                                           /// Points away from the scattering location.
                                           const float3& world_dir_gen,
                                           float& cos_theta_gen,
                                           Material* materials,
                                           float* for_pdf_w = NULL,
                                           float* rev_pdf_w = NULL,
                                           float* sin_theta = NULL)
{
    if (is_on_surface(bsdf)) {
        // surface
        float3 result = make_float3(0.f);

        if (for_pdf_w) *for_pdf_w = 0.f;
        if (rev_pdf_w) *rev_pdf_w = 0.f;
        if (sin_theta) *sin_theta = 0.f;

        const float3 local_dir_gen = bsdf.frame.to_local(world_dir_gen);

        const float cos_theta_gen_ = local_dir_gen.z;
        const float cos_theta_gen_geom = dot(local_dir_gen, srf(bsdf)->geometry_normal);

        cos_theta_gen = fabsf(cos_theta_gen_);

        // samples too parallel with tangent plane are rejected
        if (cos_theta_gen < EPS_COSINE || fabsf(cos_theta_gen_geom) < EPS_COSINE) {
            return result;
        }

        // generated direction must point to the same side from the surface
        // as the fixed one (potential refraction has zero PDF anyway since
        // it is delta)
        if (cos_theta_gen_ * cos_theta_fix(bsdf) < 0.f || cos_theta_gen_geom * srf(bsdf)->cos_theta_fix_geom < 0.f) {
            return result;
        }

        const Material& mat = *srf(bsdf)->mat_ptr;
        result += evaluate_diffuse(bsdf, mat, local_dir_gen, for_pdf_w, rev_pdf_w);
        result += evaluate_phong(bsdf, mat, local_dir_gen, for_pdf_w, rev_pdf_w);

        return result;
    } else {
        // medium
        cos_theta_gen = 1.f;

        // No need to evaluate phase function if the scattering coef is zero
        if (!med(bsdf)->scatter_coef_is_pos) {
            if (for_pdf_w) *for_pdf_w = 0.f;
            if (rev_pdf_w) *rev_pdf_w = 0.f;
            if (sin_theta) *sin_theta = 0.f;
            return make_float3(0.f);
        } else
            return med(bsdf)->scatter_coef * PhaseFunction::evaluate(med(bsdf)->world_dir_fix,
                                                                     world_dir_gen,
                                                                     med(bsdf)->mean_cosine,
                                                                     for_pdf_w,
                                                                     rev_pdf_w,
                                                                     sin_theta);
    }
}

/// Given a direction, evaluates PDF.
/// By default returns PDF with which would be aWorldDirGen generated from
/// mLocalDirFix. When aPdfDir == kReverse, it provides PDF for the
/// reverse direction.
__forceinline__ __device__ float pdf(const BSDF& bsdf,
                                     /// Points away from the scattering location,
                                     const float3& world_dir_gen,
                                     const Scene& scene,
                                     const BSDF::PdfDir pdf_dir = BSDF::FORWARD)
{
    if (is_on_surface(bsdf)) {
        // surface
        const float3 local_dir_gen = bsdf.frame.to_local(world_dir_gen);

        const float cos_theta_gen = local_dir_gen.z;
        const float cos_theta_gen_geom = dot(local_dir_gen, srf(bsdf)->geometry_normal);

        // Samples too parallel with tangent plane are rejected
        if (fabsf(cos_theta_gen) < EPS_COSINE || fabsf(cos_theta_gen_geom) < EPS_COSINE) {
            return 0.f;
        }

        // Generated direction must point to the same side from the surface
        // as the fixed one (potential refraction has zero PDF anyway
        // since it is delta)
        if (cos_theta_gen * cos_theta_fix(bsdf) < 0.f || cos_theta_gen_geom * srf(bsdf)->cos_theta_fix_geom < 0.f) {
            return 0.f;
        }

        float for_pdf_w = 0.f;
        float rev_pdf_w = 0.f;

        const Material& mat = *srf(bsdf)->mat_ptr;

        pdf_diffuse(bsdf, local_dir_gen, &for_pdf_w, &rev_pdf_w);
        pdf_phong(bsdf, mat, local_dir_gen, &for_pdf_w, &rev_pdf_w);

        GPU_ASSERT(for_pdf_w > 0.f);
        GPU_ASSERT(rev_pdf_w > 0.f);

        // not possible
        return pdf_dir == BSDF::REVERSE ? rev_pdf_w : for_pdf_w;
    } else {
        // medium
        float pdf = PhaseFunction::pdf(med(bsdf)->world_dir_fix, world_dir_gen, med(bsdf)->mean_cosine);
        GPU_ASSERT(pdf > 0.f);

        return pdf;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Sampling methods
// All sampling methods take material, 2 random numbers [0-1],
// and return BSDF factor, generated direction in local coordinates, and PDF
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ float3
sample_diffuse(const BSDF& bsdf, const Material& material, const float2& rnd, float3& local_dir_gen, float& pdf_w)
{
    if (srf(bsdf)->local_dir_fix.z < EPS_COSINE) return make_float3(0.f);

    float unweighted_pdf_w;
    local_dir_gen = sample_cos_hemisphere_w(rnd, &unweighted_pdf_w);
    pdf_w += unweighted_pdf_w * srf(bsdf)->diff_prob;

    return material.diffuse_reflectance * INV_PI_F;
}

__forceinline__ __device__ float3
sample_phong(const BSDF& bsdf, const Material& material, const float2& rnd, float3& local_dir_gen, float& pdf_w)
{
    local_dir_gen = sample_power_cos_hemisphere_w(rnd, material.phong_exponent, NULL);

    // due to numeric issues in MIS, we actually need to compute all PDFs
    // exactly the same way all the time!
    const float3 refl_local_dir_fixed = reflect_local(srf(bsdf)->local_dir_fix);
    {
        Frame frame;
        frame.set_from_z(refl_local_dir_fixed);
        local_dir_gen = frame.to_world(local_dir_gen);
    }

    const float dot_r_wi = dot(refl_local_dir_fixed, local_dir_gen);

    if (dot_r_wi <= EPS_PHONG) return make_float3(0.f);

    pdf_phong(bsdf, material, local_dir_gen, &pdf_w);

    const float3 rho = material.phong_reflectance * (material.phong_exponent + 2.f) * .5f * INV_PI_F;

    return rho * powf(dot_r_wi, material.phong_exponent);
}

__forceinline__ __device__ float3
sample_reflect(const BSDF& bsdf, const Material& material, const float2& rnd, float3& local_dir_gen, float& pdf_w)
{
    local_dir_gen = reflect_local(srf(bsdf)->local_dir_fix);

    pdf_w += srf(bsdf)->refl_prob;

    // BSDF is multiplied (outside) by cosine (local_dir_gen.z),
    // for mirror this shouldn't be done, so we pre-divide here instead
    return srf(bsdf)->reflect_coeff * material.mirror_reflectance / fabsf(local_dir_gen.z);
}

__forceinline__ __device__ float3
sample_refract(const BSDF& bsdf, const Material& material, const float2& rnd, float3& local_dir_gen, float& pdf_w)
{
    if (srf(bsdf)->ior < 0) return make_float3(0);

    float cos_i = srf(bsdf)->local_dir_fix.z;

    float cos_t;
    float eta_inc_over_eta_trans = srf(bsdf)->ior;

    if (cos_i < 0.f) {
        // hit from inside
        cos_i = -cos_i;
        cos_t = 1.f;
    } else {
        cos_t = -1.f;
    }

    const float sin_i2 = 1.f - cos_i * cos_i;
    const float sin_t2 = eta_inc_over_eta_trans * eta_inc_over_eta_trans * sin_i2;

    if (sin_t2 < 1.f) {
        // no total internal reflection
        cos_t *= sqrtf(fmaxf(0.f, 1.f - sin_t2));

        local_dir_gen = make_float3(-eta_inc_over_eta_trans * srf(bsdf)->local_dir_fix.x,
                                    -eta_inc_over_eta_trans * srf(bsdf)->local_dir_fix.y,
                                    cos_t);

        pdf_w += srf(bsdf)->refr_prob;

        const float refract_coeff = 1.f - srf(bsdf)->reflect_coeff;

        // only camera paths are multiplied by this factor, and etas
        // are swapped because radiance flows in the opposite direction
        if (!is_from_light(bsdf)) {
            return make_float3(refract_coeff * eta_inc_over_eta_trans * eta_inc_over_eta_trans / fabsf(cos_t));
        } else {
            return make_float3(refract_coeff / fabsf(cos_t));
        }
    }
    // else total internal reflection, do nothing

    pdf_w += 0.f;
    return make_float3(0.f);
}

/// Given 3 random numbers, samples new direction from BSDF.
///
/// Uses z component of random triplet to pick BSDF component from
/// which it will sample direction. If non-specular component is chosen,
/// it will also evaluate the other (non-specular) BSDF components.
/// Return BSDF factor for given direction, as well as PDF choosing that direction.
/// Can return event which has been sampled.
/// If result is Dir(0,0,0), then the sample should be discarded.
__forceinline__ __device__ float3 sample(const BSDF& bsdf,
                                         const float3& rnd,
                                         /// Points away from the scattering location
                                         float3& world_dir_gen,
                                         float& pdf_w,
                                         float& cos_theta_gen,
                                         const Scene& scene,
                                         unsigned int* sampled_event = NULL,
                                         float* sin_theta = NULL)
{
    if (is_on_surface(bsdf)) {
        // surface
        unsigned int sampled_event_;

        if (rnd.z < srf(bsdf)->diff_prob) {
            sampled_event_ = BSDF::DIFFUSE;
        } else if (rnd.z < srf(bsdf)->diff_prob + srf(bsdf)->phong_prob) {
            sampled_event_ = BSDF::PHONG;
        } else if (rnd.z < srf(bsdf)->diff_prob + srf(bsdf)->phong_prob + srf(bsdf)->refl_prob) {
            sampled_event_ = BSDF::REFLECT;
        } else {
            sampled_event_ = BSDF::REFRACT;
        }

        if (sampled_event) *sampled_event = sampled_event_;
        if (sin_theta) *sin_theta = 0.f;

        pdf_w = 0.f;
        float3 result = make_float3(0.f);
        float3 local_dir_gen;

        const Material& mat = *srf(bsdf)->mat_ptr;

        if (sampled_event_ == BSDF::DIFFUSE) {
            result += sample_diffuse(bsdf, mat, make_float2(rnd.x, rnd.y), local_dir_gen, pdf_w);

            if (is_black_or_negative(result)) return make_float3(0.f);

            result += evaluate_phong(bsdf, mat, local_dir_gen, &pdf_w);
        } else if (sampled_event_ == BSDF::PHONG) {
            result += sample_phong(bsdf, mat, make_float2(rnd.x, rnd.y), local_dir_gen, pdf_w);

            if (is_black_or_negative(result)) return make_float3(0.f);

            result += evaluate_diffuse(bsdf, mat, local_dir_gen, &pdf_w);
        } else if (sampled_event_ == BSDF::REFLECT) {
            result += sample_reflect(bsdf, mat, make_float2(rnd.x, rnd.y), local_dir_gen, pdf_w);

            if (is_black_or_negative(result)) return make_float3(0.f);
        } else {
            result += sample_refract(bsdf, mat, make_float2(rnd.x, rnd.y), local_dir_gen, pdf_w);

            if (is_black_or_negative(result)) return make_float3(0.f);
        }

        // derive local dir from world dir to mimic process later in
        // intentional code clone of pdf function. this prevents numeric
        // instability
        world_dir_gen = bsdf.frame.to_world(local_dir_gen);

        local_dir_gen = bsdf.frame.to_local(world_dir_gen);

        const float cos_theta_gen_ = local_dir_gen.z;
        const float cos_theta_gen_geom = dot(local_dir_gen, srf(bsdf)->geometry_normal);

        cos_theta_gen = fabsf(cos_theta_gen_);

        // Reject samples that are too parallel with tangent plane
        if (cos_theta_gen < EPS_COSINE || fabsf(cos_theta_gen_geom) < EPS_COSINE) {
            return make_float3(0.f);
        }

        // Refraction must cross the surface, other interactions must not
        if ((sampled_event_ == BSDF::REFRACT && cos_theta_gen_ * cos_theta_fix(bsdf) > 0.f) ||
            (sampled_event_ != BSDF::REFRACT && cos_theta_gen_ * cos_theta_fix(bsdf) < 0.f) ||
            (sampled_event_ == BSDF::REFRACT && cos_theta_gen_geom * srf(bsdf)->cos_theta_fix_geom > 0.f) ||
            (sampled_event_ != BSDF::REFRACT && cos_theta_gen_geom * srf(bsdf)->cos_theta_fix_geom < 0.f)) {
            return make_float3(0.f);
        }

        return result;
    } else {
        // medium
        cos_theta_gen = 1.f;
        if (sampled_event) *sampled_event = BSDF::SCATTER;

        // No need to evaluate phase function if the scattering coef is zero
        if (!med(bsdf)->scatter_coef_is_pos) {
            world_dir_gen = make_float3(0.f);
            pdf_w = 0.f;
            if (sin_theta) *sin_theta = 0.f;
            return make_float3(0.f);
        } else
            return med(bsdf)->scatter_coef * PhaseFunction::sample(med(bsdf)->world_dir_fix,
                                                                   med(bsdf)->mean_cosine,
                                                                   rnd,
                                                                   bsdf.frame,
                                                                   world_dir_gen,
                                                                   pdf_w,
                                                                   sin_theta);
    }
}

#endif // KERNEL_BSDF_CUH
