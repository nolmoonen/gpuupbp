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

#include "../host/env_map.hpp"
#include "../host/light.hpp"
#include "../host/medium.hpp"
#include "../shared/vec_math.h"
#include "scene_loader.hpp"

#include <optix_types.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

/// Line size for reading.
#define BUFFER_SIZE 4096

/// Material info.
struct ObjMaterial {
    /// Name of material.
    std::string name;
    /// True for non-zero emission.
    bool is_emissive;

    /** From MTL file. */

    /// "Kd": diffuse component.
    float3 diffuse;
    /// "Ka": ambient component (not used).
    float3 ambient;
    /// "Ks": specular component.
    float3 specular;
    /// "Ke": emissive component.
    float3 emmissive;
    /// "Ns": specular exponent (0 - 1000).
    float shininess;

    /** From auxiliary file. */

    /// Medium index (-1 for no medium).
    int medium_id;
    /// Index of material enclosing emissive material of area light
    /// (-1 for light in global medium)
    int enclosing_mat_id;
    /// Whether triangles with this material are real.
    bool is_real;
    /// Index of refraction (-1 for non-transmissive).
    float ior;
    /// Mirror component.
    float3 mirror;
    /// Material priority (default is -1).
    int priority;
};

/// Medium info.
struct ObjMedium {
    /// Name of medium.
    std::string name;
    /// Absorption coefficient.
    float3 absorption_coef;
    /// Emission coefficient.
    float3 emission_coef;
    /// Scattering coefficient.
    float3 scattering_coef;
    /// Continuation probability, default is -1 -> must be changed in
    /// scene.hxx for albedo.
    float continuation_probability;
    /// Henyey-Greenstein mean cosine g.
    float mean_cosine;
};

/// Triangle definition in OBJ format.
struct ObjTriangle {
    /// Vertex indices of the three triangle points.
    unsigned int vert_indices[3];
    /// normal indices. UINT32_MAX if no indices specified.
    unsigned int norm_indices[3];
    /// Index in list of materials.
    unsigned int material_idx;
};

/// Group of triangles.
/// Grouping happens by their material as defined by "usemtl".
struct TriangleGroup {
    /// Group name.
    std::string name;
    /// Number of triangles in this group.
    uint32_t triangle_count;
    /// Index to material for this group.
    unsigned int material_idx;
};

/// Camera info.
struct AuxCamera {
    /// Camera origin.
    float origin[3];
    /// Camera target.
    float target[3];
    /// Camera roll direction.
    float roll[3];
    /// Horizontal field of view (in radians).
    float horizontal_fov;
    /// Focal distance.
    float focal_distance;
    /// Camera resolution X.
    int resolution_x;
    /// Camera resolution Y.
    int resolution_y;
    /// Index to material enclosing medium the camera is located in, -1 for
    /// camera in global medium.
    int material_idx;
};

/// Additional light info
struct AuxLight {
    enum AuxLightType { AUX_DIRECTIONAL = 0, AUX_POINT = 1, AUX_BACKGROUND = 2 };

    /// Stores either position for a point light or
    /// the direction for a directional light.
    float3 position;
    /// Light emission.
    float3 emission;
    /// Type of the light.
    AuxLightType light_type;
    /// Environment map filename for background light.
    std::string env_map;
    /// Scaling of environment map.
    float env_map_scale;
    /// Rotation of env map around vertical axis (0-1, 1 = 360 degrees).
    float env_map_rotate;
};

/// Returns directory name from path
static std::string get_dir_name(const std::string& path)
{
    std::string::size_type i = path.find_last_of('/');
    if (i != std::string::npos) {
        return path.substr(0, i + 1);
    } else {
        return "";
    }
}

/// Returns id of a selected material.
unsigned int find_material(const std::vector<ObjMaterial>& materials, const char* name)
{
    unsigned int i;

    for (i = 0; i < materials.size(); i++) {
        if (materials[i].name == name) return i;
    }

    std::cerr << "Error: material not found: " << name << std::endl;
    exit(2);

    return -1;
}

/// Returns id of a selected group, if it does not exist, new is created
TriangleGroup* find_group(std::vector<TriangleGroup>& groups, const char* name)
{
    unsigned int i;

    for (i = 0; i < groups.size(); i++) {
        if (groups[i].name == name) return &groups[i];
    }
    TriangleGroup group;
    group.material_idx = 0;
    group.name = name;
    group.triangle_count = 0;
    groups.push_back(group);

    return &groups.back();
}

/// Returns id of a selected medium, if it does not exist, new is created
ObjMedium* find_medium(std::vector<ObjMedium>& media, const char* name)
{
    unsigned int i;

    for (i = 0; i < media.size(); i++) {
        if (media[i].name == name) return &media[i];
    }
    ObjMedium medium;
    medium.name = name;
    medium.absorption_coef = make_float3(0.f);
    medium.emission_coef = make_float3(0.f);
    medium.scattering_coef = make_float3(.5f);
    medium.mean_cosine = 0.0f;
    medium.continuation_probability = -1.0f;
    media.push_back(medium);

    return &media.back();
}

/// Fills camera info from matrix
void fill_camera_info(AuxCamera& camera, float* row0, float* row1, float* row2, float* row3)
{
    // First perform invert  -> just compute third column, plus xyz -> zxy (3dmax rules :( )
    float inverse[3][4] = {
        {row0[0], row1[0], row2[0], row3[0]}, // row0[0] * row3[1] + row1[0] * row3[2] + row2[0] * row3[0] },
        {row0[1], row1[1], row2[1], row3[1]}, // row0[1] * row3[1] + row1[1] * row3[2] + row2[2] * row3[0] },
        {row0[2], row1[2], row2[2], row3[2]}  // row0[2] * row3[1] + row1[2] * row3[2] + row2[1] * row3[0]
    };
    /* Now we will transform these points
    a: 0, 0, 0, 1
    b: 0, 0, -1, 1
    c: 0, 1, 0, 1
    Camera origin is then A, camera target is B, and camera roll is (C-A) normalized
    */
    camera.origin[0] = inverse[0][3];
    camera.origin[1] = inverse[1][3];
    camera.origin[2] = inverse[2][3];
    camera.target[0] = inverse[0][3] - inverse[0][2];
    camera.target[1] = inverse[1][3] - inverse[1][2];
    camera.target[2] = inverse[2][3] - inverse[2][2];
    camera.roll[0] = inverse[0][1];
    camera.roll[1] = inverse[1][1];
    camera.roll[2] = inverse[2][1];
    // Normalize roll
    float size = 1.f / sqrtf(camera.roll[0] * camera.roll[0] + camera.roll[1] * camera.roll[1] +
                             camera.roll[2] * camera.roll[2]);
    camera.roll[0] *= size;
    camera.roll[1] *= size;
    camera.roll[2] *= size;
}

void create_default_mat(ObjMaterial& material, const char* name)
{
    material.name = name;
    material.shininess = 0.0f;
    material.diffuse = make_float3(0.f);
    material.ambient = make_float3(0.f);
    material.specular = make_float3(0.f);
    material.is_emissive = false;
    material.priority = -1;
    material.ior = -1.0f;
    material.mirror = make_float3(0.f);
    material.medium_id = -1;
    material.enclosing_mat_id = -1;
    material.is_real = true;
}

/// Reads material file
void read_mtl(std::vector<ObjMaterial>& materials, std::vector<ObjMedium>& media, const std::string& filename)
{
    char buf[BUFFER_SIZE];

    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: could not open ``" << filename << "''" << std::endl;
        exit(2);
    }

    ObjMaterial* material = nullptr;

    while (fscanf(file, "%s", buf) != EOF) {
        switch (buf[0]) {
        case '#': /* comment */
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        case 'n': /* newmtl */
            fgets(buf, sizeof(buf), file);
            sscanf(buf, "%s", buf);
            materials.push_back(ObjMaterial());
            material = &materials.back();
            create_default_mat(*material, buf);
            break;
        case 'N':
            assert(material);
            if (buf[1] != 's') break;
            fscanf(file, "%f", &material->shininess); /* 0 - 1000 */
            break;
        case 'K':
            assert(material);
            switch (buf[1]) {
            case 'd':
                fscanf(file, "%f %f %f", &material->diffuse.x, &material->diffuse.y, &material->diffuse.z);
                break;
            case 's':
                fscanf(file, "%f %f %f", &material->specular.x, &material->specular.y, &material->specular.z);
                break;
            case 'a':
                fscanf(file, "%f %f %f", &material->ambient.x, &material->ambient.y, &material->ambient.z);
                break;
            case 'e':
                fscanf(file, "%f %f %f", &material->emmissive.x, &material->emmissive.y, &material->emmissive.z);
                material->is_emissive =
                    material->emmissive.x > 0.f || material->emmissive.y > 0.f || material->emmissive.z > 0.f;
                break;
            default:
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                break;
            }
            break;
        default:
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        }
    }
    fclose(file);
}

/// Reads aux file.
void read_aux(std::vector<ObjMaterial>& materials,
              std::vector<ObjMedium>& media,
              std::vector<AuxLight>& lights,
              AuxCamera& camera,
              int& global_medium_id,
              const std::string& filename)
{
    char buf[BUFFER_SIZE];

    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: could not open ``" << filename << "''" << std::endl;
        exit(2);
    }

    ObjMaterial* material = NULL;
    ObjMedium* medium = NULL;
    // camera matrix
    float tm[4][3];
    camera.resolution_x = camera.resolution_y = 256; // Default
    camera.material_idx = -1;
    global_medium_id = -1;
    AuxLight al;

    while (fscanf(file, "%s", buf) != EOF) {
        switch (buf[0]) {
        case '#': // comment
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        case 'T': // transform matrix camera
            // one of TM_ROWX, X=0,1,2,3
            fscanf(file, "%f %f %f", &tm[buf[6] - '0'][0], &tm[buf[6] - '0'][1], &tm[buf[6] - '0'][2]);
            break;
        case 'C': // camera options
            switch (buf[7]) {
            case 'F': // CAMERA_FOV
                fscanf(file, "%f", &camera.horizontal_fov);
                break;
            case 'T': // CAMERA_TDIST
                fscanf(file, "%f", &camera.focal_distance);
                break;
            case 'M': // CAMERA_MATERIAL
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%s", buf);
                camera.material_idx = find_material(materials, buf);
                break;
            case 'R': // camera resolution
                switch (buf[18]) {
                case 'X': // CAMERA_RESOLUTION_X
                    fscanf(file, "%i", &camera.resolution_x);
                    if (camera.resolution_x < 1) {
                        camera.resolution_x = 256;
                    }
                    break;
                case 'Y': // CAMERA_RESOLUTION_Y
                    fscanf(file, "%i", &camera.resolution_y);
                    if (camera.resolution_y < 1) {
                        camera.resolution_y = 256;
                    }
                    break;
                default:
                    std::cerr << "Error: unknown camera option: " << buf << std::endl;
                    exit(2);
                }
                break;
            default:
                std::cerr << "Error: unknown camera option: " << buf << std::endl;
                exit(2);
            }
            break;
        case 'm': // material, medium, mediumId, or mirror
            switch (buf[1]) {
            case 'a': // material
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%s", buf);
                material = &materials[0] + find_material(materials, buf);
                break;
            case 'e':
                switch (buf[6]) {
                case 'I': // mediumId
                    if (material == NULL) {
                        std::cerr << "Error: using mediumId option without material selection" << std::endl;
                        exit(2);
                    }
                    fgets(buf, sizeof(buf), file);
                    sscanf(buf, "%s", buf);
                    material->medium_id = (int)(find_medium(media, buf) - &media[0]);
                    break;
                case '\0': // medium
                    fgets(buf, sizeof(buf), file);
                    sscanf(buf, "%s", buf);
                    medium = find_medium(media, buf);
                    break;
                default:
                    /* eat up rest of line */
                    fgets(buf, sizeof(buf), file);
                    break;
                }
                break;
            case 'i': // mirror
                if (material == NULL) {
                    std::cerr << "Error: using mirror option without material selection" << std::endl;
                    exit(2);
                }
                fscanf(file, "%f %f %f", &material->mirror.x, &material->mirror.y, &material->mirror.z);
                break;
            default:
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                break;
            }

            break;
        case 'i': // ior
            if (material == NULL) {
                std::cerr << "Error: using IOR option without material selection" << std::endl;
                exit(2);
            }
            fscanf(file, "%f", &material->ior);
            break;
        case 'p': // priority
            if (material == NULL) {
                std::cerr << "Error: using priority option without material selection" << std::endl;
                exit(2);
            }
            fscanf(file, "%i", &material->priority);
            break;
        case 'g': // g, geometryType, or globalMediumId
            switch (buf[1]) {
            case '\0': // g
                if (medium == NULL) {
                    std::cerr << "Error: using g option without medium selection" << std::endl;
                    exit(2);
                }
                fscanf(file, "%f", &medium->mean_cosine);
                break;
            case 'e': // geometryType
                if (material == NULL) {
                    std::cerr << "Error: using geometryType option without material selection" << std::endl;
                    exit(2);
                }
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%s", buf);
                if (buf[0] == 'r') {
                    material->is_real = true;
                } else if (buf[0] == 'i') {
                    material->is_real = false;
                } else {
                    std::cerr << "Error: unknown geometry type" << std::endl;
                    exit(2);
                }
                break;
            case 'l': // globalMediumId
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%s", buf);
                global_medium_id = (int)(find_medium(media, buf) - &media[0]);
                break;
            default:
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                break;
            }
            break;
        case 'K': // Ke
            if (material == NULL) {
                std::cerr << "Error: using Ke option without material selection" << std::endl;
                exit(2);
            }
            fscanf(file, "%f %f %f", &material->emmissive.x, &material->emmissive.y, &material->emmissive.z);
            material->is_emissive =
                material->emmissive.x > 0.f || material->emmissive.y > 0.f || material->emmissive.z > 0.f;
            break;
        case 'a': // absorption
            if (medium == NULL) {
                std::cerr << "Error: using absorption option without medium selection" << std::endl;
                exit(2);
            }
            fscanf(
                file, "%f %f %f", &medium->absorption_coef.x, &medium->absorption_coef.y, &medium->absorption_coef.z);
            break;
        case 'c': // continuation_probability
            if (medium == NULL) {
                std::cerr << "Error: using continuation_probability option without medium selection" << std::endl;
                exit(2);
            }
            fscanf(file, "%f", &medium->continuation_probability);
            break;
        case 'e': // emission or enclosingMatId
            switch (buf[1]) {
            case 'm': // emission
                if (medium == NULL) {
                    std::cerr << "Error: using emission option without medium selection" << std::endl;
                    exit(2);
                }
                fscanf(file, "%f %f %f", &medium->emission_coef.x, &medium->emission_coef.y, &medium->emission_coef.z);
                break;
            case 'n': // enclosingMatId
                if (material == NULL) {
                    std::cerr << "Error: using enclosingMatId option without material selection" << std::endl;
                    exit(2);
                }
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%s", buf);
                material->enclosing_mat_id = find_material(materials, buf);
                break;
            }
            break;
        case 's': // scattering
            if (medium == NULL) {
                std::cerr << "Error: using scattering option without medium selection" << std::endl;
                exit(2);
            }
            fscanf(
                file, "%f %f %f", &medium->scattering_coef.x, &medium->scattering_coef.y, &medium->scattering_coef.z);
            break;
        case 'l': // light
            switch (buf[6]) {
            case 'p': // light_point
                al.light_type = AuxLight::AUX_POINT;
                fscanf(file,
                       "%f %f %f %f %f %f",
                       &al.position.x,
                       &al.position.y,
                       &al.position.z,
                       &al.emission.x,
                       &al.emission.y,
                       &al.emission.z);
                lights.push_back(al);
                break;
            case 'd': // light_directional
                al.light_type = AuxLight::AUX_DIRECTIONAL;
                fscanf(file,
                       "%f %f %f %f %f %f",
                       &al.position.x,
                       &al.position.y,
                       &al.position.z,
                       &al.emission.x,
                       &al.emission.y,
                       &al.emission.z);
                lights.push_back(al);
                break;
            case 'b': // light_background_...
                al.light_type = AuxLight::AUX_BACKGROUND;
                switch (buf[17]) {
                case 'c': // light_background_constant
                    fscanf(file, "%f %f %f", &al.emission.x, &al.emission.y, &al.emission.z);
                    break;
                case 'e': // light_directional_em
                    fscanf(file, "%f %f", &al.env_map_scale, &al.env_map_rotate);
                    fgets(buf, sizeof(buf), file);
                    sscanf(buf, "%s", buf);
                    al.env_map = std::string(buf);
                    if (al.env_map[1] != ':') {
                        // get absolute
                        al.env_map = get_dir_name(filename) + al.env_map;
                    }
                    break;
                }
                lights.push_back(al);
                break;
            default:
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                break;
            }
            break;
        default:
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        }
    }
    fclose(file);
    fill_camera_info(camera, tm[0], tm[1], tm[2], tm[3]);
}

/// Takes in a triangle list where the triangles are sorted by their
/// group.
void process_triangles(const std::vector<ObjTriangle>& t_list,
                       Triangles& tr_real,
                       Triangles& tr_imag,
                       const std::vector<float>& vertices,
                       const std::vector<float>& normals,
                       const std::vector<ObjMaterial>& materials,
                       SceneLoader& scene_loader,
                       /// Index of the next light to add.
                       uint32_t& tr_light_idx,
                       float3& bbox_min,
                       float3& bbox_max)
{
    // index of the next record to be created
    uint32_t rcrd_idx = 0;
    // cache the index of the previously added record
    // as consecutive triangles usually need the same record
    uint32_t prev_record_idx = UINT32_MAX;
    uint32_t idx_real = 0;
    uint32_t idx_imag = 0;
    for (uint32_t k = 0; k < t_list.size(); k++) {
        int32_t mat_idx = t_list[k].material_idx;
        const ObjMaterial& mat = materials[mat_idx];
        Triangles& tr = mat.is_real ? tr_real : tr_imag;
        uint32_t idx = mat.is_real ? idx_real++ : idx_imag++;
        assert(idx < tr.count);

        // read in vertices
        assert(3 * t_list[k].vert_indices[0] + 2 < vertices.size());
        const float3 p0 = make_float3(vertices[3 * t_list[k].vert_indices[0] + 0],
                                      vertices[3 * t_list[k].vert_indices[0] + 1],
                                      vertices[3 * t_list[k].vert_indices[0] + 2]);
        tr.verts[3 * idx + 0] = make_float4(p0, 0.f);
        assert(3 * t_list[k].vert_indices[1] + 2 < vertices.size());
        const float3 p1 = make_float3(vertices[3 * t_list[k].vert_indices[1] + 0],
                                      vertices[3 * t_list[k].vert_indices[1] + 1],
                                      vertices[3 * t_list[k].vert_indices[1] + 2]);
        tr.verts[3 * idx + 1] = make_float4(p1, 0.f);
        assert(3 * t_list[k].vert_indices[2] + 2 < vertices.size());
        const float3 p2 = make_float3(vertices[3 * t_list[k].vert_indices[2] + 0],
                                      vertices[3 * t_list[k].vert_indices[2] + 1],
                                      vertices[3 * t_list[k].vert_indices[2] + 2]);
        tr.verts[3 * idx + 2] = make_float4(p2, 0.f);
        // face normal
        float3 face_normal = normalize(cross(p1 - p0, p2 - p0));
        // read in normals (if they exist)
        float3 n0, n1, n2;
        if (t_list[k].norm_indices[0] != UINT32_MAX && t_list[k].norm_indices[1] != UINT32_MAX &&
            t_list[k].norm_indices[2] != UINT32_MAX) {
            assert(3 * t_list[k].norm_indices[0] + 2 < normals.size());
            n0 = make_float3(normals[3 * t_list[k].norm_indices[0] + 0],
                             normals[3 * t_list[k].norm_indices[0] + 1],
                             normals[3 * t_list[k].norm_indices[0] + 2]);
            assert(3 * t_list[k].norm_indices[1] + 2 < normals.size());
            n1 = make_float3(normals[3 * t_list[k].norm_indices[1] + 0],
                             normals[3 * t_list[k].norm_indices[1] + 1],
                             normals[3 * t_list[k].norm_indices[1] + 2]);
            assert(3 * t_list[k].norm_indices[2] + 2 < normals.size());
            n2 = make_float3(normals[3 * t_list[k].norm_indices[2] + 0],
                             normals[3 * t_list[k].norm_indices[2] + 1],
                             normals[3 * t_list[k].norm_indices[2] + 2]);
            // check face normal
            float3 mdl = (n0 + n1 + n2) / 3.f;
            if (dot(face_normal, mdl) < 0.f) {
                face_normal = -face_normal;
            }
        } else {
            // otherwise, use face normal
            n0 = face_normal;
            n1 = face_normal;
            n2 = face_normal;
        }
        tr.norms[4 * idx + 0] = n0;
        tr.norms[4 * idx + 1] = n1;
        tr.norms[4 * idx + 2] = n2;
        tr.norms[4 * idx + 3] = face_normal;
        // find the record for this triangle
        int32_t light_idx = -1;
        // if the material is emissive, also create an area light
        if (mat.is_emissive) {
            assert(tr_light_idx < scene_loader.scene.light_count);
            // modify the light index of the record
            light_idx = static_cast<int32_t>(tr_light_idx);
            // create the light
            int light_mat = mat.enclosing_mat_id < 0 ? -1 : mat.enclosing_mat_id;
            int light_med = light_mat < 0 ? -1 : materials[light_mat].medium_id;
            init_area_light(scene_loader.scene.lights[tr_light_idx], p0, p1, p2, mat.emmissive, light_mat, light_med);
            // created new area light, increment light index
            tr_light_idx++;
        }

        // check if we can use cache
        if (prev_record_idx != UINT32_MAX && scene_loader.records[prev_record_idx].id_material == mat_idx &&
            scene_loader.records[prev_record_idx].id_light == light_idx) {
            // use cache
            tr.records[idx] = prev_record_idx;
        } else {
            // search if record exists
            bool found = false;
            for (uint32_t j = 0; j < rcrd_idx; j++) {
                if (scene_loader.records[j].id_material == mat_idx && scene_loader.records[j].id_light == light_idx) {
                    // if exists, use this and update prev_idx
                    tr.records[idx] = j;
                    prev_record_idx = j;
                    found = true;
                    break;
                }
            }

            // create record if it does not exist, update prev_idx
            if (!found) {
                assert(rcrd_idx < scene_loader.record_count);
                prev_record_idx = rcrd_idx;
                tr.records[idx] = rcrd_idx;
                // init record
                Record* rec = &scene_loader.records[rcrd_idx++];
                rec->id_material = mat_idx;
                rec->id_light = light_idx;
            }
        }
        // update bounding boxes
        SceneLoader::grow_bb(bbox_min, bbox_max, p0);
        SceneLoader::grow_bb(bbox_min, bbox_max, p1);
        SceneLoader::grow_bb(bbox_min, bbox_max, p2);
    }
    // assert all records and lights are used and our assumptions hold
    assert(rcrd_idx == scene_loader.record_count);
    assert(tr_light_idx == scene_loader.scene.light_count);
}

/// Reads obj from a selected file
void read_obj(const std::string& filename,
              std::vector<float>& vertices,
              std::vector<float>& normals,
              std::vector<ObjMaterial>& materials,
              std::vector<ObjMedium>& media,
              std::vector<TriangleGroup>& groups,
              std::vector<ObjTriangle>& triangles)
{
    /* open the file */
    FILE* file = fopen(filename.c_str(), "r");
    if (!file) {
        std::cerr << "Error: could not open ``" << filename << "''" << std::endl;
        exit(2);
    }

    TriangleGroup* group = nullptr;     // current group
    unsigned int material = UINT32_MAX; // current material
    unsigned int v, n, t;
    char buf[BUFFER_SIZE];
    while (fscanf(file, "%s", buf) != EOF) {
        float x, y, z;
        switch (buf[0]) {
        case '#': // comment
            // eat up rest of line
            fgets(buf, sizeof(buf), file);
            break;
        case 'v': // v, vn, vt
            switch (buf[1]) {
            case '\0': // vertex
                fscanf(file, "%f %f %f", &x, &y, &z);
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
                break;
            case 'n': // normal
                fscanf(file, "%f %f %f", &x, &y, &z);
                normals.push_back(x);
                normals.push_back(y);
                normals.push_back(z);
                break;
            case 't': // texcoord
                fscanf(file, "%f %f", &x, &y);
                // intentionally discard
                break;
            }
            break;
        case 'm': // mtllib
        {
            fgets(buf, sizeof(buf), file);
            sscanf(buf, "%s", buf);
            std::string m_mtllibname = buf;
            read_mtl(materials, media, get_dir_name(filename) + m_mtllibname);
            break;
        }
        case 'u': // usemtl
            fgets(buf, sizeof(buf), file);
            sscanf(buf, "%s", buf);
            group = find_group(groups, buf);
            group->material_idx = material = find_material(materials, buf);
            break;
        case 'f': /* face */
        {
            assert(group);
            triangles.push_back(ObjTriangle());
            auto triangle = triangles.end() - 1;
            v = n = t = 0;
            fscanf(file, "%s", buf);
            /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
            if (strstr(buf, "//")) {
                /* v//n */
                sscanf(buf, "%d//%d", &v, &n);
                triangle->vert_indices[0] = v - 1;
                triangle->norm_indices[0] = n - 1;
                fscanf(file, "%d//%d", &v, &n);
                triangle->vert_indices[1] = v - 1;
                triangle->norm_indices[1] = n - 1;
                fscanf(file, "%d//%d", &v, &n);
                triangle->vert_indices[2] = v - 1;
                triangle->norm_indices[2] = n - 1;
                triangle->material_idx = material;
                group->triangle_count++;
                while (fscanf(file, "%d//%d", &v, &n) > 0) {
                    triangles.push_back(ObjTriangle());
                    triangle = triangles.end() - 1;
                    auto prev = triangle - 1;
                    triangle->vert_indices[0] = prev->vert_indices[0];
                    triangle->norm_indices[0] = prev->norm_indices[0];
                    triangle->vert_indices[1] = prev->vert_indices[2];
                    triangle->norm_indices[1] = prev->norm_indices[2];
                    triangle->vert_indices[2] = v - 1;
                    triangle->norm_indices[2] = n - 1;
                    triangle->material_idx = material;
                    group->triangle_count++;
                }
            } else if (sscanf(buf, "%d/%d/%d", &v, &t, &n) == 3) {
                /* v/t/n */
                triangle->vert_indices[0] = v - 1;
                triangle->norm_indices[0] = n - 1;
                fscanf(file, "%d/%d/%d", &v, &t, &n);
                triangle->vert_indices[1] = v - 1;
                triangle->norm_indices[1] = n - 1;
                fscanf(file, "%d/%d/%d", &v, &t, &n);
                triangle->vert_indices[2] = v - 1;
                triangle->norm_indices[2] = n - 1;
                triangle->material_idx = material;
                group->triangle_count++;
                while (fscanf(file, "%d/%d/%d", &v, &t, &n) > 0) {
                    triangles.push_back(ObjTriangle());
                    triangle = triangles.end() - 1;
                    auto prev = triangle - 1;
                    triangle->vert_indices[0] = prev->vert_indices[0];
                    triangle->norm_indices[0] = prev->norm_indices[0];
                    triangle->vert_indices[1] = prev->vert_indices[2];
                    triangle->norm_indices[1] = prev->norm_indices[2];
                    triangle->vert_indices[2] = v - 1;
                    triangle->norm_indices[2] = n - 1;
                    triangle->material_idx = material;
                    group->triangle_count++;
                }
            } else if (sscanf(buf, "%d/%d", &v, &t) == 2) {
                /* v/t */
                triangle->vert_indices[0] = v - 1;
                triangle->norm_indices[0] = UINT32_MAX;
                fscanf(file, "%d/%d", &v, &t);
                triangle->vert_indices[1] = v - 1;
                triangle->norm_indices[1] = UINT32_MAX;
                fscanf(file, "%d/%d", &v, &t);
                triangle->vert_indices[2] = v - 1;
                triangle->norm_indices[2] = UINT32_MAX;
                triangle->material_idx = material;
                group->triangle_count++;
                while (fscanf(file, "%d/%d", &v, &t) > 0) {
                    triangles.push_back(ObjTriangle());
                    triangle = triangles.end() - 1;
                    auto prev = triangle - 1;
                    triangle->vert_indices[0] = prev->vert_indices[0];
                    triangle->norm_indices[0] = UINT32_MAX;
                    triangle->vert_indices[1] = prev->vert_indices[2];
                    triangle->norm_indices[1] = UINT32_MAX;
                    triangle->vert_indices[2] = v - 1;
                    triangle->norm_indices[2] = UINT32_MAX;
                    triangle->material_idx = material;
                    group->triangle_count++;
                }
            } else {
                /* v */
                sscanf(buf, "%d", &v);
                triangle->vert_indices[0] = v - 1;
                triangle->norm_indices[0] = UINT32_MAX;
                fscanf(file, "%d", &v);
                triangle->vert_indices[1] = v - 1;
                triangle->norm_indices[1] = UINT32_MAX;
                fscanf(file, "%d", &v);
                triangle->vert_indices[2] = v - 1;
                triangle->norm_indices[2] = UINT32_MAX;
                triangle->material_idx = material;
                group->triangle_count++;
                while (fscanf(file, "%d", &v) > 0) {
                    triangles.push_back(ObjTriangle());
                    triangle = triangles.end() - 1;
                    auto prev = triangle - 1;
                    triangle->vert_indices[0] = prev->vert_indices[0];
                    triangle->norm_indices[0] = UINT32_MAX;
                    triangle->vert_indices[1] = prev->vert_indices[2];
                    triangle->norm_indices[1] = UINT32_MAX;
                    triangle->vert_indices[2] = v - 1;
                    triangle->norm_indices[2] = UINT32_MAX;
                    triangle->material_idx = material;
                    group->triangle_count++;
                }
            }
        } break;

        default:
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        }
    }
    fclose(file);
}

void SceneLoader::load_from_obj(const char* file, const float2& resolution, float3& bbox_min, float3& bbox_max)
{
    // list of vertex x, y, and z positions
    std::vector<float> vertices;
    // list of vertex x, y, and z normals
    std::vector<float> normals;
    // list of obj triangles
    std::vector<ObjTriangle> obj_triangles;
    // list of obj materials
    std::vector<ObjMaterial> obj_materials;
    // list of obj media
    std::vector<ObjMedium> obj_media;
    // every "usemtl" statement creates a group of triangles that all share
    // the same material. However, multiple groups may have the same
    // material.
    std::vector<TriangleGroup> group;
    // list of auxiliary lights
    std::vector<AuxLight> aux_lights;
    // camera as specified in aux file
    AuxCamera aux_camera;
    // the global medium, remains -1 if none is specified
    int32_t obj_global_medium_id = -1;

    // read in OBJ file
    read_obj(file, vertices, normals, obj_materials, obj_media, group, obj_triangles);

    // read in auxiliary file
    read_aux(obj_materials, obj_media, aux_lights, aux_camera, obj_global_medium_id, std::string(file) + ".aux");

    record_count = 0;
    scene.light_count = static_cast<uint32_t>(aux_lights.size());
    scene.triangles_real.count = 0;
    scene.triangles_imag.count = 0;

    // count the number of records, lights, and triangles
    for (uint32_t i = 0; i < group.size(); i++) {
        if (obj_materials[group[i].material_idx].is_emissive) {
            // one area light for each triangle
            scene.light_count += group[i].triangle_count;
            // all triangles have a unique light id and same material
            record_count += group[i].triangle_count;
        } else {
            // all triangles have the same light id (-1) and material
            record_count++;
        }

        if (obj_materials[group[i].material_idx].is_real) {
            scene.triangles_real.count += group[i].triangle_count;
        } else {
            scene.triangles_imag.count += group[i].triangle_count;
        }
    }

    // allocate memory for triangles
    scene.triangles_real.verts = static_cast<float4*>(malloc(3 * sizeof(float4) * scene.triangles_real.count));
    scene.triangles_real.norms = static_cast<float3*>(malloc(4 * sizeof(float3) * scene.triangles_real.count));
    scene.triangles_real.records = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * scene.triangles_real.count));

    scene.triangles_imag.verts = static_cast<float4*>(malloc(3 * sizeof(float4) * scene.triangles_imag.count));
    scene.triangles_imag.norms = static_cast<float3*>(malloc(4 * sizeof(float3) * scene.triangles_imag.count));
    scene.triangles_imag.records = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * scene.triangles_imag.count));

    // initialize all non-area lights
    scene.lights = static_cast<AbstractLight*>(malloc(sizeof(AbstractLight) * scene.light_count));
    uint32_t light_idx;
    for (light_idx = 0; light_idx < aux_lights.size(); light_idx++) {
        switch (aux_lights[light_idx].light_type) {
        case AuxLight::AUX_POINT: {
            init_point_light(scene.lights[light_idx], aux_lights[light_idx].position, aux_lights[light_idx].emission);
        } break;
        case AuxLight::AUX_BACKGROUND: {
            // only process if not already set
            if (scene.background_light_idx == UINT32_MAX) {
                init_background_light(scene.lights[light_idx], aux_lights[light_idx].emission, 1.f);
                if (aux_lights[light_idx].env_map.length() > 0) {
                    init_env_map(scene.env_map,
                                 aux_lights[light_idx].env_map,
                                 aux_lights[light_idx].env_map_rotate,
                                 aux_lights[light_idx].env_map_scale);
                    // indicate that the scene has an environment map
                    scene.has_env_map = true;
                    // indicate that this background uses the
                    // environment map
                    scene.lights[light_idx].background.uses_env_map = true;
                }

                scene.background_light_idx = light_idx;
            }
        } break;
        case AuxLight::AUX_DIRECTIONAL: {
            init_directional_light(
                scene.lights[light_idx], aux_lights[light_idx].position, aux_lights[light_idx].emission);
        } break;
        }
    }

    // allocate and initialize records
    records = static_cast<Record*>(malloc(sizeof(Record) * record_count));

    // initialize all triangles and all area lights
    process_triangles(obj_triangles,
                      scene.triangles_real,
                      scene.triangles_imag,
                      vertices,
                      normals,
                      obj_materials,
                      *this,
                      light_idx,
                      bbox_min,
                      bbox_max);

    /** init camera */
    int cam_mat = aux_camera.material_idx < 0 ? -1 : aux_camera.material_idx;
    int cam_med = cam_mat < 0 ? -1 : obj_materials[cam_mat].medium_id;
    init_camera(scene.camera,
                make_float3(aux_camera.origin[0], aux_camera.origin[1], aux_camera.origin[2]),
                make_float3(aux_camera.target[0], aux_camera.target[1], aux_camera.target[2]),
                make_float3(aux_camera.roll[0], aux_camera.roll[1], aux_camera.roll[2]),
                resolution,
                // convert from radians to degrees
                aux_camera.horizontal_fov * 57.295779f, // 360 / (2 * PI)
                aux_camera.focal_distance,
                cam_mat,
                cam_med);

    // Handle materials
    scene.mat_count = static_cast<uint32_t>(obj_materials.size());
    scene.materials = static_cast<Material*>(malloc(sizeof(Material) * scene.mat_count));
    for (uint32_t i = 0; i < obj_materials.size(); ++i) {
        assert(obj_materials[i].priority != -1 || obj_materials[i].medium_id == -1);
        scene.materials[i].diffuse_reflectance = obj_materials[i].diffuse;
        scene.materials[i].phong_reflectance = obj_materials[i].specular;
        scene.materials[i].mirror_reflectance = obj_materials[i].mirror;
        scene.materials[i].ior = obj_materials[i].ior;
        scene.materials[i].phong_exponent = obj_materials[i].shininess;
        scene.materials[i].priority = obj_materials[i].priority;
        scene.materials[i].real = obj_materials[i].is_real;
        if (obj_materials[i].shininess == 0.f) {
            scene.materials[i].phong_reflectance = make_float3(0.f);
        }
        scene.materials[i].med_idx = obj_materials[i].medium_id;
    }

    /** Handle medium */
    scene.med_count = static_cast<uint32_t>(obj_media.size());
    if (obj_global_medium_id == -1) {
        // if no global medium, allocate an additional slot for a global medium
        // to be created
        scene.med_count++;
    }
    scene.media = static_cast<Medium*>(malloc(sizeof(Medium) * scene.med_count));
    for (uint32_t i = 0; i < obj_media.size(); i++) {
        float contProb = obj_media[i].continuation_probability;
        if (contProb == -1.f) {
            // Set as albedo
            float3 albedo =
                obj_media[i].scattering_coef / (obj_media[i].scattering_coef + obj_media[i].absorption_coef);
            float maxAlbedo = fmaxf(albedo);
            if (std::isnan(maxAlbedo) || std::isinf(maxAlbedo)) {
                contProb = 1.f;
            } else {
                contProb = maxAlbedo > MEDIUM_SURVIVAL_PROB ? maxAlbedo : MEDIUM_SURVIVAL_PROB;
            }
        }
        init_homogeneous_medium(scene.media[i],
                                obj_media[i].absorption_coef,
                                obj_media[i].emission_coef,
                                obj_media[i].scattering_coef,
                                contProb,
                                obj_media[i].mean_cosine);
    }
    if (obj_global_medium_id == -1) {
        // if no global medium, create the default one (clear)
        init_medium_clear(scene.media[scene.med_count - 1]);
        scene.global_medium_idx = static_cast<int32_t>(scene.med_count) - 1;
    } else {
        scene.global_medium_idx = obj_global_medium_id;
    }
}
