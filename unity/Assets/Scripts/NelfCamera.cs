using UnityEngine;
using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class NelfCamera
{
    public int H { get; private set; }
    public int W { get; private set; }
    private Matrix4x4 K;

    public float x, y, z;
    public float theta, phi;
    public float fov;

    private Matrix4x4 c2w;

    private float uv_depth = 2.0f;
    private float st_depth = -2.0f;

    private Vector3[] plane_normal;
    private Vector3[] plane_uv;
    private Vector3[] plane_st;

    public NelfCamera(float x = 0, float y = 0, float z = 0, float theta = 0, float phi = 0, float fov = 90, int H = 256, int W = 512)
    {
        this.H = H;
        this.W = W;
        this.K = CalculateIntrinsic();

        this.x = x;
        this.y = y;
        this.z = z;
        this.theta = theta;
        this.phi = phi;

        this.fov = fov;

        this.c2w = SetC2W();

        InitializePlanes();
    }

    private void InitializePlanes()
    {
        plane_normal = new Vector3[H * W];
        plane_uv = new Vector3[H * W];
        plane_st = new Vector3[H * W];

        for (int i = 0; i < H * W; i++)
        {
            plane_normal[i] = new Vector3(0f, 0f, 1f);
            plane_uv[i] = new Vector3(0f, 0f, uv_depth);
            plane_st[i] = new Vector3(0f, 0f, st_depth);
        }
    }

    // private Matrix4x4 CalculateIntrinsic()
    // {
    //     float cx = W / 2f;
    //     float cy = H / 2f;

    //     float focal_length = Mathf.Sqrt(W * W + H * H);

    //     Matrix4x4 intrinsic = Matrix4x4.identity;
    //     intrinsic[0, 0] = focal_length;
    //     intrinsic[1, 1] = focal_length;
    //     intrinsic[0, 2] = cx;
    //     intrinsic[1, 2] = cy;

    //     return intrinsic;
    // }

    // Python K에 맞춤
    private Matrix4x4 CalculateIntrinsic()
    {
        Matrix4x4 intrinsic = Matrix4x4.identity;

        intrinsic[0, 0] = 603.77478027f; // fx
        intrinsic[1, 1] = 603.77478027f; // fy
        intrinsic[0, 2] = 256.0f;        // cx
        intrinsic[1, 2] = 160.0f;        // cy

        return intrinsic;
    }

    public (Vector3[], Vector3[]) GetRaysOD()
    {
        Vector3[] rays_o = new Vector3[H * W];
        Vector3[] rays_d = new Vector3[H * W];

        for (int j = 0; j < H; j++)
        {
            for (int i = 0; i < W; i++)
            {
                Vector3 dir = new Vector3(
                    (i - K[0, 2]) / K[0, 0],
                    -(j - K[1, 2]) / K[1, 1],
                    -1f
                );

                rays_d[j * W + i] = c2w.MultiplyVector(dir).normalized;
                rays_o[j * W + i] = new Vector3(c2w[0, 3], c2w[1, 3], c2w[2, 3]);
            }
        }

        return (rays_o, rays_d);
    }

    // 예전 모델
    public float[,] GetUVST_x()
    {
        // var (rays_o, rays_d) = GetRaysOD();

        // Vector3[] inter_uv = GetRaysInterWithPlane(plane_uv, rays_o, rays_d);
        // Vector3[] inter_st = GetRaysInterWithPlane(plane_st, rays_o, rays_d);

        // float[,] uvst = new float[H * W, 4];

        // for (int i = 0; i < H * W; i++)
        // {
        //     uvst[i, 0] = inter_uv[i].x;
        //     uvst[i, 1] = inter_uv[i].y;
        //     uvst[i, 2] = inter_st[i].x;
        //     uvst[i, 3] = inter_st[i].y;
        // }

        float fovRad = fov * Mathf.Deg2Rad;
        float halfW_at_uv = Mathf.Tan(fovRad * 0.5f) * uv_depth;
        float halfH_at_uv = halfW_at_uv * (H / (float)W);

        float[,] uvst = new float[H * W, 4];

        Vector3 ray_o = new Vector3(c2w[0, 3], c2w[1, 3], c2w[2, 3]); // 카메라 원점

        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                // 카메라 좌표에서 ray 방향
                Vector3 dir = new Vector3(
                    (i - K[0, 2]) / K[0, 0],
                    -(j - K[1, 2]) / K[1, 1],
                    -1f
                );

                Vector3 ray_d = c2w.MultiplyVector(dir).normalized; // 월드로 회전

                // uv 평면 교점
                float t_uv = (uv_depth - ray_o.z) / ray_d.z;
                Vector3 inter_uv = ray_o + t_uv * ray_d;

                // [-1, 1] 정규화
                float u = Mathf.Clamp(inter_uv.x / halfW_at_uv, -1f, 1f);
                float v = Mathf.Clamp(inter_uv.y / halfH_at_uv, -1f, 1f);

                // s, t는 카메라 위치를 scale
                float stScale = 0.25f;
                float s = x * stScale;
                float t = y * stScale;

                int idx = j * W + i;
                uvst[idx, 0] = u;
                uvst[idx, 1] = v;
                uvst[idx, 2] = s;
                uvst[idx, 3] = t;
            }
        }

        return uvst;
    }

    // explicit
    public float[,] GetUVST()
    {
        float[,] uvst = new float[H * W, 4];

        for (int j = 0; j < H; j++)
        {
            for (int i = 0; i < W; i++)
            {
                // 카메라 좌표에서 ray 방향
                Vector3 dir = new Vector3(
                    (i - K[0, 2]) / K[0, 0],
                    -(j - K[1, 2]) / K[1, 1],
                    -1f
                );

                Vector3 ray_d = c2w.MultiplyVector(dir).normalized; // 월드로 회전
                Vector3 ray_o = new Vector3(c2w[0, 3], c2w[1, 3], c2w[2, 3]); // 카메라 원점

                // uv 평면 교점
                float t_uv = (uv_depth - ray_o.z) / ray_d.z;
                Vector3 inter_uv = ray_o + t_uv * ray_d;

                // st 평면 교점
                float t_st = (st_depth - ray_o.z) / ray_d.z;
                Vector3 inter_st = ray_o + t_st * ray_d;

                int idx = j * W + i;
                uvst[idx, 0] = inter_uv.x;
                uvst[idx, 1] = inter_uv.y;
                uvst[idx, 2] = inter_st.x;
                uvst[idx, 3] = inter_st.y;
            }
        }

        return uvst;
    }

    public float[,] GetUVST_Burst(int W, int H, Matrix4x4 c2w, Matrix4x4 K, float uv_depth, float st_depth)
    {
        var uvstNative = new NativeArray<float4>(W * H, Allocator.TempJob);

        var job = new UVSTJob // Job 생성
        {
            width = W,
            height = H,
            uv_depth = uv_depth,
            st_depth = st_depth,

            fx = K[0, 0],
            fy = K[1, 1],
            cx = K[0, 2],
            cy = K[1, 2],
            c2w = ConvertToFloat4x4(c2w), // Matrix4x4 → float4x4

            uvstOut = uvstNative
        };

        JobHandle handle = job.Schedule(W * H, 64); // 픽셀 병렬 처리
        handle.Complete(); // 메인 스레드에서 job이 완료될 때까지 대기

        // 2D 배열로 복사
        float[,] uvst = new float[W * H, 4];
        for (int i = 0; i < uvstNative.Length; i++)
        {
            uvst[i, 0] = uvstNative[i].x;
            uvst[i, 1] = uvstNative[i].y;
            uvst[i, 2] = uvstNative[i].z;
            uvst[i, 3] = uvstNative[i].w;
        }

        uvstNative.Dispose(); // 메모리 해제
        return uvst;
    }

    // Python 
    public float[,] GetUVST_ExplicitPythonStyle(int W, int H, Matrix4x4 c2w)
    {
        float[,] uvst = new float[W * H, 4];

        float stScale = 0.25f;
        float sConst = c2w[0, 3] * stScale;
        float tConst = c2w[1, 3] * stScale;

        for (int y = 0; y < H; y++)
        {
            float v = 1f - 2f * y / (H - 1);   // torch.linspace(1, -1, H)

            for (int x = 0; x < W; x++)
            {
                float u = -1f + 2f * x / (W - 1); // torch.linspace(-1, 1, W)

                int idx = y * W + x;

                // Unity 내부는 [u,v,s,t]로 맞춤
                uvst[idx, 0] = u;
                uvst[idx, 1] = v;
                uvst[idx, 2] = sConst;
                uvst[idx, 3] = tConst;
            }
        }

        return uvst;
    }

    public static float4x4 ConvertToFloat4x4(Matrix4x4 m)
    {
        return new float4x4(
            new float4(m.m00, m.m10, m.m20, m.m30),
            new float4(m.m01, m.m11, m.m21, m.m31),
            new float4(m.m02, m.m12, m.m22, m.m32),
            new float4(m.m03, m.m13, m.m23, m.m33)
        );
    }

    public float[,] getInputTwoPlane()
    {
        Vector3 xy = new Vector3(c2w[0, 3], c2w[1, 3], 0);

        float[,] uvst = new float[H * W, 4];

        for (int j = 0; j < H; j++)
        {
            for (int i = 0; i < W; i++)
            {
                // -1에서 1 사이의 정규화된 좌표로 변환
                float normalizedI = (float)i / (W - 1) * 2f - 1f;
                float normalizedJ = (float)j / (H - 1) * 2f - 1f;

                int idx = j * W + i;

                // ray 생성 (xy, i, j)
                uvst[idx, 0] = xy.x;
                uvst[idx, 1] = xy.y;
                uvst[idx, 2] = normalizedI;
                uvst[idx, 3] = normalizedJ;
            }
        }

        return uvst;
    }

    public (float[,], float[,]) getInputTwoPlaneWithViewdirs()
    {
        Vector3 xy = new Vector3(c2w[0, 3], c2w[1, 3], 0);

        float[,] ray = new float[H * W, 4];
        float[,] viewdirs = new float[H * W, 3];

        for (int j = 0; j < H; j++)
        {
            for (int i = 0; i < W; i++)
            {
                // -1에서 1 사이의 정규화된 좌표로 변환
                float normalizedI = (float)i / (W - 1) * 2f - 1f;
                float normalizedJ = (float)j / (H - 1) * 2f - 1f;

                int idx = j * W + i;

                // ray 생성 (xy, i, j)
                ray[idx, 0] = xy.x;
                ray[idx, 1] = xy.y;
                ray[idx, 2] = normalizedI;
                ray[idx, 3] = normalizedJ;

                // viewdirs 계산
                Vector3 dir = new Vector3(
                    (i - K[0, 2]) / K[0, 0],
                    -(j - K[1, 2]) / K[1, 1],
                    -1f
                );

                Vector3 ray_d = c2w.MultiplyVector(dir);
                ray_d.Normalize();

                viewdirs[idx, 0] = ray_d.x;
                viewdirs[idx, 1] = ray_d.y;
                viewdirs[idx, 2] = ray_d.z;
            }
        }

        return (ray, viewdirs);
    }

    private Vector3[] GetRaysInterWithPlane(Vector3[] p0, Vector3[] rays_o, Vector3[] rays_d)
    {
        Vector3[] inter_point = new Vector3[H * W];

        for (int i = 0; i < H * W; i++)
        {
            float s1 = Vector3.Dot(p0[i], plane_normal[i]);
            float s2 = Vector3.Dot(rays_o[i], plane_normal[i]);
            float s3 = Vector3.Dot(rays_d[i], plane_normal[i]);

            float dist = (s1 - s2) / s3;
            inter_point[i] = rays_o[i] + dist * rays_d[i];
        }

        return inter_point;
    }

    private Matrix4x4 SetC2W()
    {
        float theta_rad = theta * Mathf.Deg2Rad;
        float phi_rad = phi * Mathf.Deg2Rad;
        float psi = Mathf.PI;

        Matrix4x4 R_x = Matrix4x4.identity;
        R_x[1, 1] = Mathf.Cos(phi_rad);
        R_x[1, 2] = -Mathf.Sin(phi_rad);
        R_x[2, 1] = Mathf.Sin(phi_rad);
        R_x[2, 2] = Mathf.Cos(phi_rad);

        Matrix4x4 R_y = Matrix4x4.identity;
        R_y[0, 0] = Mathf.Cos(theta_rad);
        R_y[0, 2] = Mathf.Sin(theta_rad);
        R_y[2, 0] = -Mathf.Sin(theta_rad);
        R_y[2, 2] = Mathf.Cos(theta_rad);

        Matrix4x4 R_z = Matrix4x4.identity;
        R_z[0, 0] = Mathf.Cos(psi);
        R_z[0, 1] = -Mathf.Sin(psi);
        R_z[1, 0] = Mathf.Sin(psi);
        R_z[1, 1] = Mathf.Cos(psi);

        Matrix4x4 R = R_z * R_y * R_x;

        Matrix4x4 c2w = Matrix4x4.identity;
        c2w[0, 3] = x;
        c2w[1, 3] = y;
        c2w[2, 3] = z;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                c2w[i, j] = R[i, j];
            }
        }

        return c2w;
    }

    public void Move(float dx, float dy, float dz)
    {
        x += dx;
        y += dy;
        z += dz;
        UpdateCamera();
    }

    public void Rotate(float dtheta, float dphi)
    {
        theta += dtheta;
        phi += dphi;
        UpdateCamera();
    }

    private void UpdateCamera()
    {
        c2w = SetC2W();
    }

    public void SetYawFromQuaternion(Quaternion rotation)
    {
        float yaw = rotation.eulerAngles.y;
        this.theta = yaw;
        UpdateCamera();
    }

    public void SetC2WFromUnity(Transform t)
    {
        c2w = Matrix4x4.TRS(t.position, t.rotation, Vector3.one); // 오브젝트 → 월드 변환
    }

    // internal float[,] GetUVST()
    // {
    //     throw new NotImplementedException();
    // }

    public Matrix4x4 GetC2WMatrix()
    {
        return this.c2w;
    }

    public Matrix4x4 GetKMatrix()
    {
        return this.K;
    }

}