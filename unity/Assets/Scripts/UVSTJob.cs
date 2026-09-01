using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

[BurstCompile]
public struct UVSTJob : IJobParallelFor
{
    [ReadOnly] public int width;
    [ReadOnly] public int height;

    [ReadOnly] public float uv_depth;
    [ReadOnly] public float st_depth;

    [ReadOnly] public float fx, fy, cx, cy;
    [ReadOnly] public float4x4 c2w; // Camera-to-world

    [WriteOnly] public NativeArray<float4> uvstOut; // [x,y,s,t]

    public void Execute(int index)
    {
        // index → 2D 이미지 좌표
        int j = index / width;
        int i = index % width;

        // 이미지 좌표를 정규화된 카메라 좌표계로 변환
        float x = (i - cx) / fx;
        float y = -(j - cy) / fy;
        float z = -1f;

        // 카메라 좌표계에서의 ray 방향
        float3 dir_cam = math.normalize(new float3(x, y, z));

        // c2w 좌표계로 회전
        float3 ray_d = math.normalize(new float3(
            c2w.c0.xyz * dir_cam.x +
            c2w.c1.xyz * dir_cam.y +
            c2w.c2.xyz * dir_cam.z
        ));

        float3 ray_o = c2w.c3.xyz;

        // ray와 uv 평면의 교차점 계산
        float t_uv = (uv_depth - ray_o.z) / ray_d.z;
        float3 inter_uv = ray_o + t_uv * ray_d;

        // ray와 st 평면의 교차점 계산
        float t_st = (st_depth - ray_o.z) / ray_d.z;
        float3 inter_st = ray_o + t_st * ray_d;

        uvstOut[index] = new float4(inter_uv.x, inter_uv.y, inter_st.x, inter_st.y);
    }
}
