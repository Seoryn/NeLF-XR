using UnityEngine;
using Unity.Sentis; // Sentis
using UnityEngine.UI;
using UnityEngine.InputSystem; // OVRInput
using System.Diagnostics;
using System.Collections;
using System;

public class NelfRenderer : MonoBehaviour
{
    public int width = 512;
    public int height = 512;
    int B => width * height;

    public RawImage targetUI;
    public bool flipY = false;

    [Tooltip("Drag and drop your ONNX model here")]
    public ModelAsset onnxModel; // Sentis

    private Model runtimeModel;
    private Worker worker;
    private NelfCamera nelfCamera;

    private const float moveStep = 0.3f;
    private float renderInterval = 1f / 72f;    // 1 / fps

    float uv_depth = 1.0f;
    float st_depth = 2.0f;

    private Vector3 movement = Vector3.zero;
    private Quaternion lastRotation = Quaternion.identity;

    private float lastRenderTime = 0f;
    private float inputTime = 0f;       // 입력 시점 저장
    private bool isRendering = false;   // 중복 렌더링 방지

    private Texture2D tex;
    private Color[] colors;

    private const int warmupSkip = 5;
    private double totalHMDRender = 0.0;
    private int totalHMDCount = 0;
    private bool printedAverage = false; // 중복 출력 방지

    private RenderTexture result;

    [Header("Foreground-only (FG)")]
    public bool useFG = false;

    public TextAsset fgOccBin;          // uint8[Nu*Nv*Ns*Nt]
    public TextAsset fgOccShapeBin;     // [Nu,Nv,Ns,Nt]
    public Vector4 rayMin;              // (s0, t0, u0, v0)
    public Vector4 rayMax;              // (s1, t1, u1, v1)

    byte[] occFlat;                     // length = Nu*Nv*Ns*Nt
    int Nu, Nv, Ns, Nt;
    int[] fgIndices;             // 매 프레임 갱신

    void Start()
    {
        if (onnxModel == null)
        {
            UnityEngine.Debug.LogError("ONNX 모델이 없음");
            return;
        }

        InitializeModel();
        InitializeCamera();
        InitializeTexture();

        // [FG] 전경 모드 초기화 (필요 시)
        if (useFG)
        {
            if (!InitializeFG())
            {
                UnityEngine.Debug.LogError("[FG] 초기화 실패하여 ALL 모드로 렌더링");
                useFG = false;
            }
        }

        StartCoroutine(RenderImageCoroutine());
    }

    void InitializeModel()
    {
        try
        {
            var sourceModel = ModelLoader.Load(onnxModel);
            runtimeModel = sourceModel;
            worker = new Worker(runtimeModel, BackendType.GPUCompute);  // GPU
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"Error initializing the model: {e.Message}");
        }
    }

    void InitializeCamera()
    {
        nelfCamera = new NelfCamera(0, 0, 0, 6, 0, 90, height, width);
    }

    void InitializeTexture()
    {
        tex = new Texture2D(width, height, TextureFormat.RGBA32, false, true);
        colors = new Color[width * height];

        if (targetUI)
        {
            targetUI.texture = tex;

            // y축 반전
            if (flipY)
                targetUI.rectTransform.localScale = new Vector3(-1, -1, 1);
            else
                targetUI.rectTransform.localScale = Vector3.one;
        }

        // [FG] RenderTexture 준비 (FG 모드에서만 표시용으로 교체)
        result = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf)
        {
            enableRandomWrite = true,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };
        result.Create();
    }

    // [FG] 전경 모드 초기화
    bool InitializeFG()
    {
        try
        {
            if (fgOccBin == null || fgOccShapeBin == null)
            {
                UnityEngine.Debug.LogError("[FG] fgOccBin / fgOccShapeBin 지정 필요");
                return false;
            }

            // 1) occ shape 로드 (int32[4])
            byte[] shpBytes = fgOccShapeBin.bytes;
            if (shpBytes.Length != 16)
            {
                UnityEngine.Debug.LogError($"[FG] fgOccShapeBin 길이가 16byte가 아님: {shpBytes.Length}");
                return false;
            }
            int[] shape = new int[4];
            Buffer.BlockCopy(shpBytes, 0, shape, 0, shpBytes.Length);
            Nu = shape[0];
            Nv = shape[1];
            Ns = shape[2];
            Nt = shape[3];
            UnityEngine.Debug.Log($"[FG] occ shape = ({Nu},{Nv},{Ns},{Nt})");

            // 2) occ flat 로드 (uint8[])
            occFlat = fgOccBin.bytes;
            if (occFlat.Length != Nu * Nv * Ns * Nt)
            {
                UnityEngine.Debug.LogWarning($"[FG] occ 길이({occFlat.Length}) != Nu*Nv*Ns*Nt({Nu*Nv*Ns*Nt})");
            }

            fgIndices = new int[B];

            return true;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"[FG] InitializeFGDynamic 예외: {e.Message}");
            return false;
        }
    }

    // 연속 좌표 → grid index
    int ToIdx(float val, float vMin, float vMax, int Nbin)
    {
        float t = (val - vMin) / (vMax - vMin + 1e-8f);
        t *= Nbin;
        int idx = Mathf.FloorToInt(t);

        if (idx < 0) idx = 0;
        else if (idx >= Nbin) idx = Nbin - 1;
        return idx;
    }

    // 4D index → 1D index
    int OccFlatIndex(int bu, int bv, int bs, int bt)
    {
        return (((bu * Nv) + bv) * Ns + bs) * Nt + bt;  // occ shape uvst 순서로 가정
    }

    // 매 프레임용 FG index 계산
    int BuildFgIndices(float[] flatUVST)
    {
        // flatUVST: 길이 = B*4, (u,v,s,t) 라고 가정
        float s0 = rayMin.x;
        float t0 = rayMin.y;
        float u0 = rayMin.z;
        float v0 = rayMin.w;

        float s1 = rayMax.x;
        float t1 = rayMax.y;
        float u1 = rayMax.z;
        float v1 = rayMax.w;

        int count = 0;

        // 각 픽셀 ray마다 uvst 꺼냄
        for (int i = 0; i < B; i++)
        {
            int baseIdx = i * 4;

            // uvst 순서
            // float u = flatUVST[baseIdx + 0];
            // float v = flatUVST[baseIdx + 1];
            // float s = flatUVST[baseIdx + 2];
            // float t = flatUVST[baseIdx + 3];

            // int bu = ToIdx(u, u0, u1, Nu);
            // int bv = ToIdx(v, v0, v1, Nv);
            // int bs = ToIdx(s, s0, s1, Ns);
            // int bt = ToIdx(t, t0, t1, Nt);

            // stuv 순서
            float s = flatUVST[baseIdx + 0];
            float t = flatUVST[baseIdx + 1];
            float u = flatUVST[baseIdx + 2];
            float v = flatUVST[baseIdx + 3];

            int bu = ToIdx(u, u0, u1, Nu);
            int bv = ToIdx(v, v0, v1, Nv);
            int bs = ToIdx(s, s0, s1, Ns);
            int bt = ToIdx(t, t0, t1, Nt);

            // int bs = ToIdx(s, s0, s1, Ns);
            // int bt = ToIdx(t, t0, t1, Nt);
            // int bu = ToIdx(u, u0, u1, Nu);
            // int bv = ToIdx(v, v0, v1, Nv);

            int occIdx = OccFlatIndex(bu, bv, bs, bt);
            bool isFg = occFlat[occIdx] != 0;

            if (isFg)
            {
                fgIndices[count] = i;  // 이 픽셀 index는 전경
                count++;
            }
        }
        return count; // N_fg
    }

    void Update()
    {
        movement = Vector3.zero;
        bool buttonPressed = false;

        // if (aButton.action.WasPressedThisFrame())  // A 버튼 → 오른쪽 이동
        // {
        //     movement += Vector3.right * moveStep;
        //     buttonPressed = true;
        // }

        // if (OVRInput.GetDown(OVRInput.Button.Three)) // X 버튼 → 왼쪽 이동
        // {
        //     movement -= Vector3.right * moveStep;
        //     buttonPressed = true;
        // }

        // if (OVRInput.GetDown(OVRInput.Button.Two))  // B 버튼 → 앞으로 이동
        // {
        //     movement += Vector3.forward * moveStep;
        //     buttonPressed = true;
        // }

        // if (OVRInput.GetDown(OVRInput.Button.Four)) // Y 버튼 → 뒤로 이동
        // {
        //     movement -= Vector3.forward * moveStep;
        //     buttonPressed = true;
        // }

        // if (OVRInput.GetDown(OVRInput.Button.PrimaryHandTrigger)) // 왼쪽 트리거(Q) → 아래 이동
        // {
        //     movement.y -= moveStep;
        //     buttonPressed = true;
        // }

        // if (OVRInput.GetDown(OVRInput.Button.SecondaryHandTrigger)) // 오른쪽 트리거(E) → 위로 이동
        // {
        //     movement.y += moveStep;
        //     buttonPressed = true;
        // }

        // 키보드 입력 (WASD + Q/E)
        if (Input.GetKeyDown(KeyCode.D)) // 오른쪽
        {
            movement += Vector3.right * moveStep;
            buttonPressed = true;
        }
        if (Input.GetKeyDown(KeyCode.A)) // 왼쪽
        {
            movement -= Vector3.right * moveStep;
            buttonPressed = true;
        }
        if (Input.GetKeyDown(KeyCode.W)) // 앞으로
        {
            movement += Vector3.forward * moveStep;
            buttonPressed = true;
        }
        if (Input.GetKeyDown(KeyCode.S)) // 뒤로
        {
            movement -= Vector3.forward * moveStep;
            buttonPressed = true;
        }
        if (Input.GetKeyDown(KeyCode.Q)) // 아래
        {
            movement.y -= moveStep;
            buttonPressed = true;
        }
        if (Input.GetKeyDown(KeyCode.E)) // 위로
        {
            movement.y += moveStep;
            buttonPressed = true;
        }

        // 버튼이 눌린 경우 현재 시간 저장
        if (buttonPressed)
        {
            inputTime = Time.realtimeSinceStartup;
        }
    }

    void LateUpdate()
    {
        // Camera.main.transform.position = initialHMDPosition;
        // Camera.main.transform.rotation = initialHMDRotation;

        // 고개 좌우 회전 반영
        var cam = Camera.main;
        if (cam == null) return;

        var rot = cam.transform.rotation;
        // nelfCamera.SetYawFromQuaternion(rot);    // yaw
        nelfCamera.SetC2WFromUnity(cam.transform);  // c2w

        float delta = Quaternion.Angle(rot, lastRotation);
        bool poseChanged = (delta > 0.5f) || (movement != Vector3.zero);
        bool canRenderNow = (Time.time - lastRenderTime) > renderInterval;

        if (!isRendering && canRenderNow && poseChanged)
        {
            if (movement != Vector3.zero)
            {
                nelfCamera.Move(movement.x, movement.y, movement.z);
                movement = Vector3.zero; // 이동 후 즉시 초기화
            }
            
            inputTime = Time.realtimeSinceStartup;

            StartCoroutine(RenderImageCoroutine());
            lastRotation = rot; // 렌더 직후 회전값 갱신
            lastRenderTime = Time.time;
        }
    }

    private float[] Flatten2D(float[,] array)
    {
        int rows = array.GetLength(0);
        int cols = array.GetLength(1);

        float[] result = new float[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i * cols + j] = array[i, j];

        return result;
    }

    private IEnumerator RenderImageCoroutine()
    {
        isRendering = true;

        var totaltimer = Stopwatch.StartNew();
        var stagetimer = new Stopwatch();

        if (worker == null)
        {
            UnityEngine.Debug.LogError("Worker is not initialized.");
            isRendering = false;
            yield break;
        }

        if (!useFG)
        {
            // 1) uvst 계산
            stagetimer.Start();
            // float[,] uvst = nelfCamera.GetUVST();
            float[,] uvst = nelfCamera.GetUVST_Burst(
                width,
                height,
                nelfCamera.GetC2WMatrix(),
                nelfCamera.GetKMatrix(),
                uv_depth,
                st_depth
            );
            stagetimer.Stop();
            UnityEngine.Debug.Log($"uvst 계산: {stagetimer.ElapsedMilliseconds} ms");

            // 2) 입력 tensor 생성
            float[] flatUVST = Flatten2D(uvst);
            var input = new Tensor<float>(new TensorShape(B, 4));
            input.Upload(flatUVST);

            // 3) onnx 추론
            stagetimer.Restart();
            worker.Schedule(input);
            yield return null; // GPU 추론 완료 대기

            // 4) 출력 tensor 다운로드
            var output = worker.PeekOutput() as Tensor<float>;
            float[] pixelData = output.DownloadToArray();

            // 5) Texture2D에 출력
            for (int i = 0; i < colors.Length; i++)
            {
                int baseIndex = i * 3;
                colors[i].r = pixelData[baseIndex + 0];
                colors[i].g = pixelData[baseIndex + 1];
                colors[i].b = pixelData[baseIndex + 2];
                colors[i].a = 1f;
            }
            tex.SetPixels(colors);
            tex.Apply(false, false);

            stagetimer.Stop();
            UnityEngine.Debug.Log($"ONNX 추론 + Texture 업데이트: {stagetimer.ElapsedMilliseconds} ms");

            input.Dispose();
            output.Dispose();
        }

        // [FG] FG 모드
        else
        {
            if (occFlat == null || Nu == 0)
            {
                UnityEngine.Debug.LogError("[FG] 인덱스가 초기화되지 않음");
                useFG = false;
                isRendering = false;
                yield break;
            }

            // 1) uvst 계산
            stagetimer.Start();
            float[,] uvst = nelfCamera.GetUVST_ExplicitPythonStyle(
                width,
                height,
                nelfCamera.GetC2WMatrix()
            );
            stagetimer.Stop();
            UnityEngine.Debug.Log($"[FG] uvst 계산: {stagetimer.ElapsedMilliseconds} ms");

            // 2) 2D → 1D
            float[] flatUVSTRaw = Flatten2D(uvst);     // 길이 = B*4
            float[] flatSTUV = ReorderUVSTtoSTUV(flatUVSTRaw);

            // 3) 이번 프레임 전경 인덱스 계산
            var fgTimer = Stopwatch.StartNew();
            int N_fg = BuildFgIndices(flatSTUV);
            fgTimer.Stop();
            UnityEngine.Debug.Log($"[FG] fgIndices 계산: {fgTimer.ElapsedMilliseconds} ms (N_fg={N_fg})");

            if (N_fg == 0)
            {
                // 전경 없으면 전체 검정
                for (int i = 0; i < colors.Length; i++)
                {
                    colors[i].r = colors[i].g = colors[i].b = 0f;
                    colors[i].a = 1f;
                }
                tex.SetPixels(colors);
                tex.Apply(false, false);
                isRendering = false;
                yield break;
            }

            // 4) 전경 ray만
            float[] fgUVST = new float[N_fg * 4];
            for (int i = 0; i < N_fg; i++)
            {
                int srcIdx = fgIndices[i];
                int srcBase = srcIdx * 4;
                int dstBase = i * 4;

                fgUVST[dstBase + 0] = flatSTUV[srcBase + 0];
                fgUVST[dstBase + 1] = flatSTUV[srcBase + 1];
                fgUVST[dstBase + 2] = flatSTUV[srcBase + 2];
                fgUVST[dstBase + 3] = flatSTUV[srcBase + 3];
            }

            // 5) Sentis 입력: (N_fg, 4)
            using (var input = new Tensor<float>(new TensorShape(N_fg, 4)))
            {
                input.Upload(fgUVST);

                stagetimer.Restart();
                worker.Schedule(input);
                yield return null; // GPU 추론 대기

                var output = worker.PeekOutput() as Tensor<float>;
                float[] fgOut = output.DownloadToArray();
                stagetimer.Stop();
                UnityEngine.Debug.Log($"[FG] ONNX 추론: {stagetimer.ElapsedMilliseconds} ms");

                int channels = output.shape[1]; // 출력 채널 수 (N_fg, 3) 가정

                // 6) 전체 픽셀을 배경으로 초기화
                for (int i = 0; i < colors.Length; i++)
                {
                    colors[i].r = 0f;
                    colors[i].g = 0f;
                    colors[i].b = 0f;
                    colors[i].a = 1f;
                }

                // 7) 전경 위치에만 색 채우기
                for (int i = 0; i < N_fg; i++)
                {
                    int dstIdx = fgIndices[i]; // 0 ~ B-1
                    if (dstIdx < 0 || dstIdx >= colors.Length)
                        continue;

                    int baseIndex = i * channels; // C=3 가정
                    colors[dstIdx].r = fgOut[baseIndex + 0];
                    colors[dstIdx].g = fgOut[baseIndex + 1];
                    colors[dstIdx].b = fgOut[baseIndex + 2];
                }

                tex.SetPixels(colors);
                tex.Apply(false, false);

                output.Dispose();
            }
        }

        // 6) 전체 시간
        totaltimer.Stop();
        UnityEngine.Debug.Log($"전체 렌더링 시간: {totaltimer.ElapsedMilliseconds} ms");

        // 7) 입력부터 HMD 렌더링까지의 시간
        if (inputTime > 0f)
        {
            float endTime = Time.realtimeSinceStartup;
            float totalDelay = (endTime - inputTime) * 1000f; // ms
            UnityEngine.Debug.Log($"입력부터 최종 HMD 렌더링까지 걸린 시간: {totalDelay} ms");

            totalHMDCount++;
            if (totalHMDCount > warmupSkip)
                totalHMDRender += totalDelay;

            inputTime = 0f; // 초기화
        }

        isRendering = false;
    }


    float[] ReorderUVSTtoSTUV(float[] flatUVST)
    {
        float[] flatSTUV = new float[flatUVST.Length];

        for (int i = 0; i < B; i++)
        {
            int b = i * 4;

            float u = flatUVST[b + 0];
            float v = flatUVST[b + 1];
            float s = flatUVST[b + 2];
            float t = flatUVST[b + 3];

            flatSTUV[b + 0] = s;
            flatSTUV[b + 1] = t;
            flatSTUV[b + 2] = u;
            flatSTUV[b + 3] = v;
        }

        return flatSTUV;
    }


    private void ComputeAndLogMetrics(Texture2D textureRendered)
    {
        Texture2D textureOriginal = textureRendered;

        float psnr = CalculatePSNR(textureOriginal, textureRendered);
        float ssim = CalculateSSIM(textureOriginal, textureRendered);

        UnityEngine.Debug.Log($"PSNR: {psnr}, SSIM: {ssim}");
    }

    private float CalculatePSNR(Texture2D original, Texture2D rendered)
    {
        float mse = 0;
        for (int y = 0; y < original.height; y++)
        {
            for (int x = 0; x < original.width; x++)
            {
                Color originalColor = original.GetPixel(x, y);
                Color renderedColor = rendered.GetPixel(x, y);

                mse += Mathf.Pow(originalColor.r - renderedColor.r, 2);
                mse += Mathf.Pow(originalColor.g - renderedColor.g, 2);
                mse += Mathf.Pow(originalColor.b - renderedColor.b, 2);
            }
        }
        mse /= (original.width * original.height * 3);

        if (mse == 0)
        {
            return 50f; // PSNR 값이 무한대에 가까운 경우 100으로 반환 (적절히 설정)
        }

        float maxPixelValue = 1.0f; // 색상값이 0에서 1 사이일 때
        return 20 * Mathf.Log10(maxPixelValue / Mathf.Sqrt(mse));
    }

    private float CalculateSSIM(Texture2D original, Texture2D rendered)
    {
        float c1 = 6.5025f, c2 = 58.5225f;
        float meanOriginal = 0f, meanRendered = 0f;
        float varOriginal = 0f, varRendered = 0f;
        float cov = 0f;

        for (int y = 0; y < original.height; y++)
        {
            for (int x = 0; x < original.width; x++)
            {
                Color originalColor = original.GetPixel(x, y);
                Color renderedColor = rendered.GetPixel(x, y);

                meanOriginal += originalColor.grayscale;
                meanRendered += renderedColor.grayscale;
            }
        }
        meanOriginal /= (original.width * original.height);
        meanRendered /= (original.width * original.height);

        for (int y = 0; y < original.height; y++)
        {
            for (int x = 0; x < original.width; x++)
            {
                Color originalColor = original.GetPixel(x, y);
                Color renderedColor = rendered.GetPixel(x, y);

                varOriginal += Mathf.Pow(originalColor.grayscale - meanOriginal, 2);
                varRendered += Mathf.Pow(renderedColor.grayscale - meanRendered, 2);
                cov += (originalColor.grayscale - meanOriginal) * (renderedColor.grayscale - meanRendered);
            }
        }
        varOriginal /= (original.width * original.height);
        varRendered /= (original.width * original.height);
        cov /= (original.width * original.height);

        return (2 * meanOriginal * meanRendered + c1) * (2 * cov + c2) /
               ((meanOriginal * meanOriginal + meanRendered * meanRendered + c1) * (varOriginal + varRendered + c2));
    }

    void OnDestroy()
    {
        PrintAverageHMDDelay();  // 종료 시 평균 렌더링 시간 로그 출력
        worker?.Dispose();
        result?.Release();
    }
    
    void OnApplicationQuit()
    {
        PrintAverageHMDDelay();  // 앱 종료 시에도 한 번 더 보장
    }

    private void PrintAverageHMDDelay()
    {
        if (printedAverage) return;
        printedAverage = true;

        int validSamples = Mathf.Max(0, totalHMDCount - warmupSkip);
        if (validSamples > 0)
        {
            double avg = totalHMDRender / validSamples;
            UnityEngine.Debug.Log($"[AVG] HMD 렌더링 평균: {avg:F3} ms (N={validSamples})");
        }
        else
        {
            UnityEngine.Debug.Log("[AVG] HMD 렌더링 평균: 유효 샘플 없음");
        }
    }

    // [FG] 유틸: RT 클리어 & meta 클래스
    void ClearRT(RenderTexture rt, Color c)
    {
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        GL.Clear(true, true, c);
        RenderTexture.active = prev;
    }

    [Serializable]
    private class Meta
    {
        public int width, height;
        public float fx, fy, cx, cy;
    }
}