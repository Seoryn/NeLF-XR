using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;
using System.Diagnostics;
using System.Collections;

public class BlackKeyRenderer : MonoBehaviour
{
    public ModelAsset onnxModel;

    public int width = 256;
    public int height = 256;
    int B => width * height;

    // public Renderer targetMesh;         // Mesh
    public RawImage targetUI;           // UI

    public Material blackKeyMat;
    [Range(0, 0.2f)] public float thresh = 0.06f;
    [Range(0, 0.2f)] public float smooth = 0.03f;
    [Range(0, 1f)] public float despill = 0.20f;

    public bool flipY = false;

    Model runtimeModel;
    Worker worker;

    Texture2D tex;      // RGBA32
    byte[] raw;         // W*H*4
    float[] rays;       // B*4
    float[] rgb;        // B*3

    private NelfCamera nelfCamera;

    private const float moveStep = 0.3f;
    private float renderInterval = 1f / 72f;    // 1 / fps

    float uvDepth = 1.0f;
    float stDepth = 2.0f;

    private Vector3 movement = Vector3.zero;
    private Quaternion lastRotation = Quaternion.identity;

    private float lastRenderTime = 0f;
    private float inputTime = 0f;
    bool isRendering = false;

    void Start()
    {
        tex = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        tex.wrapMode = TextureWrapMode.Clamp;

        raw = new byte[B*4];
        rays = new float[B*4];
        rgb = new float[B*3];

        // if (targetMesh)
        // {
        //     targetMesh.sharedMaterial = blackKeyMat;
        //     targetMesh.sharedMaterial.mainTexture = tex;
        // }
        if (targetUI)
        {
            targetUI.texture = tex;
            targetUI.material = blackKeyMat;
        }

        runtimeModel = ModelLoader.Load(onnxModel);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        nelfCamera = new NelfCamera(0, 0, 0, 0, 0, 90, height, width);

        StartCoroutine(RenderImageCoroutine());
        lastRenderTime = Time.time;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }

    void Update()
    {
        // 키보드 입력 (WASD + Q/E)
        movement = Vector3.zero;
        bool buttonPressed = false;

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

        if (buttonPressed)
        {
            inputTime = Time.realtimeSinceStartup;
        }
    }

    void LateUpdate()
    {
        // 고개 좌우 회전 반영
        var cam = Camera.main;
        if (cam == null || nelfCamera == null) return;

        var rot = cam.transform.rotation;
        nelfCamera.SetC2WFromUnity(cam.transform);

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
            StartCoroutine(RenderImageCoroutine());
            lastRotation = rot; // 렌더 직후 회전값 갱신
            lastRenderTime = Time.time;
        }
    }

    IEnumerator RenderImageCoroutine()
    {
        isRendering = true;

        var totaltimer = Stopwatch.StartNew();
        var stagetimer = new Stopwatch();

        // 1) UVST 계산
        stagetimer.Start();
        float[,] uvst = nelfCamera.GetUVST_Burst(
            width,
            height,
            nelfCamera.GetC2WMatrix(),
            nelfCamera.GetKMatrix(),
            uvDepth,
            stDepth
        );

        for (int i = 0, j = 0; i < B; i++)
        {
            rays[j++] = uvst[i, 0];
            rays[j++] = uvst[i, 1];
            rays[j++] = uvst[i, 2];
            rays[j++] = uvst[i, 3];
        }

        using var input = new Tensor<float>(new TensorShape(B, 4), rays);
        stagetimer.Stop();
        UnityEngine.Debug.Log($"uvst 계산: {stagetimer.ElapsedMilliseconds} ms");

        // 2) 추론 및 텍스처 업데이트
        stagetimer.Restart();

        worker.Schedule(input);
        yield return null;

        var output = worker.PeekOutput("rgb") as Tensor<float>;
        if (output == null)
            output = worker.PeekOutput() as Tensor<float>;

        float[] rgb = output.DownloadToArray();  // [B*3]

        if (!flipY)
        {
            for (int i = 0, j = 0; i < rgb.Length; i += 3)
            {
                raw[j++] = (byte)Mathf.RoundToInt(Mathf.Clamp01(rgb[i + 0]) * 255f);
                raw[j++] = (byte)Mathf.RoundToInt(Mathf.Clamp01(rgb[i + 1]) * 255f);
                raw[j++] = (byte)Mathf.RoundToInt(Mathf.Clamp01(rgb[i + 2]) * 255f);
                raw[j++] = 255;
            }
        }
        else
        {
            int j = 0;
            for (int y = height - 1; y >= 0; y--)
            {
                int rowStart = y * width * 3;
                for (int x = 0; x < width; x++)
                {
                    int i = rowStart + x * 3;
                    raw[j++] = (byte)Mathf.RoundToInt(Mathf.Clamp01(rgb[i + 0]) * 255f);
                    raw[j++] = (byte)Mathf.RoundToInt(Mathf.Clamp01(rgb[i + 1]) * 255f);
                    raw[j++] = (byte)Mathf.RoundToInt(Mathf.Clamp01(rgb[i + 2]) * 255f);
                    raw[j++] = 255;
                }
            }
        }
        tex.LoadRawTextureData(raw);
        tex.Apply(false, false);

        stagetimer.Stop();
        UnityEngine.Debug.Log($"ONNX 추론 + Texture 업데이트: {stagetimer.ElapsedMilliseconds} ms");

        // Shader 반영
        if (blackKeyMat)
        {
            blackKeyMat.SetFloat("_Thresh", thresh);
            blackKeyMat.SetFloat("_Smooth", smooth);
            blackKeyMat.SetFloat("_Despill", despill);
        }

        totaltimer.Stop();
        UnityEngine.Debug.Log($"전체 렌더링 시간: {totaltimer.ElapsedMilliseconds} ms");

        if (inputTime > 0f)
        {
            float endTime = Time.realtimeSinceStartup;
            float totalDelay = (endTime - inputTime) * 1000f;
            UnityEngine.Debug.Log($"입력부터 최종 HMD 렌더링까지 걸린 시간: {totalDelay:F1} ms");
            inputTime = 0f;
        }

        isRendering = false;
    }
}
