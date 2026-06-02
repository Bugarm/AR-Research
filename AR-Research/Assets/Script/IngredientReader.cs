using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

/*
 *  YOLO Inference Script with Image Detection
 *  ============================================
 *
 * Place this script on the Main Camera and set the script parameters according to the tooltips.
 *
 */

public class IngredientReader : MonoBehaviour
{
    [Tooltip("Drag a YOLO model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Drag the classes.txt here")]
    public TextAsset classesAsset;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Drag a border box texture here")]
    public Texture2D borderTexture;

    [Tooltip("Select an appropriate font for the labels")]
    public Font font;

    [Tooltip("Test image for debugging model detection")]
    public Texture2D testImage;

    [Tooltip("Enable camera feed for live detection")]
    [SerializeField]
    bool useCameraFeed = true;

    [Tooltip("Frames to skip between detections (higher = faster but less frequent)")]
    [SerializeField, Range(1, 30)]
    int detectionFrameSkip = 5;

    const BackendType backend = BackendType.GPUCompute;

    private Transform displayLocation;
    private Worker worker;
    private string[] labels;
    private RenderTexture targetRT;
    private Sprite borderSprite;
    private Camera sceneCamera;
    private int frameCounter = 0;

    private UI_Update uiUpdate;

    //Image size for the model
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    private bool hasDetected = false;

    List<GameObject> boxPool = new();

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float scoreThreshold = 0.5f;

    Tensor<float> centersToCorners;

    //bounding box data
    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
    }

    // Threading infrastructure
    private struct InferenceResult
    {
        public List<BoundingBox> boxes;
        public string ingredient;
    }

    private Thread inferenceThread;
    private Queue<Texture2D> inferenceQueue = new Queue<Texture2D>();
    private InferenceResult lastResult;
    private bool isRunning = true;
    private object lockObject = new object();
    private bool hasNewResult = false;
    private Thread cameraInitThread;
    private bool cameraInitialized = false;
    private bool cameraSetupComplete = false;

    void Start()
    {
        //Parse neural net labels
        labels = classesAsset.text.Split('\n');

        LoadModel();

        targetRT = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.Default);
        //Create image to display
        displayLocation = displayImage.transform;

        borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));

        // Start inference thread
        inferenceThread = new Thread(InferenceWorkerThread)
        {
            IsBackground = true,
            Name = "InferenceThread"
        };
        inferenceThread.Start();

        //Initialize camera or test image
        if (useCameraFeed)
        {
            InitializeUnityCameraAsync();
        }
        else if (testImage != null)
        {
            RunDetectionOnImage(testImage);
            hasDetected = true;
        }
    }

    void LoadModel()
    {
        var model1 = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model1);
        var modelOutput = Functional.Forward(model1, inputs)[0];
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
        var allScores = modelOutput[0, 4.., ..];
        var scores = Functional.ReduceMax(allScores, 0);
        var classIDs = Functional.ArgMax(allScores, 0);
        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));
        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);
        var coords = Functional.IndexSelect(boxCoords, 0, indices);
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);

        worker = new Worker(graph.Compile(coords, labelIDs), backend);
    }

    private void InitializeUnityCameraAsync()
    {
        cameraInitThread = new Thread(InitializeUnityCameraThread)
        {
            IsBackground = true,
            Name = "CameraInitThread"
        };
        cameraInitThread.Start();
    }

    private void InitializeUnityCameraThread()
    {
        try
        {
            // Just validate that we need Unity camera
            // Actual Camera.main access happens on main thread in Update()
            Debug.Log("Preparing to initialize Unity Scene Camera...");
            cameraInitialized = true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"Camera initialization error: {ex.Message}");
        }
    }

    private void Update()
    {
        // Complete camera initialization on main thread only (one-time, not every frame)
        if (cameraInitialized && !cameraSetupComplete)
        {
            if (sceneCamera == null)
            {
                // Get Camera.main on main thread
                sceneCamera = Camera.main;

                Debug.Log($"Unity Scene Camera initialized: {sceneCamera.name}");
                cameraSetupComplete = true;
                cameraInitialized = false;
            }
        }

        // Update UI with latest inference results (main thread only)
        lock (lockObject)
        {
            if (hasNewResult)
            {
                UpdateUIWithResults(lastResult);
                hasNewResult = false;
            }
        }

        // Process camera feed and queue inference
        if (useCameraFeed && sceneCamera != null)
        {
            frameCounter++;
            if (frameCounter >= detectionFrameSkip)
            {
                frameCounter = 0;
                CaptureUnityCamera();
            }
        }
    }

    void CaptureUnityCamera()
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = targetRT;

        sceneCamera.targetTexture = targetRT;
        sceneCamera.Render();

        RenderTexture.active = currentRT;
        sceneCamera.targetTexture = null;

        displayImage.texture = targetRT;
        ExecuteMLImmediate();
    }

    public void RunDetectionOnImage(Texture image)
    {
        if (image == null)
        {
            Debug.LogError("Image is null");
            return;
        }

        float aspect = image.width * 1f / image.height;
        Graphics.Blit(image, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
        displayImage.texture = targetRT;

        ExecuteMLImmediate();
    }

    private void InferenceWorkerThread()
    {
        while (isRunning)
        {
            Texture2D imageToProcess = null;

            lock (lockObject)
            {
                if (inferenceQueue.Count > 0)
                {
                    imageToProcess = inferenceQueue.Dequeue();
                }
            }

            if (imageToProcess != null)
            {
                try
                {
                    // Convert texture to RenderTexture for inference
                    Graphics.Blit(imageToProcess, targetRT);
                    var result = ProcessInference();

                    lock (lockObject)
                    {
                        lastResult = result;
                        hasNewResult = true;
                    }

                    Destroy(imageToProcess);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Inference error: {ex.Message}");
                }
            }
            else
            {
                Thread.Sleep(1); // Prevent busy waiting
            }
        }
    }

    private InferenceResult ProcessInference()
    {
        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);
        worker.Schedule(inputTensor);

        using var output = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        using var labelIDs = (worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndClone();

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        int boxesFound = output.shape[0];
        var boxes = new List<BoundingBox>();

        for (int n = 0; n < Mathf.Min(boxesFound, 200); n++)
        {
            boxes.Add(new BoundingBox
            {
                centerX = output[n, 0] * scaleX - displayWidth / 2,
                centerY = output[n, 1] * scaleY - displayHeight / 2,
                width = output[n, 2] * scaleX,
                height = output[n, 3] * scaleY,
                label = labels[labelIDs[n]],
            });
        }

        string ingredient = boxesFound > 0 ? labels[labelIDs[0]].Trim() : "None";

        return new InferenceResult { boxes = boxes, ingredient = ingredient };
    }

    private void ExecuteMLImmediate()
    {
        var result = ProcessInference();
        UpdateUIWithResults(result);
    }

    private void UpdateUIWithResults(InferenceResult result)
    {
        // Disable boxes beyond what was detected
        for (int i = result.boxes.Count; i < boxPool.Count; i++)
        {
            boxPool[i].SetActive(false);
        }

        // Draw the bounding boxes
        for (int n = 0; n < result.boxes.Count; n++)
        {
            DrawBox(result.boxes[n], n, displayImage.rectTransform.rect.height * 0.05f);
        }

        GlobalData.CurrentIngredient = result.ingredient;
        UI_Update.Instance.UpdateIngredient();
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }

        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);

        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        Transform labelTransform = panel.transform.GetChild(0);
        Text label = labelTransform.GetComponent<Text>();
        if (label.text != box.label)
        {
            label.text = box.label;
        }
        if (label.fontSize != (int)fontSize)
        {
            label.fontSize = (int)fontSize;
        }
    }

    public GameObject CreateNewBox(Color color)
    {
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);
        return panel;
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
    }

    void OnDestroy()
    {
        isRunning = false;
        
        if (inferenceThread != null && inferenceThread.IsAlive)
        {
            inferenceThread.Join(5000);
        }

        if (cameraInitThread != null && cameraInitThread.IsAlive)
        {
            cameraInitThread.Join(5000);
        }

        centersToCorners?.Dispose();
        worker?.Dispose();
    }
}

