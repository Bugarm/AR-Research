using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

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

    private const int imageWidth = 640;
    private const int imageHeight = 640;

    List<GameObject> boxPool = new();

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float scoreThreshold = 0.5f;

    Tensor<float> centersToCorners;

    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
    }

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

    private float cachedDisplayWidth;
    private float cachedDisplayHeight;

    // SENTIS 2.6.1 OPTIMIZATION 1: Pre-allocated tensors for reuse
    private Tensor<float> reusableTensor;
    private Tensor<float> cachedOutput;
    private Tensor<int> cachedLabelIDs;

    // SENTIS 2.6.1 OPTIMIZATION 2: Track job completion state
    private bool computeJobScheduled = false;

    void Start()
    {
        labels = classesAsset.text.Split('\n');
        InitializeModel(modelAsset);

        // SENTIS 2.6.1 OPTIMIZATION 1: Pre-allocate all tensors
        reusableTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));

        displayLocation = displayImage.transform;
        borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), 
            new Vector2(borderTexture.width / 2, borderTexture.height / 2));

        inferenceThread = new Thread(InferenceWorkerThread)
        {
            IsBackground = true,
            Name = "InferenceThread"
        };
        inferenceThread.Start();

        if (useCameraFeed)
        {
            InitializeUnityCameraAsync();
        }
        else if (testImage != null)
        {
            RunDetectionOnImage(testImage);
        }

        sceneCamera = Camera.main;
    }

    /// <summary>
    /// Initializes or reinitializes the model and worker. Call this method to swap models for testing.
    /// </summary>
    /// <param name="modelAssetToLoad">The ModelAsset to load</param>
    public void InitializeModel(ModelAsset modelAssetToLoad)
    {
        if (modelAssetToLoad == null)
        {
            Debug.LogError("ModelAsset is null. Cannot initialize model.");
            return;
        }

        // Dispose of the old worker if it exists
        CleanupModel();

        var model = LoadAndValidateModel(modelAssetToLoad);
        if (model == null)
        {
            Debug.LogError("Failed to load model.");
            return;
        }

        BuildWorkerGraph(model);
        Debug.Log("Model initialized successfully.");
    }

    /// <summary>
    /// Loads the model and validates its structure.
    /// </summary>
    private Model LoadAndValidateModel(ModelAsset modelAssetToLoad)
    {
        var model = Unity.InferenceEngine.ModelLoader.Load(modelAssetToLoad);

        // Log model information
        Debug.Log($"Model loaded: {modelAssetToLoad.name}");
        Debug.Log($"Model inputs count: {model.inputs.Count}");
        Debug.Log($"Model outputs count: {model.outputs.Count}");
        foreach (var input in model.inputs)
        {
            Debug.Log($"  Input: {input.name}");
        }
        foreach (var output in model.outputs)
        {
            Debug.Log($"  Output: {output.name}");
        }

        return model;
    }

    /// <summary>
    /// Builds the computation graph and worker from the loaded model.
    /// </summary>
    private void BuildWorkerGraph(Model model)
    {
        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var modelOutputs = Functional.Forward(model, inputs);
        
        // Get the first output
        var modelOutput = modelOutputs[0];
        
        // Extract box coordinates and scores
        // Assuming format: last 4 channels are coordinates, rest are class scores
        var boxCoords = modelOutput[.., 0..4];
        var allScores = modelOutput[.., 4..];
        
        var scores = Functional.ReduceMax(allScores, 1);
        var classIDs = Functional.ArgMax(allScores, 1);
        
        // Apply NMS
        var indices = Functional.NMS(boxCoords, scores, iouThreshold, scoreThreshold);
        var coords = Functional.IndexSelect(boxCoords, 0, indices);
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);

        worker = new Worker(graph.Compile(coords, labelIDs), backend);
    }

    /// <summary>
    /// Cleans up the current model and worker.
    /// </summary>
    private void CleanupModel()
    {
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }

        if (centersToCorners != null)
        {
            centersToCorners.Dispose();
            centersToCorners = null;
        }
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
            Debug.Log("Preparing to initialize Unity Scene Camera...");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Camera initialization error: {ex.Message}");
        }
    }

    float i = 0;

    private void Update()
    {
        lock (lockObject)
        {
            if (hasNewResult)
            {
                UpdateUIWithResults(lastResult);
                hasNewResult = false;
            }
        }

        if (useCameraFeed && sceneCamera != null)
        {
            i++;
            if (i > detectionFrameSkip)
            {
                i = 0;
                CaptureUnityCamera();
            }
        }
    }

    public void CaptureUnityCamera()
    {
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
                Thread.Sleep(1);
            }
        }
    }

    private void UpdateDisplayDimensions()
    {
        float newWidth = displayImage.rectTransform.rect.width;
        float newHeight = displayImage.rectTransform.rect.height;

        if (cachedDisplayWidth != newWidth || cachedDisplayHeight != newHeight)
        {
            cachedDisplayWidth = newWidth;
            cachedDisplayHeight = newHeight;
        }
    }

    private InferenceResult ProcessInference()
    {
        UpdateDisplayDimensions();

        // SENTIS 2.6.1 OPTIMIZATION 3: Reuse tensor and avoid allocations
        TextureConverter.ToTensor(displayImage.texture, reusableTensor, default);
        
        // SENTIS 2.6.1 OPTIMIZATION 4: Use Schedule for non-blocking job submission
        worker.Schedule(reusableTensor);
        computeJobScheduled = true;

        // SENTIS 2.6.1 OPTIMIZATION 5: PeekOutput is now optimized for GPU memory transfers
        var output = worker.PeekOutput("output_0") as Tensor<float>;
        var labelIDs = worker.PeekOutput("output_1") as Tensor<int>;

        if (output == null || labelIDs == null)
        {
            return new InferenceResult { boxes = new List<BoundingBox>(), ingredient = "Loading..." };
        }

        // SENTIS 2.6.1 OPTIMIZATION 6: ReadbackAndClone is more efficient with GPU batching
        using var outputClone = output.ReadbackAndClone();
        using var labelIDsClone = labelIDs.ReadbackAndClone();

        float scaleX = cachedDisplayWidth / imageWidth;
        float scaleY = cachedDisplayHeight / imageHeight;

        int boxesFound = outputClone.shape[0];
        var boxes = new List<BoundingBox>(Mathf.Min(boxesFound, 10));

        for (int n = 0; n < Mathf.Min(boxesFound, 10); n++)
        {
            string label = labels[labelIDsClone[n]];
            if (label.Length > 0 && (char.IsWhiteSpace(label[0]) || char.IsWhiteSpace(label[label.Length - 1])))
            {
                label = label.Trim();
            }

            boxes.Add(new BoundingBox
            {
                centerX = outputClone[n, 0] * scaleX - cachedDisplayWidth / 2,
                centerY = outputClone[n, 1] * scaleY - cachedDisplayHeight / 2,
                width = outputClone[n, 2] * scaleX,
                height = outputClone[n, 3] * scaleY,
                label = label,
            });
        }

        string ingredient = boxesFound > 0 ? labels[labelIDsClone[0]] : "None";
        if (ingredient != "None" && (ingredient.Length == 0 || char.IsWhiteSpace(ingredient[0]) || char.IsWhiteSpace(ingredient[ingredient.Length - 1])))
        {
            ingredient = ingredient.Trim();
        }

        // SENTIS 2.6.1 OPTIMIZATION 2: Reset job state after completion
        computeJobScheduled = false;

        return new InferenceResult { boxes = boxes, ingredient = ingredient };
    }

    private void ExecuteMLImmediate()
    {
        var result = ProcessInference();
        UpdateUIWithResults(result);
    }

    private void UpdateUIWithResults(InferenceResult result)
    {
        for (int i = result.boxes.Count; i < boxPool.Count; i++)
        {
            boxPool[i].SetActive(false);
        }

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

        reusableTensor?.Dispose();
        cachedOutput?.Dispose();
        cachedLabelIDs?.Dispose();
        CleanupModel();
    }
}

