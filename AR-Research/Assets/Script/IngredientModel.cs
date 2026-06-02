using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

public class IngredientModel : MonoBehaviour
{
    [Tooltip("Drag a YOLO model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Drag the classes.txt here")]
    public TextAsset classesAsset;

    [Tooltip("Link a TextMeshProUGUI component here to display the detected ingredient")]
    public TMP_Text ingredientLabel;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Optional: Link another Raw Image here to show the model's output texture for debugging")]
    public RawImage showRendered;

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

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)]
    float scoreThreshold = 0.5f;

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
    private Thread cameraInitThread;

    private float cachedDisplayWidth;
    private float cachedDisplayHeight;

    private Tensor<float> reusableTensor;
    private Tensor<float> cachedOutput;
    private Tensor<int> cachedLabelIDs;

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

    float i = 0;

    private void Update()
    {
        if (useCameraFeed && sceneCamera != null)
        {
            i++;
            if (i > detectionFrameSkip)
            {
                i = 0;
                ExecuteMLImmediate();
            }
        }
    }

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

    private void BuildWorkerGraph(Model model)
    {
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var modelOutputs = Functional.Forward(model, inputs);

        // Simply compile the model outputs directly without NMS in the graph
        // NMS will be performed during post-processing in ProcessInference()
        worker = new Worker(graph.Compile(modelOutputs), backend);
    }

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

        TextureConverter.ToTensor(displayImage.texture, reusableTensor, default);
        
        if (showRendered != null)
        {
            showRendered.texture = displayImage.texture;
        }

        worker.Schedule(reusableTensor);
        computeJobScheduled = true;

        // Get the model output
        var output = worker.PeekOutput() as Tensor<float>;

        if (output == null)
        {
            return new InferenceResult { boxes = new List<BoundingBox>(), ingredient = "Loading..." };
        }

        using var outputClone = output.ReadbackAndClone();

        float scaleX = cachedDisplayWidth / imageWidth;
        float scaleY = cachedDisplayHeight / imageHeight;

        // Extract boxes and perform NMS manually
        var detections = ExtractDetections(outputClone, scaleX, scaleY);
        detections = ApplyNMS(detections, iouThreshold, scoreThreshold);

        var boxes = new List<BoundingBox>(Mathf.Min(detections.Count, 50));
        for (int n = 0; n < Mathf.Min(detections.Count, 50); n++)
        {
            boxes.Add(detections[n]);
        }

        string ingredient = boxes.Count > 0 ? boxes[0].label : "None";

        computeJobScheduled = false;
        
        return new InferenceResult { boxes = boxes, ingredient = ingredient };
    }

    private List<BoundingBox> ExtractDetections(Tensor<float> output, float scaleX, float scaleY)
    {
        var detections = new List<BoundingBox>();

        // Assuming output shape is [num_detections, 4 + num_classes]
        int numDetections = output.shape[0];
        int numClasses = output.shape[1] - 4;

        for (int i = 0; i < numDetections; i++)
        {
            float cx = output[i, 0];
            float cy = output[i, 1];
            float w = output[i, 2];
            float h = output[i, 3];

            // Find the class with highest confidence
            float maxConfidence = 0;
            int classId = 0;
            for (int c = 0; c < numClasses; c++)
            {
                float confidence = output[i, 4 + c];
                if (confidence > maxConfidence)
                {
                    maxConfidence = confidence;
                    classId = c;
                }
            }

            // Filter by confidence threshold
            if (maxConfidence < scoreThreshold)
                continue;

            string label = classId < labels.Length ? labels[classId].Trim() : "Unknown";

            // Scale coordinates from model space (640x640) to display space
            // The Canvas uses center-based positioning where (0,0) is the center
            float displayCenterX = (cx * scaleX) - (cachedDisplayWidth / 2);
            float displayCenterY = -((cy * scaleY) - (cachedDisplayHeight / 2));

            detections.Add(new BoundingBox
            {
                centerX = displayCenterX,
                centerY = displayCenterY,
                width = w * scaleX,
                height = h * scaleY,
                label = label
            });
        }

        return detections;
    }

    private List<BoundingBox> ApplyNMS(List<BoundingBox> detections, float iouThreshold, float scoreThreshold)
    {
        if (detections.Count == 0)
            return detections;

        var result = new List<BoundingBox>();
        var sorted = new List<BoundingBox>(detections);

        while (sorted.Count > 0)
        {
            // Take the first (highest confidence) detection
            result.Add(sorted[0]);
            var current = sorted[0];
            sorted.RemoveAt(0);

            // Remove detections with high IoU
            sorted.RemoveAll(det => CalculateIoU(current, det) > iouThreshold);
        }

        return result;
    }

    private float CalculateIoU(BoundingBox box1, BoundingBox box2)
    {
        float x1_min = box1.centerX - box1.width / 2;
        float x1_max = box1.centerX + box1.width / 2;
        float y1_min = box1.centerY - box1.height / 2;
        float y1_max = box1.centerY + box1.height / 2;

        float x2_min = box2.centerX - box2.width / 2;
        float x2_max = box2.centerX + box2.width / 2;
        float y2_min = box2.centerY - box2.height / 2;
        float y2_max = box2.centerY + box2.height / 2;

        float inter_x_min = Mathf.Max(x1_min, x2_min);
        float inter_x_max = Mathf.Min(x1_max, x2_max);
        float inter_y_min = Mathf.Max(y1_min, y2_min);
        float inter_y_max = Mathf.Min(y1_max, y2_max);

        if (inter_x_max < inter_x_min || inter_y_max < inter_y_min)
            return 0;

        float intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float union = area1 + area2 - intersection;

        return union > 0 ? intersection / union : 0;
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
        ingredientLabel.text = $"Ingredient: {result.ingredient}";
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

