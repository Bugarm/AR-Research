using System.Collections.Generic;
using System.Linq;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

public class ModelLoader : MonoBehaviour
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

    private Worker worker;
    private RenderTexture processingTexture;
    private Tensor<float> reusableTensor;
    private string[] classNames;
    private int numClasses;
    
    // Cache memory lists to avoid Garbage Collection stutter
    private List<(int classIdx, float finalConf)> allDetections = new List<(int, float)>();
    
    // Run AI every 0.33 seconds (approx 3 frames a second) instead of frame counting
    private float nextRunTime = 0f;
    private const float runInterval = 0.33f; 
    
    private const int modelInputSize = 640;
    
    // Minimum score required to consider a detection valid
    private const float confidenceThreshold = 0.45f; 

    void Start()
    {
        if (classesAsset != null)
        {
            classNames = classesAsset.text
                .Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries)
                .Select(name => name.Trim())
                .ToArray();
            numClasses = classNames.Length;
        }
        else
        {
            Debug.LogError("Classes Asset is missing! Please drag Names.txt to the ModelLoader script.");
            numClasses = 120;
        }

        Model model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor[] inputs = graph.AddInputs(model);
        FunctionalTensor[] outputs = Functional.Forward(model, inputs);

        Model runtimeModel = graph.Compile(outputs);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        processingTexture = new RenderTexture(modelInputSize, modelInputSize, 0);
        processingTexture.Create();

        reusableTensor = new Tensor<float>(new TensorShape(1, 3, modelInputSize, modelInputSize));
    }

    void Update()
    {
        if (Time.unscaledTime >= nextRunTime && displayImage?.texture != null)
        {
            nextRunTime = Time.unscaledTime + runInterval;
            RunAI();
        }
    }

    private void FormatForYOLO(Texture source, RenderTexture dest)
    {
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = dest;

        GL.Clear(true, true, new Color(114f / 255f, 114f / 255f, 114f / 255f, 1f));

        GL.PushMatrix();
        GL.LoadPixelMatrix(0, dest.width, dest.height, 0);

        float scale = Mathf.Min((float)dest.width / source.width, (float)dest.height / source.height);
        float newWidth = source.width * scale;
        float newHeight = source.height * scale;

        float xOffset = (dest.width - newWidth) / 2f;
        float yOffset = (dest.height - newHeight) / 2f;

        Graphics.DrawTexture(new Rect(xOffset, yOffset, newWidth, newHeight), source);

        GL.PopMatrix();
        RenderTexture.active = previous;
    }

    public void RunAI()
    {
        if (classNames == null || classNames.Length == 0) return;

        // Draw the web camera directly into the 640x640 processing box
        FormatForYOLO(displayImage.texture, processingTexture);

        if (showRendered != null) showRendered.texture = processingTexture;

        TextureTransform transform = new TextureTransform().SetTensorLayout(TensorLayout.NCHW);

        TextureConverter.ToTensor(processingTexture, reusableTensor, transform);

        worker.Schedule(reusableTensor);
        
        using Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        float[] results = outputTensor.DownloadToArray();

        allDetections.Clear(); // Recycle list to avoid memory buildup

        int yolov8Features = 4 + numClasses; 
        int yolov5Features = 5 + numClasses; 
        int nmsFeatures = 6;                 

        if (results.Length % nmsFeatures == 0 && results.Length / nmsFeatures <= 1000)
        {
            int numDetections = results.Length / nmsFeatures;
            for (int d = 0; d < numDetections; d++)
            {
                int startIndex = d * nmsFeatures;
                int classIdx = Mathf.RoundToInt(results[startIndex + 4]);
                float score = results[startIndex + 5];

                if (classIdx >= 0 && classIdx <= 1 && score > 1.0f || score % 1 == 0) 
                {
                    score = results[startIndex + 4];
                    classIdx = Mathf.RoundToInt(results[startIndex + 5]);
                }
                
                if (score >= confidenceThreshold && classIdx >= 0 && classIdx < classNames.Length)
                {
                    allDetections.Add((classIdx, score));
                }
            }
        }
        else if (results.Length % yolov8Features == 0)
        {
            int numDetections = results.Length / yolov8Features;
            for (int d = 0; d < numDetections; d++)
            {
                float maxClassScore = 0f;
                int maxClassIdx = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float classScore = results[(4 + c) * numDetections + d];
                    if (classScore > maxClassScore)
                    {
                        maxClassScore = classScore;
                        maxClassIdx = c;
                    }
                }

                if (maxClassIdx >= 0 && maxClassScore >= confidenceThreshold)
                    allDetections.Add((maxClassIdx, maxClassScore));
            }
        }
        else if (results.Length % yolov5Features == 0)
        {
            int numDetections = results.Length / yolov5Features;
            for (int d = 0; d < numDetections; d++)
            {
                int detectionStart = d * yolov5Features;
                float objectness = results[detectionStart + 4]; 

                if (objectness >= confidenceThreshold)
                {
                    float maxClassScore = 0f;
                    int maxClassIdx = -1;

                    for (int c = 0; c < numClasses; c++)
                    {
                        float classScore = results[detectionStart + 5 + c];
                        if (classScore > maxClassScore)
                        {
                            maxClassScore = classScore;
                            maxClassIdx = c;
                        }
                    }

                    if (maxClassIdx >= 0 && (objectness * maxClassScore) >= confidenceThreshold)
                        allDetections.Add((maxClassIdx, objectness * maxClassScore));
                }
            }
        }
        else
        {
            Debug.LogError($"Unsupported Tensor output shape. Length: {results.Length}");
            return;
        }

        if (allDetections.Count == 0)
        {
            ingredientLabel.text = "No ingredient detected";
            return;
        }

        // Only sort after evaluating everything to be faster
        allDetections.Sort((a, b) => b.finalConf.CompareTo(a.finalConf));

        var bestDet = allDetections[0];
        string detectedIngredient = classNames[bestDet.classIdx];
        ingredientLabel.text = $"Detected Ingredient: {detectedIngredient} ({bestDet.finalConf:F2})";
        
        GlobalData.CurrentIngredient = detectedIngredient;
    }

    private void OnDisable()
    {
        worker?.Dispose();
        reusableTensor?.Dispose();
        if (processingTexture != null) processingTexture.Release();
    }
}
