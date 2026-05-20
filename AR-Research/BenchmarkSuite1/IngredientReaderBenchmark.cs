using BenchmarkDotNet.Attributes;
using UnityEngine;
using System.Collections.Generic;
using Microsoft.VSDiagnostics;

[SimpleJob(warmupCount: 3, targetCount: 5)]
[CPUUsageDiagnoser]
public class IngredientReaderBenchmark
{
    private IngredientReader ingredientReader;
    private RenderTexture testRenderTexture;
    private Camera testCamera;
    private Texture2D testImage;
    [GlobalSetup]
    public void Setup()
    {
        // Create a test camera
        var cameraGO = new GameObject("TestCamera");
        testCamera = cameraGO.AddComponent<Camera>();
        testCamera.enabled = false;
        // Create a render texture
        testRenderTexture = new RenderTexture(640, 640, 0);
        testRenderTexture.Create();
        // Create a simple test image
        testImage = new Texture2D(640, 640, TextureFormat.RGB24, false);
        for (int i = 0; i < testImage.width; i++)
        {
            for (int j = 0; j < testImage.height; j++)
            {
                testImage.SetPixel(i, j, Color.white);
            }
        }

        testImage.Apply();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (testRenderTexture != null)
        {
            testRenderTexture.Release();
            Object.Destroy(testRenderTexture);
        }

        if (testImage != null)
        {
            Object.Destroy(testImage);
        }

        if (testCamera != null)
        {
            Object.Destroy(testCamera.gameObject);
        }
    }

    [Benchmark]
    public void BenchmarkCameraRender()
    {
        // Simulate what CaptureUnityCamera does
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = testRenderTexture;
        testCamera.targetTexture = testRenderTexture;
        testCamera.Render(); // This is the operation we want to profile
        RenderTexture.active = currentRT;
        testCamera.targetTexture = null;
    }

    [Benchmark]
    public void BenchmarkRenderTextureSetup()
    {
        // Measure just the RenderTexture setup overhead
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = testRenderTexture;
        RenderTexture.active = currentRT;
    }

    [Benchmark]
    public void BenchmarkGraphicsBlitOperation()
    {
        // Measure Graphics.Blit which is used in RunDetectionOnImage
        RenderTexture targetRT = testRenderTexture;
        float aspect = testImage.width * 1f / testImage.height;
        Graphics.Blit(testImage, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
    }
}