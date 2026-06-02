using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class DynamicARImageAdder : MonoBehaviour
{
    [SerializeField] private ARTrackedImageManager trackedImageManager;
    
    private MutableRuntimeReferenceImageLibrary mutableLibrary;
    private bool isInitialized = false;

    private void Start()
    {
        // Initialize and load images
        StartCoroutine(InitializeAndLoadImages());
    }

    /// <summary>
    /// Initializes AR and loads images from the directory
    /// </summary>
    private IEnumerator InitializeAndLoadImages()
    {
        if (trackedImageManager == null)
        {
            Debug.LogError("ARTrackedImageManager is not assigned!");
            yield break;
        }

        // Wait a frame for AR to initialize
        yield return null;
        yield return new WaitForSeconds(0.5f);

        // Get the mutable library from the tracker
        InitializeMutableLibrary();

        if (!isInitialized)
        {
            Debug.LogError("Failed to initialize mutable library!");
            yield break;
        }

        // Load images from directory
        yield return StartCoroutine(LoadImagesFromDirectory());
    }

    /// <summary>
    /// Initializes the mutable reference image library
    /// </summary>
    private void InitializeMutableLibrary()
    {
        var referenceLibrary = trackedImageManager.referenceLibrary;

        if (referenceLibrary is MutableRuntimeReferenceImageLibrary mutableLib)
        {
            mutableLibrary = mutableLib;
            isInitialized = true;
            Debug.Log("✓ Mutable library initialized successfully");
        }
        else
        {
            Debug.LogError("✗ Reference library must be a MutableRuntimeReferenceImageLibrary!");
        }
    }

    /// <summary>
    /// Loads all images from Assets/Models/imagesToSave directory
    /// </summary>
    private IEnumerator LoadImagesFromDirectory()
    {
        string sourceDirectory = Path.Combine(Application.dataPath, "Models", "imagesToSave");

        if (!Directory.Exists(sourceDirectory))
        {
            Debug.LogError($"✗ Image directory not found: {sourceDirectory}");
            yield break;
        }

        // Get all image files
        string[] imageExtensions = { "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tga" };
        List<string> imageFiles = new List<string>();

        foreach (string extension in imageExtensions)
        {
            imageFiles.AddRange(Directory.GetFiles(sourceDirectory, extension, SearchOption.TopDirectoryOnly));
        }

        if (imageFiles.Count == 0)
        {
            Debug.LogWarning($"⚠ No images found in: {sourceDirectory}");
            yield break;
        }

        Debug.Log($"✓ Found {imageFiles.Count} images. Adding to Ingredients library...\n");

        int successCount = 0;
        foreach (string imagePath in imageFiles)
        {
            string fileName = Path.GetFileNameWithoutExtension(imagePath);

            Texture2D texture = LoadTextureFromFile(imagePath);
            if (texture == null)
            {
                Debug.LogWarning($"⚠ Skipped: {fileName}");
                continue;
            }

            yield return StartCoroutine(AddImageToLibraryCoroutine(texture, fileName));
            successCount++;
        }

        Debug.Log($"\n✓ Successfully added {successCount}/{imageFiles.Count} images to Ingredients library!");
    }

    /// <summary>
    /// Loads a texture from file
    /// </summary>
    private Texture2D LoadTextureFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            Debug.LogError($"File not found: {filePath}");
            return null;
        }

        byte[] fileData = File.ReadAllBytes(filePath);
        Texture2D texture = new Texture2D(1, 1, TextureFormat.RGB24, false);

        if (!texture.LoadImage(fileData))
        {
            Debug.LogError($"Failed to load image data from: {filePath}");
            Destroy(texture);
            return null;
        }

        texture.name = Path.GetFileNameWithoutExtension(filePath);
        texture.Apply();
        return texture;
    }

    /// <summary>
    /// Adds an image to the mutable library
    /// </summary>
    private IEnumerator AddImageToLibraryCoroutine(Texture2D texture, string imageName, float physicalWidthInMeters = 0.1f)
    {
        if (mutableLibrary == null)
        {
            Debug.LogError("Mutable library not initialized");
            yield break;
        }

        // Schedule the job to add the image
        var jobState = mutableLibrary.ScheduleAddImageWithValidationJob(texture, imageName, physicalWidthInMeters);

        // Wait for the job to complete
        while (!jobState.jobHandle.IsCompleted)
        {
            yield return null;
        }

        Debug.Log($"  ✓ Added '{imageName}' ({texture.width}x{texture.height}px)");
    }

    /// <summary>
    /// Manually add a single image at runtime
    /// </summary>
    public void AddImageAtRuntime(Texture2D newTexture, string targetName, float physicalWidthInMeters)
    {
        if (mutableLibrary == null)
        {
            Debug.LogError("Library not initialized!");
            return;
        }

        if (!isInitialized)
        {
            Debug.LogError("Not yet initialized. Wait for startup to complete.");
            return;
        }

        StartCoroutine(AddImageToLibraryCoroutine(newTexture, targetName, physicalWidthInMeters));
    }
}