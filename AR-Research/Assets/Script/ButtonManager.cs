using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ButtonManager : MonoBehaviour
{
    [Header("Slide Settings")]
    [Tooltip("How far the UI element should move to the right.")]
    [SerializeField] private float slideDistance = 100f;
    [Tooltip("How long the slide should take in seconds.")]
    [SerializeField] private float slideDuration = 0.5f;

    private bool isSliding = false;
    private bool isToggled = false; // Tracks if it's currently moved to the right
    private Vector2 originalPosition;
    [SerializeField] private RectTransform rectTransform;

    private void Start()
    {
        if (rectTransform != null)
        {
            // Record the exact starting anchored position
            originalPosition = rectTransform.anchoredPosition;
        }
    }

    public void SwitchRecipeScene()
    {
        UnityEngine.SceneManagement.SceneManager.LoadScene("RecipeMobile");
    }

    public void SwitchChosenScene()
    {
        //Save recipe data here
        UnityEngine.SceneManagement.SceneManager.LoadScene("RecipeChosen");
    }

    public void SwitchARmode()
    {
        UnityEngine.SceneManagement.SceneManager.LoadScene("ARmode");
    }

    public void SlideRight()
    {
        if (!isSliding && rectTransform != null)
        {
            StartCoroutine(SlideRoutine());
        }
    }

    private IEnumerator SlideRoutine()
    {
        isSliding = true;

        Vector2 startPosition = rectTransform.anchoredPosition;

        // If it's already toggled, target the original position. Otherwise, target the slid position.
        Vector2 targetPosition = isToggled
            ? originalPosition
            : originalPosition + new Vector2(slideDistance, 0);

        float elapsedTime = 0f;

        while (elapsedTime < slideDuration)
        {
            // Smoothly move using anchoredPosition (designed for UI)
            rectTransform.anchoredPosition = Vector2.Lerp(startPosition, targetPosition, elapsedTime / slideDuration);
            elapsedTime += Time.deltaTime;
            yield return null; // Wait for the next frame
        }

        // Ensure the UI element perfectly snaps to the final position at the end
        rectTransform.anchoredPosition = targetPosition;

        // Flip the toggle state so it goes the other way next time
        isToggled = !isToggled;

        isSliding = false;
    }
}
