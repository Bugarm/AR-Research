using UnityEngine;
using System.IO;
using System.Collections.Generic;
using TMPro;

[System.Serializable]
public class RecipeIngredient
{
    public string name;
    public string desc;
}

[System.Serializable]
public class Recipe
{
    public string name;
    public RecipeIngredient[] ingredients;
}

public class RecipeManager : MonoBehaviour
{
    // Tracks the ingredients that have been successfully checked
    private HashSet<string> collectedIngredients = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
    [SerializeField] TMP_Text recipePanel;
    Recipe recipe;
    string json;
    string filePath;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        filePath = Path.Combine(Application.dataPath, "Models", "RecipeExample.json");

        if (!File.Exists(filePath))
        {
            Debug.LogError($"Recipe file not found at path: {filePath}");
            return;
        }

        json = File.ReadAllText(filePath);
        recipe = JsonUtility.FromJson<Recipe>(json);
        UpdateRecipePanel();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void UpdateRecipePanel()
    { 
        recipePanel.text = $"Recipe: {recipe.name}\nIngredients:\n";
        if (recipe.ingredients != null)
        {
            foreach (var ingredient in recipe.ingredients)
            {
                // Check if ingredient is collected and apply strikethrough formatting
                if (collectedIngredients.Contains(ingredient.name))
                {
                    recipePanel.text += $"- <s>{ingredient.name}</s>\n";
                }
                else
                {
                    recipePanel.text += $"- {ingredient.name}\n";
                }
            }
        }

        // Force TextMeshPro to calculate the new text bounds
        recipePanel.ForceMeshUpdate();

        // Adjust the RectTransform's height to match the new preferred height of the text content
        recipePanel.rectTransform.sizeDelta = new Vector2(recipePanel.rectTransform.sizeDelta.x, recipePanel.preferredHeight);
    }

    private void CheckRecipeFinished(Recipe recipe)
    {
        if (recipe == null || recipe.ingredients == null) return;

        foreach (var ingredient in recipe.ingredients)
        {
            // If any ingredient is missing from our collected set, the recipe is not finished
            if (!collectedIngredients.Contains(ingredient.name))
            {
                return;
            }
        }

        recipePanel.text = "Ingredients Collected!";
        // Force TextMeshPro to calculate the new text bounds
        recipePanel.ForceMeshUpdate();

        // Adjust the RectTransform's height to match the new preferred height of the text content
        recipePanel.rectTransform.sizeDelta = new Vector2(recipePanel.rectTransform.sizeDelta.x, recipePanel.preferredHeight);
        // If we make it through the loop, all ingredients are collected
        Debug.Log("recipe finished");
    }

    public void CheckRecipeWithIngredient()
    {
        if (recipe != null && recipe.ingredients != null)
        {
            foreach (var ingredient in recipe.ingredients)
            {
                //print(ingredient.name + " " + GlobalData.CurrentIngredient);
                if (string.Equals(ingredient.name, GlobalData.CurrentIngredient, System.StringComparison.OrdinalIgnoreCase))
                {
                    Debug.Log($"Recipe contains the current ingredient: {ingredient.name}");
                    
                    // Mark this ingredient as accounted for
                    collectedIngredients.Add(ingredient.name);
                    // Update the recipe panel to show the strikethrough
                    UpdateRecipePanel();

                    // Check if the recipe is now complete
                    CheckRecipeFinished(recipe);

                    return;
                }
            }
        }

        Debug.Log($"Recipe does not contain the current ingredient: {GlobalData.CurrentIngredient}");
    }
}
