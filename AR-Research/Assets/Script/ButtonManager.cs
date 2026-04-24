using UnityEngine;

public class ButtonManager : MonoBehaviour
{
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
}
