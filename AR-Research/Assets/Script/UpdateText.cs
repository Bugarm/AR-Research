using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UpdateText : MonoBehaviour
{
    TMP_Text ingredientText;
     void Awake()
    {
        ingredientText = GetComponent<TMP_Text>();
    }
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    int i = 0;
    void Update()
    {
        i++;
        if (i > 20)
        {
            ingredientText.text = GlobalData.CurrentIngredient;
            i = 0;
        }

    }
}
