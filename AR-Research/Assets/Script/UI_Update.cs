using TMPro;
using UnityEngine;

public class UI_Update : MonoBehaviour
{
    public static UI_Update Instance { get; private set; }

    [SerializeField] private TMP_Text curIngredient;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }
        
    // Update is called once per frame
    void Update()
    {
        
    }

    public void UpdateIngredient()
    {
        curIngredient.text = "Current Ingredient: " + GlobalData.CurrentIngredient;
    }
}
