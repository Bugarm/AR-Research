using UnityEngine;

public abstract class GlobalData
{
    private static string currentIngredient = "";

    public static string CurrentIngredient
    {
        get { return currentIngredient; }
        set { currentIngredient = value; }
    }
}
