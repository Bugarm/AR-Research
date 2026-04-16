using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.Inputs.Readers;
using UnityEngine.Events;

/// <summary>
/// Handles pointer UI detection and provides a template for dropdown menu animations.
/// Detects when pointer is over UI elements and manages dropdown menu visibility with animations.
/// </summary>
public class PointerCheck : MonoBehaviour
{
    [Header("Pointer Detection")]
    private bool m_IsPointerOverUI;

    [Header("Dropdown Menu Configuration")]
    [SerializeField]
    [Tooltip("Button that opens the dropdown menu.")]
    private Button m_DropdownToggleButton;

    /// <summary>
    /// Button that opens the dropdown menu.
    /// </summary>
    public Button dropdownToggleButton
    {
        get => m_DropdownToggleButton;
        set => m_DropdownToggleButton = value;
    }

    [SerializeField]
    [Tooltip("Button that closes the dropdown menu.")]
    private Button m_DropdownCloseButton;

    /// <summary>
    /// Button that closes the dropdown menu.
    /// </summary>
    public Button dropdownCloseButton
    {
        get => m_DropdownCloseButton;
        set => m_DropdownCloseButton = value;
    }

    [SerializeField]
    [Tooltip("The dropdown menu container.")]
    private GameObject m_DropdownMenu;

    /// <summary>
    /// The dropdown menu container.
    /// </summary>
    public GameObject dropdownMenu
    {
        get => m_DropdownMenu;
        set => m_DropdownMenu = value;
    }

    [SerializeField]
    [Tooltip("The animator for the dropdown menu animations.")]
    private Animator m_DropdownMenuAnimator;

    /// <summary>
    /// The animator for the dropdown menu animations.
    /// </summary>
    public Animator dropdownMenuAnimator
    {
        get => m_DropdownMenuAnimator;
        set => m_DropdownMenuAnimator = value;
    }

    [Header("Input Configuration")]
    [SerializeField]
    private XRInputValueReader<Vector2> m_TapStartPositionInput = new XRInputValueReader<Vector2>("Tap Start Position");

    /// <summary>
    /// Input to use for the screen tap start position.
    /// </summary>
    public XRInputValueReader<Vector2> tapStartPositionInput
    {
        get => m_TapStartPositionInput;
        set => XRInputReaderUtility.SetInputProperty(ref m_TapStartPositionInput, value, this);
    }

    [SerializeField]
    private XRInputValueReader<Vector2> m_DragCurrentPositionInput = new XRInputValueReader<Vector2>("Drag Current Position");

    /// <summary>
    /// Input to use for the screen drag current position.
    /// </summary>
    public XRInputValueReader<Vector2> dragCurrentPositionInput
    {
        get => m_DragCurrentPositionInput;
        set => XRInputReaderUtility.SetInputProperty(ref m_DragCurrentPositionInput, value, this);
    }

    [Header("Events")]
    [SerializeField]
    [Tooltip("Invoked when the dropdown menu is shown.")]
    private UnityEvent m_OnDropdownMenuShown = new UnityEvent();

    /// <summary>
    /// Invoked when the dropdown menu is shown.
    /// </summary>
    public UnityEvent onDropdownMenuShown
    {
        get => m_OnDropdownMenuShown;
        set => m_OnDropdownMenuShown = value;
    }

    [SerializeField]
    [Tooltip("Invoked when the dropdown menu is hidden.")]
    private UnityEvent m_OnDropdownMenuHidden = new UnityEvent();

    /// <summary>
    /// Invoked when the dropdown menu is hidden.
    /// </summary>
    public UnityEvent onDropdownMenuHidden
    {
        get => m_OnDropdownMenuHidden;
        set => m_OnDropdownMenuHidden = value;
    }

    private bool m_ShowDropdownMenu;

    /// <summary>
    /// See <see cref="MonoBehaviour"/>.
    /// </summary>
    private void OnEnable()
    {
        if (m_DropdownToggleButton != null)
            m_DropdownToggleButton.onClick.AddListener(ShowDropdownMenu);

        if (m_DropdownCloseButton != null)
            m_DropdownCloseButton.onClick.AddListener(HideDropdownMenu);
    }

    /// <summary>
    /// See <see cref="MonoBehaviour"/>.
    /// </summary>
    private void OnDisable()
    {
        m_ShowDropdownMenu = false;

        if (m_DropdownToggleButton != null)
            m_DropdownToggleButton.onClick.RemoveListener(ShowDropdownMenu);

        if (m_DropdownCloseButton != null)
            m_DropdownCloseButton.onClick.RemoveListener(HideDropdownMenu);
    }

    /// <summary>
    /// See <see cref="MonoBehaviour"/>.
    /// </summary>
    private void Start()
    {
        if (m_DropdownMenu != null)
        {
            HideDropdownMenu();
        }
    }

    /// <summary>
    /// See <see cref="MonoBehaviour"/>.
    /// </summary>
    private void Update()
    {
        UpdatePointerOverUIState();

        if (m_ShowDropdownMenu)
        {
            // Close dropdown if tapped or dragged outside UI bounds
            if (!m_IsPointerOverUI && (m_TapStartPositionInput.TryReadValue(out _) || m_DragCurrentPositionInput.TryReadValue(out _)))
            {
                HideDropdownMenu();
            }
        }
    }

    /// <summary>
    /// Updates the state of whether the pointer is over a UI element.
    /// Uses EventSystem to check if the pointer is over any UI game object.
    /// </summary>
    private void UpdatePointerOverUIState()
    {
        m_IsPointerOverUI = EventSystem.current != null && EventSystem.current.IsPointerOverGameObject(-1);
    }

    /// <summary>
    /// Checks if the pointer is currently over a UI element.
    /// </summary>
    /// <returns>True if the pointer is over UI, false otherwise.</returns>
    public bool CheckIfPointerOverUI()
    {
        return EventSystem.current != null && EventSystem.current.IsPointerOverGameObject(-1);
    }

    /// <summary>
    /// Shows the dropdown menu with animation.
    /// </summary>
    private void ShowDropdownMenu()
    {
        m_ShowDropdownMenu = true;

        if (m_DropdownMenu != null)
        {
            m_DropdownMenu.SetActive(true);
        }

        if (m_DropdownMenuAnimator != null && !m_DropdownMenuAnimator.GetBool("Show"))
        {
            m_DropdownMenuAnimator.SetBool("Show", true);
        }

        m_OnDropdownMenuShown?.Invoke();
    }

    /// <summary>
    /// Hides the dropdown menu with animation.
    /// </summary>
    public void HideDropdownMenu()
    {
        if (m_DropdownMenuAnimator != null)
        {
            m_DropdownMenuAnimator.SetBool("Show", false);
        }

        m_ShowDropdownMenu = false;

        m_OnDropdownMenuHidden?.Invoke();
    }

    /// <summary>
    /// Toggles the dropdown menu visibility.
    /// </summary>
    public void ToggleDropdownMenu()
    {
        if (m_ShowDropdownMenu)
        {
            HideDropdownMenu();
        }
        else
        {
            ShowDropdownMenu();
        }
    }

    /// <summary>
    /// Gets whether the dropdown menu is currently shown.
    /// </summary>
    /// <returns>True if dropdown menu is visible, false otherwise.</returns>
    public bool IsDropdownMenuShown()
    {
        return m_ShowDropdownMenu;
    }
}