"""
Plant Disease Class Names Mapping
"""

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_healthy"
]


def get_class_names():
    """Return list of class names"""
    return CLASS_NAMES


def get_class_name(idx):
    """Get class name by index"""
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"Unknown_Class_{idx}"


def get_class_display_name(class_name):
    """Convert class name to display format"""
    return class_name.replace('_', ' ')


def idx_to_display_name(idx):
    """Convert index to display name"""
    return get_class_display_name(get_class_name(idx))
