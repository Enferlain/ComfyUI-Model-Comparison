from .nodes import ModelComparisoner

NODE_CLASS_MAPPINGS = {
    "ModelComparisoner": ModelComparisoner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelComparisoner": "Model Comparison Grid"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
