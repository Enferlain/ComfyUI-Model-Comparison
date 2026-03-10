from .nodes import ModelComparisoner

NODE_CLASS_MAPPINGS = {"ModelComparisoner": ModelComparisoner}

NODE_DISPLAY_NAME_MAPPINGS = {"ModelComparisoner": "Model Comparison Grid"}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
