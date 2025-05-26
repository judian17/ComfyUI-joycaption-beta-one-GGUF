# Initialize empty mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Attempt to import GGUF nodes
try:
    from . import nodes_gguf 
    
    # Populate mappings directly with GGUF nodes
    NODE_CLASS_MAPPINGS.update({
        "JJC_JoyCaption_GGUF": nodes_gguf.JoyCaptionGGUF,
        "JJC_JoyCaption_Custom_GGUF": nodes_gguf.JoyCaptionCustomGGUF,
        "JJC_JoyCaption_GGUF_ExtraOptions": nodes_gguf.JoyCaptionGGUFExtraOptions,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "JJC_JoyCaption_GGUF": "JoyCaption (GGUF)",
        "JJC_JoyCaption_Custom_GGUF": "JoyCaption (Custom GGUF)",
        "JJC_JoyCaption_GGUF_ExtraOptions": "JoyCaption GGUF Extra Options",
    })
    print("[JoyCaption] GGUF nodes loaded successfully.")
except ImportError as e:
    print(f"[JoyCaption] GGUF nodes not available. Error: {e}")
    print("[JoyCaption] This usually means 'llama-cpp-python' is not installed or there's an issue in 'nodes_gguf.py'.")
except Exception as e: # Catch any other error during import of nodes_gguf
    print(f"[JoyCaption] Error loading GGUF nodes from 'nodes_gguf.py': {e}")
    # Ensure mappings remain empty or minimal if GGUF nodes fail to load
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
