import json
import os

class TextOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "save_to_file": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "output.txt"}),
                "format": (["txt", "json", "markdown"],),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "file_path")
    OUTPUT_NODE = True
    FUNCTION = "output_text"
    CATEGORY = "Sütterlin HTR/Output"

    def output_text(self, text, save_to_file, output_path, format):
        file_path = ""
        if save_to_file and output_path:
            file_path = output_path
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)) or '.', exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                if format == "json":
                    json.dump({"text": text}, f, ensure_ascii=False, indent=2)
                else:
                    f.write(text)
                    
        # Return string to be displayed in a UI widget if connected
        return {"ui": {"string": [text]}, "result": (text, file_path)}


class TextToJSON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "key_name": ("STRING", {"default": "text"}),
            }
        }
    
    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("json",)
    FUNCTION = "convert"
    CATEGORY = "Sütterlin HTR/Output"

    def convert(self, text, key_name):
        return ({key_name: text},)
