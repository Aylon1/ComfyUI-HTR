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
            # If output_path is a directory (or ends with a separator), append a default filename
            if os.path.isdir(file_path) or file_path.endswith(os.sep) or file_path.endswith("/"):
                ext = {"json": ".json", "markdown": ".md"}.get(format, ".txt")
                file_path = os.path.join(file_path.rstrip("/\\"), f"transcription{ext}")
            # Ensure the file has an extension; if not, add one
            elif not os.path.splitext(file_path)[1]:
                ext = {"json": ".json", "markdown": ".md"}.get(format, ".txt")
                file_path = file_path + ext
            # Ensure parent directory exists
            parent_dir = os.path.dirname(os.path.abspath(file_path))
            os.makedirs(parent_dir, exist_ok=True)

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
