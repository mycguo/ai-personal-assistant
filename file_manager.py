import json
import os

FILE_DB_PATH = "uploaded_files.json"

def load_files():
    if not os.path.exists(FILE_DB_PATH):
        return []
    try:
        with open(FILE_DB_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_file_entry(name, uri, mime_type):
    files = load_files()
    # Check if file with same name already exists, if so update it
    for file in files:
        if file["name"] == name:
            file["uri"] = uri
            file["mime_type"] = mime_type
            break
    else:
        files.append({"name": name, "uri": uri, "mime_type": mime_type})
    
    with open(FILE_DB_PATH, "w") as f:
        json.dump(files, f, indent=4)

def get_all_file_uris():
    files = load_files()
    return [f["uri"] for f in files]

def clear_files():
    if os.path.exists(FILE_DB_PATH):
        os.remove(FILE_DB_PATH)
