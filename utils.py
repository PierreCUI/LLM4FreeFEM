import json
import ast
import os


def to_json_obj(resp):
    s = resp.strip()
    try:
        return json.loads(s)
    except:
        pass
    try:
        return ast.literal_eval(s)
    except:
        pass


def write_to_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if type(content) == str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
    elif type(content) == dict:
        with open(path, "w", encoding="utf-8") as f:
            for key, value in content.items():
                f.write(f"{key}\n")
                f.write(f"{value}\n\n")
    else:
        raise TypeError
