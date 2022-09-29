from typing import Dict, Any
import hashlib
import json

def dict_hash(dictionary: Dict[str, Any]) -> str:
    md5 = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    md5.update(encoded)
    return md5.hexdigest()


