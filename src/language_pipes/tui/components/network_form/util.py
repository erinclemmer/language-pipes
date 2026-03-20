def validate_address(address: str) -> bool:
    parts = address.split(".")
    valid = True
    if len(parts) < 4:
        return False
    
    for part in parts:
        try:
            num = int(part)
            if num < 0 or num > 255:
                valid = False
        except ValueError:
            valid = False
    return valid

def validate_port(port: str):
    valid = True
    try:
        res = int(port)
        if res < 0 or res > 65535:
            valid = False
    except Exception:
        valid = False
    return valid