import uuid

def generate_uuid(raw_id: str) -> str:
    """Generates a deterministic UUID based on a string input."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
