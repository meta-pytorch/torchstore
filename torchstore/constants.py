import os
MONARCH_HOSTMESH_V1 = os.environ.get("MONARCH_HOSTMESH_V1", "0").lower() in ("1", "true")
