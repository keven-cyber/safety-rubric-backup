"""Allocate an isolated short Ray root within Linux AF_UNIX socket limits."""
import tempfile

def create_ray_temp():
    # Explicit /tmp avoids a long inherited TMPDIR. mkdtemp creates a new private
    # directory atomically; no prior job directory is reused, cleared or linked.
    return tempfile.mkdtemp(prefix="r12-", dir="/tmp")

if __name__ == "__main__":
    print(create_ray_temp())
