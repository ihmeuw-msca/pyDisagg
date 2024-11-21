import functools
import subprocess
import tomllib

run = functools.partial(subprocess.run, shell=True)


def build_doc(version: str) -> None:
    """Build documentation for a specific version."""
    run(f"git checkout v{version}")
    run("git checkout main -- conf.py")
    run("git checkout main -- meta.toml")

    run("make html")
    run(f"mv _build/html pages/{version}")
    run("rm -rf _build")
    run("git checkout main")


def build_init_page(default_version: str, latest_version: str) -> None:
    """Create the initial page to redirect to the default or latest version."""
    redirect_version = default_version or latest_version
    with open("pages/index.html", "w") as f:
        f.write(
            f"""<!doctype html>
<meta http-equiv="refresh" content="0; url=./{redirect_version}/index.html">"""
        )


if __name__ == "__main__":
    # Create the pages folder
    run("rm -rf pages")
    run("mkdir pages")

    # Load configuration from meta.toml
    with open("meta.toml", "rb") as f:
        meta = tomllib.load(f)

    # Extract default and available versions
    default_version = meta.get("default", {}).get("version", None)
    versions = meta.get("versions", {}).get("available", [])

    # Sort versions in descending order (e.g., 0.6.0 > 0.5.1 > 0.5.0)
    versions.sort(reverse=True, key=lambda v: tuple(map(int, v.split("."))))

    # Build documentation for all versions
    for version in versions:
        build_doc(version)

    # Build the initial page
    build_init_page(default_version, versions[0])
