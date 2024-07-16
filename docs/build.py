import functools
import subprocess

import tomllib

run = functools.partial(subprocess.run, shell=True)


def build_doc(version: str) -> None:
    run(f"git checkout v{version}")
    run("git checkout main -- conf.py")
    run("git checkout main -- versions.toml")

    run("make html")
    run(f"mv _build/html pages/{version}")
    run("rm -rf _build")
    run("git checkout main")


def build_init_page(version: str) -> None:
    with open("pages/index.html", "w") as f:
        f.write(
            f"""<!doctype html>
<meta http-equiv="refresh" content="0; url=./{version}/index.html">"""
        )


if __name__ == "__main__":
    # create pages folder
    run("rm -rf pages")
    run("mkdir pages")

    # get versions
    with open("meta.toml", "rb") as f:
        versions = tomllib.load(f)["versions"]
    versions.sort(reverse=True, key=lambda v: tuple(map(int, v.split("."))))

    # build documentations for different versions
    for version in versions:
        build_doc(version)

    # build initial page that redirect to the latest version
    build_init_page(versions[0])
