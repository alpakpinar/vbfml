import os
import tarfile

from vbfml.util import vbfml_path


def get_repo_files():
    """Returns a list of tracked files in the vbfml repo"""
    import git

    repo = git.Repo(vbfml_path(".."))

    to_iterate = [repo.tree()]
    to_add = []

    while len(to_iterate):
        for item in to_iterate.pop():
            if item.type == "tree":
                to_iterate.append(item)
            elif item.type == "blob":
                to_add.append(item.abspath)
    return to_add


def pack_repo(
    path_to_gridpack: str, training_directory: str = None, overwrite: bool = False
):
    """Creates a gridpack file containing a compressed vbfml repository + the training directory."""
    if os.path.exists(path_to_gridpack) and not overwrite:
        raise RuntimeError(
            f"Gridpack file already exists. Will not overwrite {path_to_gridpack} unless 'overwrite=True' is specified."
        )
    tar = tarfile.open(path_to_gridpack, "w")
    files = get_repo_files()
    for f in files:
        tar.add(
            name=f,
            arcname=f.replace(os.path.abspath(vbfml_path("..")), "vbfml"),
        )
    # Add the training directory to the gridpack
    tar.add(
        name=training_directory,
        arcname=training_directory.replace(os.path.abspath(vbfml_path("..")), "vbfml"),
    )
    tar.close()
    return
