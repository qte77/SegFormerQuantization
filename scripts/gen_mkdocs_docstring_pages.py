"""
mkdocstrings: Generate the code reference pages.
See Automatic code reference pages,
https://mkdocstrings.github.io/recipes/
"""
from pathlib import Path
# import mkdocs_gen_files

def process_directory(directory, src, root): # , nav
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix == ".py":
            module_path = path.relative_to(src).with_suffix("")
            doc_path = path.relative_to(src).with_suffix(".md")
            full_doc_path = Path("reference", doc_path)

            parts = tuple(module_path.parts)
            file_pkg_url = ".".join(parts)

            if parts[-1] == "__init__":
                parts = parts[:-1]
            elif parts[-1] == "__main__":
                continue

            # nav[parts] = doc_path.as_posix()

            # with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            #    fd.write(f"::: {file_pkg_url}")
            
            with open(full_doc_path, "w") as f:
                f.write(f"::: {file_pkg_url}")

            # mkdocs_gen_files.set_edit_path(
            #    full_doc_path, path.relative_to(root)
            # )
        elif path.is_dir():
            process_directory(path, src, root) # , nav

# nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "src"
print(f"Checking for Python files in: {src}")

process_directory(src, src, root) # , nav

# with mkdocs_gen_files.open("reference/Code.md", "w") as nav_file:
#    nav_file.writelines(nav.build_literate_nav())

with open("reference/Code.md", "a") as f:
    f.write("","")