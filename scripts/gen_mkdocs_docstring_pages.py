"""
mkdocstrings: Generate the code reference pages.
See Automatic code reference pages,
https://mkdocstrings.github.io/recipes/
"""
from pathlib import Path
import mkdocs_gen_files

def process_directory(directory, nav, src, root):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix == ".py":
            module_path = path.relative_to(src).with_suffix("")
            doc_path = path.relative_to(src).with_suffix(".md")
            full_doc_path = Path("reference", doc_path)

            parts = tuple(module_path.parts)

            if parts[-1] == "__init__":
                parts = parts[:-1]
            elif parts[-1] == "__main__":
                continue

            nav[parts] = doc_path.as_posix()

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                ident = ".".join(parts)
                fd.write(f"::: {ident}")

            mkdocs_gen_files.set_edit_path(
                full_doc_path, path.relative_to(root)
            )
        elif path.is_dir():
            process_directory(path, nav, src, root)

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "src"
print(f"mkdocs: {src}")

process_directory(src, nav, src, root)

with mkdocs_gen_files.open("reference/Code.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())