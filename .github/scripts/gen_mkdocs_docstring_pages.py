from pathlib import Path
from os import environ
import mkdocs_gen_files

src_dir = Path(environ["APP_FOLDER"])
code_index_file = environ["CODE_INDEX"]
path_prefix = "docstrings"
path_suffix = ".md"

files_found = sorted(src_dir.glob("**/*.py"))

# create docstrings md
for path in files_found:
    if path.name != "__init__.py":
        doc_path = Path(path_prefix, path.relative_to(src_dir)).with_suffix(path_suffix)
        with mkdocs_gen_files.open(doc_path, "w") as f:
            module_path = ".".join(path.with_suffix("").parts)
            #if path.name == "__init__.py":
            #    print(f"# {path.parent.name}", file=f)
            #    print(f"::: {path.parent.name}", file=f)
            #else:
            print(f"# {module_path}", file=f)
            print(f"::: {module_path}", file=f)
        mkdocs_gen_files.set_edit_path(doc_path, path)

# append docstrings to navigation file
with mkdocs_gen_files.open(code_index_file, "a") as nav_file:
    nav_file.write("# App Reference\n\n")
    for path in files_found:
        if path.name != "__init__.py":
            module_path = ".".join(path.with_suffix("").parts)
            doc_path = Path(path_prefix, path.relative_to(src_dir)).with_suffix(path_suffix)
            print(f"* [{module_path}]({doc_path})", file=nav_file)

