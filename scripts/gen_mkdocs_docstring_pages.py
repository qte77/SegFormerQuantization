from pathlib import Path
from os import environ
import mkdocs_gen_files

src_dir = Path(environ["APP_FOLDER"])
code_index_file = environ["CODE_INDEX"]

for path in src_dir.glob("**/*.py"):
    doc_path = Path("reference", path.relative_to(src_dir)).with_suffix(".md")
    with mkdocs_gen_files.open(doc_path, "w") as f:
        module_path = ".".join(path.with_suffix("").parts)
        print(f"# {module_path}", file=f)
        print(f"::: {module_path}", file=f)
    mkdocs_gen_files.set_edit_path(doc_path, path)

with mkdocs_gen_files.open(code_index_file, "w") as nav_file:
    nav_file.write("# App Reference\n\n")
    for path in sorted(src_dir.glob("**/*.py")):
        module_path = ".".join(path.with_suffix("").parts)
        doc_path = Path("reference", path.relative_to(src_dir)).with_suffix(".md")
        print(f"* [{module_path}]({doc_path})", file=nav_file)

