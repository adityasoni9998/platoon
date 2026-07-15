import os
import json
import re
import subprocess
import tempfile
import collections
import argparse
import logging
import logging.handlers
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from queue import Empty
import uuid
import shutil
import unidiff
import libcst as cst
import libcst.matchers as m
import ast
import tokenize
from io import StringIO
import logging


def parse_class_docstrings(target_file: str) -> list:
    """Parse docstrings from classes AND functions/methods."""
    try:
        with open(target_file, 'r', encoding="utf-8-sig") as f:
            source_code = f.read()
    except Exception as e:
        return []
        
    # Parse the code string
    try:
        parsed_code = ast.parse(source_code)
    except SyntaxError:
        return []
    
    docstring_nodes = []
    # Iterate through nodes to find class and function definitions
    for node in ast.walk(parsed_code):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Retrieve the docstring
            docstring = ast.get_docstring(node)
            if docstring:
                # Find the start and end lines of the docstring
                # The docstring is the first statement in the body
                if node.body and isinstance(node.body[0], ast.Expr):
                    docstring_node = node.body[0]
                    start_line = docstring_node.lineno
                    # Prefer end_lineno if available, otherwise fall back to counting lines
                    if hasattr(docstring_node, 'end_lineno') and docstring_node.end_lineno is not None:
                        end_line = docstring_node.end_lineno
                    else:
                        end_line = start_line + len(docstring.splitlines()) - 1
                    docstring_nodes.append({
                        'start_line': start_line,
                        'end_line': end_line,
                        'content': docstring
                    })
    return docstring_nodes

def parse_import_nodes(target_file):
    try:
        with open(target_file, 'r', encoding="utf-8-sig") as f:
            source_code = f.read()
    except Exception as e:
        return []

    # Parse the source code
    tree = ast.parse(source_code)
    class ImportCollector(ast.NodeVisitor):
        def __init__(self):
            self.imports = []

        def visit_Import(self, node):
            self.imports.append({
                "type": "import",
                "module": None,  # Regular imports don't specify a module
                "names": [alias.name for alias in node.names],
                "start_line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)  # Use node.lineno if end_lineno is not available
            })
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            self.imports.append({
                "type": "from import",
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "start_line": node.lineno,
                "end_line": getattr(node, 'end_lineno', node.lineno)  # Use node.lineno if end_lineno is not available
            })
            self.generic_visit(node)

    import_collector = ImportCollector()
    import_collector.visit(tree)

    # return the collected imports
    return import_collector.imports


def parse_comment_nodes(target_file):
    comment_nodes = []
    try:
        with open(target_file, 'r', encoding="utf-8-sig") as f:
            source_code = f.read()
    except Exception as e:
        return []
    # Tokenize the source code to find comments and their locations
    source = StringIO(source_code)
    tokens = tokenize.generate_tokens(source.readline)

    for token_type, token_string, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            # For comments, this will usually be the same as start_line
            comment_nodes.append({
                "start_line": start[0],
                "end_line": end[0],
                "content": token_string
            })
            logging.debug(f"Found comment: {token_string} starting at line {start[0]} and ending at line {end[0]}")
    return comment_nodes


def is_import_statement(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False


def is_comment(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False 


def is_docstring(line_num, nodes):
    for node in nodes:
        if line_num >= node['start_line'] and line_num <= node['end_line']:
            return True
    return False


class GlobalVariableVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self):
        self.global_assigns = []

    def leave_Module(self, original_node: cst.Module) -> list:
        assigns = []
        for stmt in original_node.body:
            # Match simple assignments
            if m.matches(stmt, m.SimpleStatementLine()) and m.matches(stmt.body[0], m.Assign()):
                start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
                end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
                assigns.append([stmt, start_pos, end_pos])

            # Match annotated assignments (AnnAssign)
            elif m.matches(stmt, m.SimpleStatementLine()) and m.matches(stmt.body[0], m.AnnAssign()):
                start_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).start
                end_pos = self.get_metadata(cst.metadata.PositionProvider, stmt).end
                assigns.append([stmt, start_pos, end_pos])

        self.global_assigns.extend(assigns)


def parse_global_var_from_code(file_content: str) -> dict[str, dict]:
    """Parse global variables."""
    try:
        tree = cst.parse_module(file_content)
    except:
        return file_content

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = GlobalVariableVisitor()
    wrapper.visit(visitor)

    global_assigns = {}
    for assign_stmt, start_pos, end_pos in visitor.global_assigns:
        # Handle both Assign and AnnAssign cases
        if isinstance(assign_stmt.body[0], cst.Assign):
            for t in assign_stmt.body:
                try:
                    targets = [t.targets[0].target.value]
                except:
                    try:
                        targets = t.targets[0].target.elements
                        targets = [x.value.value for x in targets]
                    except:
                        targets = []
                for target_var in targets:
                    global_assigns[target_var] = {
                        "start_line": start_pos.line,
                        "end_line": end_pos.line,
                    }
        elif isinstance(assign_stmt.body[0], cst.AnnAssign):
            targets = [assign_stmt.body[0].target.value]
        else:
            targets = []

        for target_var in targets:
            global_assigns[target_var] = {
                "start_line": start_pos.line,
                "end_line": end_pos.line,
            }
    return global_assigns


def parse_global_var_from_file(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8-sig") as f:
            file_content = f.read()
        global_vars = parse_global_var_from_code(file_content)
        return global_vars
    except Exception as _:
        return {}


def is_global_var(line, global_vars):
    for gvar, lrange in global_vars.items():
        if line >= lrange['start_line'] and line <= lrange['end_line']:
            return gvar
    return None


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"CRITICAL ERROR: Error in reading file {file_path}: {e}")
            return [], [], None
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef) or isinstance(
                    n, ast.AsyncFunctionDef
                ):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()

def get_oracle_filenames(patch, ignore_pr_with_file_add_remove=True):
    """
    Returns the filenames that are changed in the patch
    """
    source_files = []
    file_added_or_removed = False
    for patch_file in unidiff.PatchSet(patch):
        if patch_file.is_added_file or patch_file.is_removed_file:
            file_added_or_removed = True
            continue
        file_path = patch_file.source_file.split("a/", 1)[-1]
        if file_path != "/dev/null":
            source_files.append(file_path)

    if ignore_pr_with_file_add_remove and file_added_or_removed:
        return set()
    gold_docs = set()
    for source_file in source_files:
        gold_docs.add(source_file)
    return gold_docs

def parse_patch(patch, ignore_import=True):
    """
    Parse a git patch into a structured format.

    Parameters:
        patch (str): The git patch as a string.

    Returns:
        list: A list of dictionaries representing the file changes.
    """
    parsed_patches = []
    patch_set = unidiff.PatchSet(patch)
    # Iterate over each file in the patch set
    for patched_file in patch_set:
        # NOTE: we ignore files created/deleted by the patch
        if patched_file.is_added_file or patched_file.is_removed_file:
            continue
        source_file = patched_file.source_file
        if source_file.startswith('a/') or source_file.startswith('b/'):
            source_file = source_file[2:]
        target_file = patched_file.target_file
        if target_file.startswith('a/') or target_file.startswith('b/'):
            target_file = target_file[2:]

        # NOTE: for renamed files, we push the hunks to the old filename
        # There are only 2 cases possible: file is renamed or an existing file is edited - file removal and addition is already ignored
        filename = source_file

        # NOTE: ignore all changes to non-python files
        if not filename.endswith('.py'):
            continue

        parsed_file_patch = dict()
        parsed_file_patch['file'] = [source_file, target_file]
        parsed_file_patch['hunks'] = []
        
        # Iterate over each hunk (a block of changes) in the file
        for hunk in patched_file:
            parsed_hunk = {
                'start_line': hunk.source_start,
                'changes': defaultdict(list)
            }
            
            # Iterate over each line in the hunk
            for line in hunk:
                if not str(line)[1:].strip():
                    continue
                
                if line.is_removed:
                    parsed_hunk['changes']['delete'].append({
                                "content": str(line)[1:],
                                "line": line.source_line_no,
                            })
                elif line.is_added:
                    parsed_hunk['changes']['add'].append({
                                "content": str(line)[1:],
                                "line": line.target_line_no,
                            })
            parsed_file_patch['hunks'].append(parsed_hunk)

        parsed_patches.append(parsed_file_patch)
    return parsed_patches

def group_patch_by_file(patch: str):
    """
    Groups a patch string by file.

    Args:
        patch (str): The patch content as a string.

    Returns:
        dict: A dictionary where the keys are file paths, and the values are the corresponding patch content.
    """
    patch_by_file = defaultdict(list)
    patch_lines = patch.splitlines(keepends=True)

    current_file = None
    file_header_pattern = r"^(---) (.+)"

    for line in patch_lines:
        normalized_line = line.rstrip("\r\n")
        match = re.match(file_header_pattern, normalized_line)
        if match:
            current_file = re.sub(r"^(a/)", "", match.group(2))
            patch_by_file[current_file].append(line)
        else:
            if current_file:
                if normalized_line.startswith("diff --git"):
                    current_file = None
                    continue
                patch_by_file[current_file].append(line)

    return {file: "".join(hunks) for file, hunks in patch_by_file.items() if file != '/dev/null'}

def apply_patch_str(patch, apply_file_path, hunk_size):
    # Write the patch string to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as temp_patch_file:
        temp_patch_file.write(patch)
        temp_patch_file_path = temp_patch_file.name

    # NOTE: Apply the patch using patch Unix utility, assumed to be installed in the environment.
    try:
        result = subprocess.run(
            [
                '/host_tools/lib64/ld-linux-x86-64.so.2',
                '--library-path',
                '/host_tools/lib64',
                '/host_tools/patch',
                '-p1',
                '-i',
                temp_patch_file_path,
                apply_file_path,
            ],
            check=True,
            text=True,
            capture_output=True
        )
        offsets = [0 for i in range(hunk_size)]
        for out in str(result.stdout).splitlines():
            pattern = r"Hunk #(\d+) succeeded at (\d+) \(offset ([+-]?\d+) lines\)"
            match = re.search(pattern, str(out))
            # Extracting the values if a match is found
            if match:
                hunk_id = int(match.group(1))
                offset = int(match.group(3))
                offsets[hunk_id-1] = offset
        return (True, offsets)
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch {patch}: {e.stderr}{e.stdout}")
        return (False, [])
    finally:
        import os
        os.remove(temp_patch_file_path)

def check_module_existed(module, file_structure):
    s = file_structure
    module_type = module.split(':')[0].strip()
    module_name = module.split(':')[-1].strip()
    
    if module_type == 'function' and '.' not in module_name:
        for func in s['functions']:
            if func['name'] == module_name:
                return True
    elif module_type == 'function' and '.' in module_name:
        class_name = module_name.split('.')[0]
        method_name = module_name.split('.')[-1]
        cls = [cls for cls in s['classes'] if cls['name'] == class_name]
        if cls:
            method = [method for method in cls[0]['methods'] if method['name'] == method_name]
            if method:
                return True
    elif module_type == 'class':
        cls = [cls for cls in s['classes'] if cls['name'] == module_name]
        if cls:
            return True
        
    return False

def get_module_from_line_number_with_file_structure(line, file_structure, include_class=True, merge_init=False):
    s = file_structure
    for txt in s['classes']:
        for func in txt['methods']:
            if line >= func['start_line'] and line <= func['end_line']:
                if merge_init and func['name'] == '__init__':
                    desc = f"class: {txt['name']}"
                    return desc
                else:
                    desc = f"function: {txt['name']}.{func['name']}"
                    return desc
                
        # don't belong to any methods
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"class: {txt['name']}"
            if include_class:
                return desc
            else:
                return None
            
    for txt in s['functions']:
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"function: {txt['name']}"
            return desc
    
    return None

def extract_module_from_patch(model_patch, repo_dir, max_edit_file_num=1,
                              logger=None, 
                              include_gvar=False,
                              rank=0,
                              ignore_pr_with_file_add_remove=False):
    edit_files = get_oracle_filenames(model_patch, ignore_pr_with_file_add_remove)
    filtered_edit_files = []
    # NOTE: We ignore all non-python files edited by the PR (IMPORTANT)
    for fle in edit_files:
        if fle.endswith('.py'):
            filtered_edit_files.append(fle)
    if len(filtered_edit_files) == 0:
        print("No python files edited by patch (ignoring added/removed files)")
        return []
    
    file_changes = parse_patch(model_patch)
    # Group the patch by file
    patch_by_file = group_patch_by_file(model_patch)

    updated_file_changes = []
    for file_change in file_changes:
        files = file_change['file']
        file = files[0]
        if not file.endswith('.py'): continue
        target_file_path = os.path.join(repo_dir, file)
        if not os.path.exists(target_file_path):
            print(f"Source file {target_file_path} does not exist.")
            return None
        
        # initial file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        if file_lines is None:
            print(f"Failed to open python file {target_file_path}")
            return None
        old_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        old_global_vars = parse_global_var_from_file(target_file_path)
        old_import_nodes = parse_import_nodes(target_file_path)
        old_comment_nodes = parse_comment_nodes(target_file_path)
        old_docstring_nodes = parse_class_docstrings(target_file_path)
        
        # Extract the partial patch for this file
        partial_patch = patch_by_file.get(file)
        if not partial_patch:
            print(f"No patch found for {file}")
            return None
        
        # Apply the patch
        new_file_path = os.path.join(repo_dir, files[1])
        success, offsets = apply_patch_str(partial_patch, target_file_path, len(file_change['hunks']))
        if not success:
            print(f"Failed to apply patch to {target_file_path}")
            return None
        
        # handle renames
        if files[1] != files[0]:
            if os.path.exists(new_file_path):
                target_file_path = new_file_path
            else:
                new_file_path = target_file_path
        if not os.path.exists(target_file_path):
            print(f"Patched file {new_file_path} does not exist after applying patch.")
            return None

        class_info, function_names, file_lines = parse_python_file(target_file_path)
        if file_lines is None:
            print(f"Failed to open python file {target_file_path}")
            return None
        new_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        new_global_vars = parse_global_var_from_file(new_file_path)
        new_import_nodes = parse_import_nodes(new_file_path)
        new_comment_nodes = parse_comment_nodes(new_file_path)
        new_docstring_nodes = parse_class_docstrings(new_file_path)
        
        changes = collections.defaultdict(list)
        found_non_trivial_change = False
        
        for i, hunk in enumerate(file_change['hunks']):
            # process edited lines
            delete_change = hunk['changes']['delete']
            add_change = hunk['changes']['add']
                        
            for delete in delete_change:
                line = delete['line'] + offsets[i]
                if is_import_statement(line, old_import_nodes):
                    found_non_trivial_change = True
                if is_import_statement(line, old_import_nodes) or \
                    delete['content'].strip().startswith('#') or \
                    is_docstring(line, old_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, old_global_vars)
                if variable:
                    found_non_trivial_change = True
                    if include_gvar and variable not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, old_file_structure)
                found_non_trivial_change = True
                if module and not module in changes['edited_modules']:
                    changes['edited_modules'].append(module)
                    
            for add in add_change:
                line = add['line'] + offsets[i]
                if is_import_statement(line, new_import_nodes):
                    found_non_trivial_change = True
                if is_import_statement(line, new_import_nodes) or \
                    add['content'].strip().startswith('#') or \
                    is_docstring(line, new_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, new_global_vars)
                if variable:
                    found_non_trivial_change = True
                    if not include_gvar:
                        continue
                    if variable in old_global_vars and f'variable: {variable}' not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    elif variable not in old_global_vars and f'variable: {variable}' not in changes['added_modules']:
                        changes['added_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, new_file_structure)
                found_non_trivial_change = True
                if module and \
                    module not in changes['edited_modules'] and \
                    module not in changes['added_modules']:
                    
                    # check if the module existed in old file
                    if check_module_existed(module, old_file_structure):
                        changes['edited_modules'].append(module)
                    else:
                        changes['added_modules'].append(module)
        
        _changes = collections.defaultdict(list)
        for mode, change in changes.items():
            if mode in ['added_lines', 'edited_lines']:
                continue
            for c in change:
                entity_name = c.split(":")[-1].strip()
                if c.startswith("variable:"):
                    continue
                if mode in ['added_modules', 'edited_modules']:
                    _mode = mode.replace('_modules', '_entities')
                    # only append if the entity c does not begin with "class:"
                    if not c.startswith("class:"):
                        op = f"{file}:{entity_name}"
                        if op not in _changes[_mode]:
                            _changes[_mode].append(op)
                
                if c.startswith("function:") and '.' in c:
                    _c = entity_name.split('.')[0]  # class name
                    class_entry = f'{file}:{_c.strip()}'
                    # Check if class existed in old file - if so, it's edited, not added
                    class_existed = any(cls['name'] == _c.strip() for cls in old_file_structure['classes'])
                    if class_existed or mode == "edited_modules":
                        if class_entry not in _changes['edited_modules']:
                            _changes['edited_modules'].append(class_entry)
                    else:
                        if class_entry not in _changes['added_modules']:
                            _changes['added_modules'].append(class_entry)
                else:
                    if f"{file}:{entity_name}" not in _changes[mode]:
                        _changes[mode].append(f"{file}:{entity_name}")
        
        # Ignore files with zero non-trivial changes (for e.g. changes to only comments, docstrings, etc. must be ignored)
        if not found_non_trivial_change:
            continue
        
        updated_file_changes.append({
            'file': file,
            'changes': _changes
        })    
    return updated_file_changes


def generate_edited_locations_for_patch(model_patch, repo_base_dir, max_edit_file_num=100000):
    op = {}
    try:
        # We run inside apptainer, so repo_base_dir is pre-populated for us.
        file_changes = extract_module_from_patch(model_patch, repo_base_dir, max_edit_file_num, ignore_pr_with_file_add_remove=False)
        # assert file_changes is not None, "Failed to extract edited locations from patch and return None"
        if file_changes is None:
            op = {
                "file_changes": [],
                "status": "error"
            }
        else:
            op = {
                "file_changes": file_changes,
                "status": "success"
            }
        return op
    except Exception as e:
        op = {
            "file_changes": [],
            "status": f"error: {str(e)}"
        }
        return op

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edited locations for a given patch.")
    parser.add_argument("--id", type=str, required=True, help="Unique identifier for the instance")
    args = parser.parse_args()

    patch_filename = f"/tmp/{args.id}.patch"
    model_patch = ""
    with open(patch_filename, 'r', newline='') as f:
        model_patch = f.read()
    try:
        edited_locations = generate_edited_locations_for_patch(model_patch, "/testbed")
    except:
        edited_locations = {
            "file_changes": [],
            "status": "error"
        }
    
    with open(f"/tmp/{args.id}_edited_locations.json", 'w') as f:
        json.dump(edited_locations, f)
