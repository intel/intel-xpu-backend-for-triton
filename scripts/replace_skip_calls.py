"""Replace all 'pytest.skip()' to 'pass' calls."""
import ast
import os

class ReplaceToPassransformer(ast.NodeTransformer): 
    """Replace 'pytest.skip' calls to 'pass' """ 
    def visit_Call(self, node: ast.Call):  
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'skip' and isinstance(node.func.value, ast.Name) \
            and node.func.value.id == 'pytest':
                return ast.Pass()
        self.generic_visit(node)
        return node

def rewrite_test_scripts(src_dir: str, dest_dir: str):
    """Traverse all .py files under src_dir and do the replacement, put all under the dest_dir"""
    if os.path.exists(src_dir) and os.path.isdir(src_dir):
        for dirpath, sub_dirname, filename in os.walk(src_dir):
            # print(f"dirpath: {dirpath}")
            # print(f"sub_dirname: {sub_dirname}")
            # print(f"filename: {filename}")
            for filename in filename:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    with open(file_path, 'r') as file:
                        source_code = file.read()

                    tree = ast.parse(source_code)
                    modified_tree = ReplaceToPassransformer().visit(tree)
                    modified_source_code = ast.unparse(modified_tree)
                    
                    relative_path = os.path.relpath(file_path, src_dir)
                    target_path = os.path.join(dest_dir, relative_path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with open(target_path, 'w') as file:
                        file.write(modified_source_code)
    else:
        print(f"{src_dir} not exist")

if __name__ == '__main__':
    triton_dir = os.getenv('TRITON_PROJ')
    src_dir = os.path.join(triton_dir, "/python/test/") if triton_dir else None
    dest_dir = os.path.join(triton_dir, "/tmp/python/test/") if triton_dir else None
    rewrite_test_scripts(src_dir, dest_dir)
