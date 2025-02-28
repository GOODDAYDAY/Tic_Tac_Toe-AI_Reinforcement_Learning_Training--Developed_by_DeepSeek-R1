import ast
import os


def print_directory_tree(start_path):
    """
    打印目录树结构，类似tree命令的输出，并显示文件中的类名
    """
    dir_name = os.path.basename(os.path.abspath(start_path))
    print(f"{dir_name}/")
    _list_dir_contents(start_path, prefix="")


def _get_classes_from_pyfile(filepath):
    """从Python文件中提取类名"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            node = ast.parse(f.read())
    except (UnicodeDecodeError, SyntaxError):
        return []

    classes = []
    for n in node.body:
        if isinstance(n, ast.ClassDef):
            classes.append(n.name)
    return classes

def _list_dir_contents(path, prefix):
    """
    递归列出目录内容，生成树状结构，并显示类信息
    """
    try:
        # 过滤隐藏文件、__init__.py、.pyc文件
        entries = [entry for entry in os.listdir(path)
                   if (not entry.startswith('.') and
                       entry != '__init__.py' and
                       not entry.endswith('.pyc'))]
    except PermissionError:
        print(f"{prefix}└── [权限拒绝]")
        return

    # 分离目录和文件，并分别排序
    dirs = sorted([d for d in entries if os.path.isdir(os.path.join(path, d))])
    files = sorted([f for f in entries if not os.path.isdir(os.path.join(path, f))])
    sorted_entries = dirs + files

    for index, entry in enumerate(sorted_entries):
        is_last = index == len(sorted_entries) - 1
        entry_path = os.path.join(path, entry)

        connector = "└── " if is_last else "├── "
        entry_display = entry

        # 如果是Python文件且非__init__.py，添加类信息
        if entry.endswith('.py') and entry != '__init__.py':
            classes = _get_classes_from_pyfile(entry_path)
            if classes:
                entry_display += f" [Classes: {', '.join(classes)}]"

        print(f"{prefix}{connector}{entry_display}", end="")

        if os.path.isdir(entry_path):
            print("/")  # 目录添加斜杠
            # 跳过名为"1. prompt"的目录内容
            if entry != "1. prompt":
                extension = "    " if is_last else "│   "
                _list_dir_contents(entry_path, prefix + extension)
        else:
            print()  # 文件换行

if __name__ == "__main__":
    current_dir = os.getcwd()
    print_directory_tree(current_dir + "/..")
