import os

def print_directory_tree(start_path):
    """
    打印目录树结构，类似tree命令的输出。
    """
    dir_name = os.path.basename(os.path.abspath(start_path))
    print(f"{dir_name}/")
    _list_dir_contents(start_path, prefix="")

def _list_dir_contents(path, prefix):
    """
    递归列出目录内容，生成树状结构
    """
    try:
        # 添加过滤条件：排除隐藏文件、__init__.py 和 .pyc 文件
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
        print(f"{prefix}{connector}{entry}", end="")

        if os.path.isdir(entry_path):
            print("/")
            extension = "    " if is_last else "│   "
            _list_dir_contents(entry_path, prefix + extension)
        else:
            print()

if __name__ == "__main__":
    current_dir = os.getcwd()
    print_directory_tree(current_dir + "/..")
