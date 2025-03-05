import atexit
import time
from functools import wraps

# 用于存储方法执行统计信息的字典
method_stats = {}
total_execution_time = 0.0


def timer(func):
    """装饰器，用于跟踪方法执行时间"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        global total_execution_time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        total_execution_time += elapsed

        # 更新方法统计信息
        if func.__name__ in method_stats:
            method_stats[func.__name__]['total_time'] += elapsed
            method_stats[func.__name__]['calls'] += 1
        else:
            method_stats[func.__name__] = {
                'total_time': elapsed,
                'calls': 1
            }

        return result

    return wrapper


def print_statistics():
    """在程序退出时打印统计信息"""
    print("\n=== 执行时间统计 ===")

    # 将统计信息转换为列表并排序
    sorted_stats = sorted(
        method_stats.items(),
        key=lambda x: x[1]['total_time'],
        reverse=True
    )

    for name, stats in sorted_stats:
        avg_time = stats['total_time'] / stats['calls']
        # 使用 ljust 或 rjust 方法对齐输出
        print(
            f"方法 {name}".ljust(35),
            f"调用次数: {stats['calls']}".ljust(15),
            f"总执行时间: {stats['total_time']:.6f} 秒".ljust(20),
            f"平均执行时间: {avg_time:.6f} 秒"
        )

    print(f"=== 总执行时间: {total_execution_time:.6f} 秒 ===")


# 注册退出处理函数
atexit.register(print_statistics)
