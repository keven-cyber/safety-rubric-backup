#!/usr/bin/env python3
import os
import shutil
import argparse
from tqdm import tqdm

def copy_tokenizer_files(src, dst):
    """
    复制指定的tokenizer文件到目标目录
    :param src: 源目录路径
    :param dst: 目标目录路径
    """
    # 指定要复制的文件列表
    target_files = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json"
    ]
    
    try:
        # 确保源目录存在
        if not os.path.exists(src):
            raise FileNotFoundError(f"源目录不存在: {src}")

        # 创建目标目录（如果不存在）
        os.makedirs(dst, exist_ok=True)
        
        # 使用进度条复制文件
        with tqdm(total=len(target_files), desc="复制tokenizer文件") as pbar:
            for filename in target_files:
                src_file = os.path.join(src, filename)
                dst_file = os.path.join(dst, filename)
                
                # 检查源文件是否存在
                if os.path.exists(src_file):
                    # 复制文件，保持元数据
                    shutil.copy2(src_file, dst_file)
                    pbar.update(1)
                    print(f"已复制: {filename}")
                else:
                    print(f"警告: 未找到文件 {filename}")

        print(f"\n已完成tokenizer文件从 {src} 到 {dst} 的复制")

    except Exception as e:
        print(f"错误: {str(e)}")
        return False

    return True

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Tokenizer文件复制工具')
    parser.add_argument('--source', help='源目录路径')
    parser.add_argument('--destination', help='目标目录路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行复制
    copy_tokenizer_files(args.source, args.destination)

if __name__ == "__main__":
    main()