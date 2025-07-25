#!/usr/bin/env python3
"""
修复测试数据的数据源标识符
将 omni-math 改为 math_dapo 以兼容现有的奖励函数
"""

import pandas as pd
import os

def fix_test_data():
    # 读取测试数据
    test_file = "/home/wuyu/BNTO/BNTO_verl/data/test_set/omni-math.parquet"
    output_file = "/home/wuyu/BNTO/BNTO_verl/data/test_set/omni-math-fixed.parquet"
    
    print(f"Reading test data from: {test_file}")
    df = pd.read_parquet(test_file)
    
    print(f"Original data_source values: {df['data_source'].unique()}")
    
    # 将 omni-math 改为 math_dapo
    df['data_source'] = df['data_source'].replace('omni-math', 'math_dapo')
    
    print(f"Modified data_source values: {df['data_source'].unique()}")
    
    # 保存修改后的数据
    df.to_parquet(output_file, index=False)
    print(f"Fixed test data saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    fixed_file = fix_test_data()
    print(f"\nYou can now use the fixed test file: {fixed_file}")
    print("Update your script to use this file instead of the original omni-math.parquet") 