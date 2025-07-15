import pandas as pd

def process_csv(input_file, output_file):
    """
    处理CSV文件：删除Request tokens, Response tokens, Total tokens中任一字段为零的行
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 处理后的输出CSV文件路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 原始数据信息
        original_rows = len(df)
        print(f"原始文件行数: {original_rows}")
        
        # 检查三个关键字段中是否有零值
        zero_mask = (df['Request tokens'] == 0) | (df['Response tokens'] == 0) | (df['Total tokens'] == 0)
        zero_rows = df[zero_mask]
        
        if not zero_rows.empty:
            print("\n发现零值行:")
            print(zero_rows[['Timestamp', 'Model', 'Request tokens', 'Response tokens', 'Total tokens']])
            
            # 删除含有零值的行
            df_clean = df[~zero_mask]
            
            # 保存处理后的数据
            df_clean.to_csv(output_file, index=False)
            
            print(f"\n已删除 {len(zero_rows)} 行包含零值的记录")
            print(f"处理后文件行数: {len(df_clean)}")
            print(f"已保存到: {output_file}")
        else:
            print("\n未发现任何零值行")
            df.to_csv(output_file, index=False)
            print(f"原始文件已保存到: {output_file}")
            
        return df_clean if not zero_rows.empty else df
    
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None

# 示例使用
if __name__ == "__main__":
    input_csv = "/workspace/dynamo-eval/BurstGPT_test/data/BurstGPT_1.csv"  # 替换为你的输入文件路径
    output_csv = "/workspace/dynamo-eval/BurstGPT_test/data/cleaned_BurstGPT_1.csv"  # 替换为你的输出文件路径
    
    cleaned_data = process_csv(input_csv, output_csv)
    
    # 显示处理后的数据摘要（如果有处理）
    if cleaned_data is not None and not cleaned_data.empty:
        print("\n处理后的数据摘要:")
        print(cleaned_data[['Model', 'Request tokens', 'Response tokens', 'Total tokens']].describe())