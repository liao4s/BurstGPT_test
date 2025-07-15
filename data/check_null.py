import pandas as pd

def check_csv_null_values(file_path):
    """
    检查CSV文件中的空值并返回分析结果
    
    参数:
        file_path (str): CSV文件路径
        
    返回:
        dict: 包含空值统计和分析结果的字典
    """
    try:
        # 读取CSV文件（自动处理不同编码）
        df = pd.read_csv(file_path)
        
        # 检查是否存在空值
        null_exists = df.isnull().any().any()
        
        # 统计每列空值数量
        null_counts = df.isnull().sum()
        null_counts = null_counts[null_counts > 0]  # 只保留有空值的列
        
        # 计算空值占比
        null_percent = (df.isnull().mean() * 100).round(2)
        null_percent = null_percent[null_percent > 0]
        
        # 生成结果字典
        result = {
            "has_null": null_exists,
            "total_rows": len(df),
            "null_columns_count": len(null_counts),
            "null_counts": null_counts.to_dict(),
            "null_percentage": null_percent.to_dict(),
            "top_3_null_columns": null_counts.nlargest(3).to_dict()
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

# 示例使用
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_path = "/workspace/dynamo-eval/BurstGPT_test/data/BurstGPT_1.csv"
    
    # 获取分析结果
    analysis = check_csv_null_values(csv_path)
    
    # 打印分析报告
    if "error" in analysis:
        print(f"处理失败: {analysis['error']}")
    else:
        print("\n===== CSV空值分析报告 =====")
        print(f"数据集总行数: {analysis['total_rows']}")
        print(f"存在空值: {'是' if analysis['has_null'] else '否'}")
        print(f"包含空值的列数: {analysis['null_columns_count']}")
        
        if analysis['has_null']:
            print("\n[各列空值统计]")
            for col, count in analysis['null_counts'].items():
                percent = analysis['null_percentage'][col]
                print(f"- {col}: {count}个空值 ({percent}%)")
            
            print("\n[空值最多的前三列]")
            for col, count in analysis['top_3_null_columns'].items():
                print(f"- {col}: {count}个空值")