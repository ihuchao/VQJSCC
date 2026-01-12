import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import argparse

def export_tensorboard_to_csv(log_dir, output_csv=None):
    """
    从 TensorBoard 日志导出数据到 CSV
    
    Args:
        log_dir: TensorBoard 日志目录路径
        output_csv: 输出 CSV 文件路径（可选）
    """
    
    # 加载 TensorBoard 日志
    print(f"Loading TensorBoard logs from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # 获取所有标量标签
    tags = ea.Tags()['scalars']
    print(f"Found {len(tags)} scalar tags:")
    for tag in tags:
        print(f"  - {tag}")
    
    # 提取所有数据
    all_data = {}
    
    for tag in tags:
        events = ea.Scalars(tag)
        
        # 提取 step 和 value
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        all_data[f"{tag}_step"] = steps
        all_data[f"{tag}_value"] = values
    
    # 找到最大长度（有些标签可能记录频率不同）
    max_len = max(len(v) for v in all_data.values())
    
    # 填充较短的列（用 None）
    for key in all_data:
        if len(all_data[key]) < max_len:
            all_data[key].extend([None] * (max_len - len(all_data[key])))
    
    # 创建 DataFrame
    df = pd.DataFrame(all_data)
    
    # 生成输出文件名
    if output_csv is None:
        output_csv = os.path.join(log_dir, "exported_metrics.csv")
    
    # 保存到 CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Exported to: {output_csv}")
    print(f"   Total rows: {len(df)}")
    print(f"   Total columns: {len(df.columns)}")
    
    return df

def export_by_tag(log_dir, output_dir=None):
    """
    每个标签单独导出为一个 CSV 文件
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, "exported_tags")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading TensorBoard logs from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    
    for tag in tags:
        events = ea.Scalars(tag)
        
        # 创建 DataFrame
        df = pd.DataFrame({
            'step': [e.step for e in events],
            'value': [e.value for e in events],
            'wall_time': [e.wall_time for e in events]
        })
        
        # 安全的文件名（替换 / 为 _）
        safe_tag = tag.replace('/', '_')
        csv_path = os.path.join(output_dir, f"{safe_tag}.csv")
        
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Exported: {safe_tag}.csv ({len(df)} rows)")
    
    print(f"\n✅ All tags exported to: {output_dir}")

def export_specific_tags(log_dir, tags_to_export, output_csv):
    """
    只导出指定的标签
    """
    print(f"Loading TensorBoard logs from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    all_tags = ea.Tags()['scalars']
    
    # 检查标签是否存在
    missing_tags = set(tags_to_export) - set(all_tags)
    if missing_tags:
        print(f"⚠️ Warning: Following tags not found:")
        for tag in missing_tags:
            print(f"  - {tag}")
    
    # 导出数据
    data = {}
    for tag in tags_to_export:
        if tag in all_tags:
            events = ea.Scalars(tag)
            data[f"{tag}_step"] = [e.step for e in events]
            data[f"{tag}_value"] = [e.value for e in events]
    
    # 找到最大长度
    max_len = max(len(v) for v in data.values()) if data else 0
    
    # 填充
    for key in data:
        if len(data[key]) < max_len:
            data[key].extend([None] * (max_len - len(data[key])))
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Exported {len(tags_to_export)} tags to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TensorBoard logs to CSV")
    parser.add_argument("--log_dir", type=str, required=True,
                       help="Path to TensorBoard log directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["all", "by_tag", "specific"],
                       help="Export mode: all (single CSV), by_tag (separate CSVs), specific (selected tags)")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                       help="Specific tags to export (for mode='specific')")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        export_tensorboard_to_csv(args.log_dir, args.output)
    
    elif args.mode == "by_tag":
        export_by_tag(args.log_dir, args.output)
    
    elif args.mode == "specific":
        if not args.tags:
            print("❌ Error: --tags required for mode='specific'")
        else:
            output = args.output or os.path.join(args.log_dir, "selected_metrics.csv")
            export_specific_tags(args.log_dir, args.tags, output)