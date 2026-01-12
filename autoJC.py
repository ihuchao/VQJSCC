import subprocess
import os
import time
import csv
import re
from datetime import datetime

class SNRMetricsCollector:
    def __init__(self, base_model_path="output4/SNR_Sweep"):
        self.base_model_path = base_model_path
        self.snr_values = list(range(0, 22, 2))  # [0, 2, 4, ..., 20]
        self.target_iteration = 30000
        self.results = []
        
    def run_training(self, snr_db, data_path, checkpoint_path=None):
        """运行单次训练"""
        model_path = os.path.join(self.base_model_path, f"SNR_{snr_db}dB")
        os.makedirs(model_path, exist_ok=True)
        
        cmd = [
            "python", "JSCC4.py",
            "-s", data_path,
            "--model_path", model_path,
            "--mock_channel",
            "--snr_db", str(snr_db),
            "--kmeans_st_iter", "20000",
            "--iterations", str(self.target_iteration),
            "--save_iterations", str(self.target_iteration)
        ]
        
        # 如果有 checkpoint，从上次继续训练
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["--start_checkpoint", checkpoint_path])
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting training: SNR = {snr_db} dB")
        print(f"   Model path: {model_path}")
        print(f"{'='*80}\n")
        
        # 运行训练，捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        metrics = None
        for line in process.stdout:
            print(line, end='')  # 实时显示输出
            
            # 解析第 30000 次迭代的指标
            if f"[Channel Training] Iter {self.target_iteration}:" in line:
                metrics = self.parse_metrics(process.stdout, snr_db)
                break
        
        process.wait()
        
        if metrics is None:
            print(f"⚠️ Warning: Failed to capture metrics for SNR={snr_db}dB")
            # 尝试从日志文件读取
            metrics = self.read_from_log(model_path, snr_db)
        
        return metrics
    
    def parse_metrics(self, stdout, snr_db):
        """从标准输出解析指标"""
        metrics = {'SNR_dB': snr_db, 'Iteration': self.target_iteration}
        
        # 读取接下来的几行
        lines = []
        for _ in range(15):  # 读取15行（足够包含所有指标）
            try:
                line = next(stdout)
                lines.append(line)
                print(line, end='')
            except StopIteration:
                break
        
        text = '\n'.join(lines)
        
        # 使用正则表达式提取指标
        patterns = {
            'Render_Loss': r'Render Loss:\s+([\d.]+)',
            'Channel_Loss': r'Channel Loss:\s+([\d.]+)',
            'Index_Loss': r'Index Loss:\s+([\d.]+)',
            'Cont_Loss': r'Cont Loss:\s+([\d.]+)',
            'Total_Loss': r'Total Loss:\s+([\d.]+)',
            'Clean_PSNR': r'Clean PSNR:\s+([\d.]+)\s+dB',
            'Corrupted_PSNR': r'Corrupted PSNR:\s+([\d.]+)\s+dB',
            'PSNR_Drop': r'PSNR Drop:\s+([\d.]+)\s+dB',
            'Clean_SSIM': r'Clean SSIM:\s+([\d.]+)',
            'Corrupted_SSIM': r'Corrupted SSIM:\s+([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                metrics[key] = float(match.group(1))
            else:
                metrics[key] = None
                print(f"⚠️ Warning: Could not parse {key}")
        
        return metrics
    
    def read_from_log(self, model_path, snr_db):
        """从 TensorBoard 日志读取（备用方案）"""
        try:
            from tensorboard.backend.event_processing import event_accumulator
            
            ea = event_accumulator.EventAccumulator(model_path)
            ea.Reload()
            
            metrics = {'SNR_dB': snr_db, 'Iteration': self.target_iteration}
            
            # 读取各个指标
            tags = {
                'Render_Loss': 'channel/render_loss',
                'Channel_Loss': 'channel/channel_loss',
                'Index_Loss': 'channel/index_loss',
                'Cont_Loss': 'channel/cont_loss',
                'Clean_PSNR': 'channel/clean_psnr',
                'Corrupted_PSNR': 'channel/corrupted_psnr',
                'PSNR_Drop': 'channel/psnr_drop',
                'Clean_SSIM': 'channel/clean_ssim',
                'Corrupted_SSIM': 'channel/corrupted_ssim'
            }
            
            for key, tag in tags.items():
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    # 找到最接近 target_iteration 的记录
                    for event in reversed(events):
                        if event.step == self.target_iteration:
                            metrics[key] = event.value
                            break
                else:
                    metrics[key] = None
            
            return metrics
        
        except Exception as e:
            print(f"❌ Error reading from TensorBoard: {e}")
            return None
    
    def collect_all(self, data_path):
        """收集所有 SNR 的数据"""
        print(f"\n{'='*80}")
        print(f"📊 SNR Metrics Collection")
        print(f"   SNR range: {self.snr_values[0]} - {self.snr_values[-1]} dB")
        print(f"   Target iteration: {self.target_iteration}")
        print(f"   Data path: {data_path}")
        print(f"{'='*80}\n")
        
        for snr_db in self.snr_values:
            start_time = time.time()
            
            metrics = self.run_training(snr_db, data_path)
            
            if metrics:
                self.results.append(metrics)
                elapsed = time.time() - start_time
                print(f"\n✅ Completed SNR={snr_db}dB in {elapsed/60:.1f} minutes")
                print(f"   Clean PSNR: {metrics.get('Clean_PSNR', 'N/A')} dB")
                print(f"   Corrupted PSNR: {metrics.get('Corrupted_PSNR', 'N/A')} dB")
                print(f"   PSNR Drop: {metrics.get('PSNR_Drop', 'N/A')} dB\n")
            else:
                print(f"\n❌ Failed to collect metrics for SNR={snr_db}dB\n")
        
        self.save_results()
    
    def save_results(self):
        """保存结果到 CSV"""
        if not self.results:
            print("❌ No results to save")
            return
        
        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.base_model_path, f"metrics_{timestamp}.csv")
        
        # 写入 CSV
        fieldnames = ['SNR_dB', 'Iteration', 'Render_Loss', 'Channel_Loss', 
                     'Index_Loss', 'Cont_Loss', 'Total_Loss',
                     'Clean_PSNR', 'Corrupted_PSNR', 'PSNR_Drop',
                     'Clean_SSIM', 'Corrupted_SSIM']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n{'='*80}")
        print(f"✅ Results saved to: {csv_path}")
        print(f"{'='*80}\n")
        
        # 打印汇总表格
        self.print_summary()
    
    def print_summary(self):
        """打印汇总表格"""
        print("\n📊 Summary Table:")
        print("="*100)
        print(f"{'SNR(dB)':<10} {'Clean PSNR':<15} {'Corrupted PSNR':<18} {'PSNR Drop':<15} {'Clean SSIM':<15}")
        print("="*100)
        
        for metrics in self.results:
            print(f"{metrics['SNR_dB']:<10} "
                  f"{metrics.get('Clean_PSNR', 'N/A'):<15.2f} "
                  f"{metrics.get('Corrupted_PSNR', 'N/A'):<18.2f} "
                  f"{metrics.get('PSNR_Drop', 'N/A'):<15.2f} "
                  f"{metrics.get('Clean_SSIM', 'N/A'):<15.4f}")
        
        print("="*100)

# ========== 主函数 ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect metrics for different SNR values")
    parser.add_argument("-s", "--source_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--output", type=str, default="output4/SNR_Sweep",
                       help="Base output directory")
    parser.add_argument("--target_iter", type=int, default=30000,
                       help="Target iteration to collect metrics")
    parser.add_argument("--snr_min", type=int, default=0,
                       help="Minimum SNR (dB)")
    parser.add_argument("--snr_max", type=int, default=20,
                       help="Maximum SNR (dB)")
    parser.add_argument("--snr_step", type=int, default=2,
                       help="SNR step size (dB)")
    
    args = parser.parse_args()
    
    # 创建收集器
    collector = SNRMetricsCollector(args.output)
    collector.snr_values = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    collector.target_iteration = args.target_iter
    
    # 运行收集
    collector.collect_all(args.source_path)