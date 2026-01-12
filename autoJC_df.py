import subprocess
import os
import json
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import sys

def run_snr_sweep_experiment():
    """
    在 Rayleigh 信道下，对比不同 SNR 的性能
    包括 disable_film (Benchmark) 和 enable_film (View-Adaptive) 两种模式
    """
    
    # ========== 实验配置 ==========
    SNR_VALUES = [0, 5, 10, 15, 20, 25, 30]
    CHANNEL_TYPE = 'rayleigh'
    SOURCE_PATH = 'data_drum'
    BASE_OUTPUT = 'output4/SNR_Sweep_Rayleigh'
    
    # 训练配置
    ITERATIONS = 30000
    KMEANS_START_ITER = 20000
    
    # ========== 创建输出目录 ==========
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    
    # ========== 记录实验配置 ==========
    experiment_config = {
        'experiment_name': 'SNR_Sweep_Rayleigh_Benchmark',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'channel_type': CHANNEL_TYPE,
        'snr_values': SNR_VALUES,
        'iterations': ITERATIONS,
        'kmeans_start_iter': KMEANS_START_ITER,
        'film_disabled': True
    }
    
    with open(os.path.join(BASE_OUTPUT, 'experiment_config.json'), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print("\n" + "="*80)
    print("Experiment: SNR Sweep on Rayleigh Channel (Benchmark Mode)")
    print("="*80)
    print(f"Channel Type:      {CHANNEL_TYPE}")
    print(f"SNR Range:         {min(SNR_VALUES)} - {max(SNR_VALUES)} dB")
    print(f"Output Directory:  {BASE_OUTPUT}")
    print(f"FiLM:              Disabled (Benchmark)")
    print("="*80 + "\n")
    
    # ========== 运行实验 ==========
    results = []
    
    for snr in SNR_VALUES:
        print("\n" + "="*80)
        print(f"Running Experiment: SNR = {snr} dB")
        print("="*80)
        
        # 构建输出路径
        output_path = os.path.join(BASE_OUTPUT, f"SNR_{snr}dB")
        log_file = os.path.join(output_path, "training_log.txt")
        os.makedirs(output_path, exist_ok=True)
        
        # 构建命令
        cmd = [
            "python", "JSCC4.py",
            "-s", SOURCE_PATH,
            "--model_path", output_path,
            "--iterations", str(ITERATIONS),
            "--kmeans_st_iter", str(KMEANS_START_ITER),
            
            # 信道配置
            "--mock_channel",
            "--channel_type", CHANNEL_TYPE,
            "--snr_db", str(snr),
            
            # Benchmark 配置
            "--disable_film",
            
            # 学习率配置
            "--lr_index", "1e-2",
            "--lr_cont", "1e-2",
            
            # 量化配置
            "--kmeans_ncls", "64",
            "--kmeans_ncls_sh", "64",
            "--kmeans_ncls_dc", "64",
            
            # 端口配置
            "--port", str(6011 + snr),  # ← 每个SNR用不同端口
        ]
        
        print(f"\nCommand: {' '.join(cmd)}")
        print(f"Log file: {log_file}\n")
        
        # 运行训练
        start_time = time.time()
        
        try:
            # ✅ 使用 Popen 实现实时输出 + 保存日志
            with open(log_file, 'w', buffering=1) as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # 逐行读取并同时输出到屏幕和文件
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                
                # 等待进程结束
                return_code = process.wait()
                
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n✅ SNR={snr}dB completed successfully!")
            print(f"   Time elapsed: {elapsed_time/60:.2f} minutes")
            print(f"   Log saved to: {log_file}")
            
            results.append({
                'snr_db': snr,
                'status': 'success',
                'elapsed_time_min': elapsed_time / 60,
                'output_path': output_path,
                'log_file': log_file
            })
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ SNR={snr}dB failed!")
            print(f"   Return code: {e.returncode}")
            print(f"   Check log: {log_file}")
            
            results.append({
                'snr_db': snr,
                'status': 'failed',
                'return_code': e.returncode,
                'output_path': output_path,
                'log_file': log_file
            })
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user!")
            results.append({
                'snr_db': snr,
                'status': 'interrupted',
                'output_path': output_path
            })
            break
        
        except Exception as e:
            print(f"\n❌ SNR={snr}dB failed!")
            print(f"   Error: {e}")
            
            results.append({
                'snr_db': snr,
                'status': 'failed',
                'error': str(e),
                'output_path': output_path
            })
        
        print("="*80 + "\n")
    
    # ========== 保存实验结果摘要 ==========
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(BASE_OUTPUT, 'experiment_summary.csv'), index=False)
    
    print("\n" + "="*80)
    print("All Experiments Completed!")
    print("="*80)
    print(f"\nResults saved to: {BASE_OUTPUT}/experiment_summary.csv")
    print("\nSummary:")
    print(df_results.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_snr_sweep_experiment()