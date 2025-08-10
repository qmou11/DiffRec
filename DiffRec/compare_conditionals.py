#!/usr/bin/env python3
"""
Script to compare conditional vs standard DiffRec training.
This will help you see the impact of adding user group conditionals.
"""

import os
import subprocess
import time
import json
from datetime import datetime

def run_training(dataset, use_conditionals, epochs=50, batch_size=400):
    """Run training with specified parameters"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {'CONDITIONAL' if use_conditionals else 'STANDARD'} training")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Create log directory
    log_dir = f"./log/{dataset}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conditional_suffix = "conditional" if use_conditionals else "standard"
    
    # Build command
    cmd = [
        "python3", "main_conditional.py",
        "--dataset", dataset,
        "--data_path", "../datasets/",
        "--lr", "5e-5",
        "--weight_decay", "0.0",
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--dims", "[1000]",
        "--emb_size", "10",
        "--mean_type", "x0",
        "--steps", "5",
        "--noise_scale", "0.0001",
        "--noise_min", "0.0005",
        "--noise_max", "0.005",
        "--sampling_steps", "0",
        "--reweight",
        "--log_name", "compare",
        "--round", "1",
        "--gpu", "0"
    ]
    
    if use_conditionals:
        cmd.append("--use_conditionals")
    
    # Log file
    log_file = f"{log_dir}/{timestamp}_{conditional_suffix}_training.log"
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    # Run training
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()
            
            process.wait()
            exit_code = process.returncode
            
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return None
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Training completed in {duration:.1f} seconds")
    print(f"Exit code: {exit_code}")
    
    return {
        'type': conditional_suffix,
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'duration': duration,
        'exit_code': exit_code,
        'log_file': log_file
    }

def analyze_results(results):
    """Analyze and compare training results"""
    
    print(f"\n{'='*60}")
    print("📊 TRAINING COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if not results or len(results) < 2:
        print("❌ Not enough results to compare")
        return
    
    # Group results by type
    standard_result = None
    conditional_result = None
    
    for result in results:
        if result['type'] == 'standard':
            standard_result = result
        elif result['type'] == 'conditional':
            conditional_result = result
    
    if not standard_result or not conditional_result:
        print("❌ Missing results for comparison")
        return
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"{'Metric':<20} {'Standard':<15} {'Conditional':<15} {'Difference':<15}")
    print(f"{'-'*70}")
    
    # Duration comparison
    duration_diff = conditional_result['duration'] - standard_result['duration']
    duration_pct = (duration_diff / standard_result['duration']) * 100
    print(f"{'Training Time':<20} {standard_result['duration']:<15.1f}s {conditional_result['duration']:<15.1f}s {duration_diff:+.1f}s ({duration_pct:+.1f}%)")
    
    # Exit code comparison
    print(f"{'Exit Code':<20} {standard_result['exit_code']:<15} {conditional_result['exit_code']:<15} {'✅' if conditional_result['exit_code'] == 0 else '❌'}")
    
    print(f"\n💡 ANALYSIS:")
    if conditional_result['exit_code'] == 0 and standard_result['exit_code'] == 0:
        print("✅ Both training runs completed successfully!")
        if duration_diff > 0:
            print(f"   ⚠️  Conditional training took {duration_diff:.1f}s longer ({duration_pct:.1f}%)")
            print("   💭 This is expected due to additional conditional processing")
        else:
            print(f"   🎉 Conditional training was {abs(duration_diff):.1f}s faster!")
    else:
        print("❌ Some training runs failed")
        if standard_result['exit_code'] != 0:
            print(f"   - Standard training failed with exit code {standard_result['exit_code']}")
        if conditional_result['exit_code'] != 0:
            print(f"   - Conditional training failed with exit code {conditional_result['exit_code']}")
    
    print(f"\n📁 LOG FILES:")
    print(f"   Standard: {standard_result['log_file']}")
    print(f"   Conditional: {conditional_result['log_file']}")
    
    # Save comparison results
    comparison_file = f"./log/{standard_result['dataset']}/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'comparison_time': datetime.now().isoformat(),
            'standard_result': standard_result,
            'conditional_result': conditional_result,
            'analysis': {
                'duration_difference': duration_diff,
                'duration_percentage': duration_pct,
                'both_successful': (standard_result['exit_code'] == 0 and conditional_result['exit_code'] == 0)
            }
        }, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: {comparison_file}")

def main():
    """Main function to run comparison"""
    
    print("🔍 DiffRec Conditional vs Standard Training Comparison")
    print("This script will train both models and compare their performance")
    
    # Configuration
    dataset = 'ml-1m_clean'  # Use smaller dataset for faster comparison
    epochs = 20  # Reduced epochs for faster comparison
    batch_size = 400
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {dataset}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Check if dataset exists
    dataset_path = f"../datasets/{dataset}"
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset {dataset} not found at {dataset_path}")
        print("Please extract the dataset first:")
        print(f"   cd ../datasets && unrar x {dataset}.rar")
        return
    
    print(f"\n✅ Dataset found at {dataset_path}")
    
    # Ask for confirmation
    response = input(f"\nProceed with comparison? This will train both models (y/N): ")
    if response.lower() != 'y':
        print("Comparison cancelled.")
        return
    
    results = []
    
    # Run standard training
    print(f"\n🔄 Running standard training...")
    standard_result = run_training(dataset, use_conditionals=False, epochs=epochs, batch_size=batch_size)
    if standard_result:
        results.append(standard_result)
    
    # Run conditional training
    print(f"\n🔄 Running conditional training...")
    conditional_result = run_training(dataset, use_conditionals=True, epochs=epochs, batch_size=batch_size)
    if conditional_result:
        results.append(conditional_result)
    
    # Analyze results
    analyze_results(results)
    
    print(f"\n🎉 Comparison completed!")
    print("Check the log files for detailed training information.")

if __name__ == "__main__":
    main()
