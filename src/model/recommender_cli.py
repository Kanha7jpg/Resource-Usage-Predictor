"""
Command-Line Interface for Resource Recommendation Engine

Generates scaling decisions and resource recommendations from command-line.
"""

import os
import sys
import argparse
import json
import numpy as np
import logging
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.recommender import ResourceRecommender, ScalingPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommenderCLI:
    """Command-line interface for resource recommendations."""
    
    def __init__(self, model_dir: str = 'models', policy: Optional[ScalingPolicy] = None):
        """Initialize CLI."""
        self.recommender = ResourceRecommender(model_dir=model_dir, policy=policy)
    
    def load_data_from_file(self, filepath: str) -> Optional[np.ndarray]:
        """Load data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return np.array(data, dtype=float)
            elif isinstance(data, dict) and 'data' in data:
                return np.array(data['data'], dtype=float)
            else:
                raise ValueError("File must contain a list or dict with 'data' key")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def parse_inline_data(self, data_str: str) -> Optional[np.ndarray]:
        """Parse data from inline JSON string."""
        try:
            data = json.loads(data_str)
            return np.array(data, dtype=float)
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None
    
    def print_recommendation(self, recent_data: np.ndarray, model: str = 'ensemble',
                            current_cpu: int = 500, current_memory: int = 256):
        """Print formatted recommendation."""
        
        # Generate recommendation
        decision = self.recommender.recommend(recent_data, use_model=model)
        
        print("\n" + "="*80)
        print("RESOURCE RECOMMENDATION ENGINE - SCALING DECISION")
        print("="*80)
        
        # Predictions section
        print("\n[CURRENT PREDICTIONS]")
        print(f"  CPU Usage:    {decision.current_cpu_predicted:6.2f}% ({decision.cpu_utilization_level.value})")
        print(f"  Memory Usage: {decision.current_memory_predicted:6.2f}% ({decision.memory_utilization_level.value})")
        
        # Scaling decision
        action = decision.action.value.replace('_', ' ').upper()
        print(f"\n[SCALING ACTION] {action}")
        print(f"  Reason:      {decision.reason}")
        print(f"  Confidence:  {decision.confidence_score:.1%}")
        
        # Current resources
        print(f"\n[CURRENT RESOURCES]")
        print(f"  CPU Request:    {current_cpu}m")
        print(f"  Memory Request: {current_memory}Mi")
        
        # Recommended resources
        rec = decision.recommendation
        print(f"\n[RECOMMENDED RESOURCES]")
        print(f"  CPU Request:    {rec.cpu_request_millicores}m   (limit: {rec.cpu_limit_millicores}m)")
        print(f"  Memory Request: {rec.memory_request_mi}Mi (limit: {rec.memory_limit_mi}Mi)")
        
        # Changes
        cpu_change = rec.cpu_request_millicores - current_cpu
        mem_change = rec.memory_request_mi - current_memory
        cpu_pct = (cpu_change / current_cpu * 100) if current_cpu > 0 else 0
        mem_pct = (mem_change / current_memory * 100) if current_memory > 0 else 0
        
        print(f"\n[RESOURCE CHANGES]")
        print(f"  CPU:    {cpu_change:+5d}m ({cpu_pct:+6.1f}%)")
        print(f"  Memory: {mem_change:+5d}Mi ({mem_pct:+6.1f}%)")
        
        # Kubernetes YAML
        print(f"\n[KUBERNETES RESOURCE SPEC]")
        print(rec.to_kubernetes_yaml())
        
        # Scaling plan
        plan = self.recommender.generate_scaling_plan(current_cpu, current_memory, decision)
        
        print(f"\n[SCALING PLAN (JSON)]")
        print(json.dumps(plan, indent=2))
    
    def run_interactive(self):
        """Run in interactive mode."""
        print("\n" + "="*80)
        print("RESOURCE RECOMMENDATION ENGINE - INTERACTIVE MODE")
        print("="*80)
        
        while True:
            print("\n--- Enter Recent Data ---")
            print("Format: JSON array [[cpu1, mem1], [cpu2, mem2], ...]")
            print("Example: [[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]")
            print("Or type 'quit' to exit\n")
            
            data_input = input("Data: ").strip()
            
            if data_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if not data_input:
                print("Input is empty, please try again.")
                continue
            
            recent_data = self.parse_inline_data(data_input)
            if recent_data is None:
                continue
            
            print("\n--- Resource Configuration ---")
            cpu_input = input("Current CPU Request (millicores, default: 500): ").strip()
            current_cpu = int(cpu_input) if cpu_input else 500
            
            mem_input = input("Current Memory Request (Mi, default: 256): ").strip()
            current_memory = int(mem_input) if mem_input else 256
            
            print("\n--- Select Model ---")
            print("1. Ensemble (Recommended)")
            print("2. LSTM (Advanced)")
            print("3. Baseline (Traditional)")
            
            model_choice = input("Choice (1-3, default: 1): ").strip()
            model_map = {'1': 'ensemble', '2': 'lstm', '3': 'baseline'}
            model = model_map.get(model_choice, 'ensemble')
            
            self.print_recommendation(recent_data, model, current_cpu, current_memory)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Resource Recommendation Engine - CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode
  python recommender_cli.py --interactive
  
  # Generate recommendation with inline data
  python recommender_cli.py --data "[[30.5, 45.2], [31.2, 46.1]]" \\
    --current-cpu 500 --current-memory 256
  
  # Generate recommendation from file
  python recommender_cli.py --file recent_data.json --model ensemble
  
  # Custom scaling policy
  python recommender_cli.py --data "[[30.5, 45.2]]" \\
    --scale-up-cpu 75 --scale-down-cpu 25
        '''
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Recent data as JSON string'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Load data from JSON file'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['lstm', 'baseline', 'ensemble'],
        default='ensemble',
        help='Prediction model (default: ensemble)'
    )
    parser.add_argument(
        '--current-cpu',
        type=int,
        default=500,
        help='Current CPU request in millicores (default: 500)'
    )
    parser.add_argument(
        '--current-memory',
        type=int,
        default=256,
        help='Current memory request in Mi (default: 256)'
    )
    
    # Scaling policy arguments
    parser.add_argument(
        '--scale-up-cpu',
        type=float,
        default=80.0,
        help='Scale-up CPU threshold %% (default: 80.0)'
    )
    parser.add_argument(
        '--scale-down-cpu',
        type=float,
        default=30.0,
        help='Scale-down CPU threshold %% (default: 30.0)'
    )
    parser.add_argument(
        '--scale-up-memory',
        type=float,
        default=80.0,
        help='Scale-up memory threshold %% (default: 80.0)'
    )
    parser.add_argument(
        '--scale-down-memory',
        type=float,
        default=30.0,
        help='Scale-down memory threshold %% (default: 30.0)'
    )
    parser.add_argument(
        '--cpu-safety-margin',
        type=float,
        default=15.0,
        help='CPU safety margin %% (default: 15.0)'
    )
    parser.add_argument(
        '--memory-safety-margin',
        type=float,
        default=15.0,
        help='Memory safety margin %% (default: 15.0)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory with trained models'
    )
    
    args = parser.parse_args()
    
    # Create custom policy
    policy = ScalingPolicy(
        scale_up_cpu_threshold=args.scale_up_cpu,
        scale_down_cpu_threshold=args.scale_down_cpu,
        scale_up_memory_threshold=args.scale_up_memory,
        scale_down_memory_threshold=args.scale_down_memory,
        cpu_safety_margin=args.cpu_safety_margin,
        memory_safety_margin=args.memory_safety_margin
    )
    
    # Initialize CLI
    try:
        cli = RecommenderCLI(model_dir=args.model_dir, policy=policy)
    except Exception as e:
        print(f"Error initializing: {e}")
        sys.exit(1)
    
    # Run in interactive mode
    if args.interactive:
        cli.run_interactive()
        return
    
    # Load data
    if args.data:
        recent_data = cli.parse_inline_data(args.data)
    elif args.file:
        recent_data = cli.load_data_from_file(args.file)
    else:
        parser.print_help()
        sys.exit(1)
    
    if recent_data is None:
        print("Failed to load data")
        sys.exit(1)
    
    print(f"Input Data Shape: {recent_data.shape}")
    print(f"Latest CPU/Memory: {recent_data[-1]}")
    
    cli.print_recommendation(
        recent_data,
        model=args.model,
        current_cpu=args.current_cpu,
        current_memory=args.current_memory
    )


if __name__ == '__main__':
    main()
