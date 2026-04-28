"""
Resource Recommendation Engine

Generates scaling decisions and resource recommendations (CPU/Memory) 
based on predicted resource usage patterns.
"""

import os
import sys
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.predictor import ResourcePredictor

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Scaling decision actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class UtilizationLevel(Enum):
    """Utilization level categories."""
    CRITICAL = "critical"      # > 90%
    HIGH = "high"              # 70-90%
    MODERATE = "moderate"      # 40-70%
    LOW = "low"                # 20-40%
    IDLE = "idle"              # < 20%


@dataclass
class ResourceRecommendation:
    """Resource recommendation."""
    cpu_request_millicores: int      # e.g., 500
    cpu_limit_millicores: int        # e.g., 1000
    memory_request_mi: int            # e.g., 256
    memory_limit_mi: int              # e.g., 512
    
    def to_kubernetes_yaml(self) -> str:
        """Convert to Kubernetes YAML format."""
        return f"""resources:
  requests:
    cpu: {self.cpu_request_millicores}m
    memory: {self.memory_request_mi}Mi
  limits:
    cpu: {self.cpu_limit_millicores}m
    memory: {self.memory_limit_mi}Mi"""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: ScalingAction
    current_cpu_predicted: float      # Predicted CPU %
    current_memory_predicted: float   # Predicted memory %
    cpu_utilization_level: UtilizationLevel
    memory_utilization_level: UtilizationLevel
    reason: str
    recommendation: ResourceRecommendation
    confidence_score: float           # 0-1, based on prediction confidence
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'action': self.action.value,
            'current_cpu_predicted': round(self.current_cpu_predicted, 2),
            'current_memory_predicted': round(self.current_memory_predicted, 2),
            'cpu_utilization_level': self.cpu_utilization_level.value,
            'memory_utilization_level': self.memory_utilization_level.value,
            'reason': self.reason,
            'recommendation': self.recommendation.to_dict(),
            'confidence_score': round(self.confidence_score, 3)
        }


class ScalingPolicy:
    """Scaling policy configuration."""
    
    def __init__(
        self,
        # Scale-up thresholds
        scale_up_cpu_threshold: float = 80.0,
        scale_up_memory_threshold: float = 80.0,
        scale_up_factor: float = 1.5,  # Multiply current by this
        
        # Scale-down thresholds
        scale_down_cpu_threshold: float = 30.0,
        scale_down_memory_threshold: float = 30.0,
        scale_down_factor: float = 0.7,  # Multiply current by this
        
        # Safety margins (% buffer above prediction)
        cpu_safety_margin: float = 15.0,   # Add 15% buffer
        memory_safety_margin: float = 15.0,  # Add 15% buffer
        
        # Resource constraints
        min_cpu_millicores: int = 100,
        max_cpu_millicores: int = 4000,
        min_memory_mi: int = 128,
        max_memory_mi: int = 8192,
    ):
        self.scale_up_cpu_threshold = scale_up_cpu_threshold
        self.scale_up_memory_threshold = scale_up_memory_threshold
        self.scale_up_factor = scale_up_factor
        
        self.scale_down_cpu_threshold = scale_down_cpu_threshold
        self.scale_down_memory_threshold = scale_down_memory_threshold
        self.scale_down_factor = scale_down_factor
        
        self.cpu_safety_margin = cpu_safety_margin
        self.memory_safety_margin = memory_safety_margin
        
        self.min_cpu_millicores = min_cpu_millicores
        self.max_cpu_millicores = max_cpu_millicores
        self.min_memory_mi = min_memory_mi
        self.max_memory_mi = max_memory_mi


class ResourceRecommender:
    """
    Resource Recommendation Engine
    
    Generates scaling decisions and resource recommendations based on
    predicted resource usage patterns.
    """
    
    def __init__(
        self,
        model_dir: str = 'models',
        window_size: int = 10,
        policy: Optional[ScalingPolicy] = None
    ):
        """
        Initialize the recommender.
        
        Args:
            model_dir (str): Directory with trained models
            window_size (int): Window size for predictions
            policy (ScalingPolicy): Scaling policy (default provided)
        """
        self.predictor = ResourcePredictor(model_dir=model_dir, window_size=window_size)
        self.policy = policy or ScalingPolicy()
        logger.info("Resource Recommender initialized")
    
    @staticmethod
    def _get_utilization_level(usage_percent: float) -> UtilizationLevel:
        """Determine utilization level from percentage."""
        if usage_percent > 90:
            return UtilizationLevel.CRITICAL
        elif usage_percent > 70:
            return UtilizationLevel.HIGH
        elif usage_percent > 40:
            return UtilizationLevel.MODERATE
        elif usage_percent > 20:
            return UtilizationLevel.LOW
        else:
            return UtilizationLevel.IDLE
    
    def _calculate_recommendation(
        self,
        current_cpu_predicted: float,
        current_memory_predicted: float
    ) -> ResourceRecommendation:
        """
        Calculate resource recommendation based on predictions.
        
        Args:
            current_cpu_predicted (float): Predicted CPU usage %
            current_memory_predicted (float): Predicted memory usage %
        
        Returns:
            ResourceRecommendation: Recommended resource limits/requests
        """
        # Apply safety margin to predictions
        cpu_with_margin = current_cpu_predicted + self.policy.cpu_safety_margin
        memory_with_margin = current_memory_predicted + self.policy.memory_safety_margin
        
        # Ensure percentages don't exceed 100%
        cpu_with_margin = min(cpu_with_margin, 100.0)
        memory_with_margin = min(memory_with_margin, 100.0)
        
        # Calculate resources based on typical scaling patterns
        # Assuming: Request = 60% of limit, Limit = 100% of estimated need
        
        # CPU: Convert percentage to millicores
        # Typical container: 500m request, 1000m limit (50%)
        cpu_limit = int(2000 * (cpu_with_margin / 100.0))  # Max 2000m
        cpu_request = int(cpu_limit * 0.6)  # Request is 60% of limit
        
        # Memory: Convert percentage to Mi
        # Typical container: 256Mi request, 512Mi limit (50%)
        memory_limit = int(1024 * (memory_with_margin / 100.0))  # Max 1024Mi
        memory_request = int(memory_limit * 0.6)  # Request is 60% of limit
        
        # Apply constraints
        cpu_request = max(self.policy.min_cpu_millicores, 
                         min(self.policy.max_cpu_millicores, cpu_request))
        cpu_limit = max(self.policy.min_cpu_millicores, 
                       min(self.policy.max_cpu_millicores, cpu_limit))
        memory_request = max(self.policy.min_memory_mi, 
                            min(self.policy.max_memory_mi, memory_request))
        memory_limit = max(self.policy.min_memory_mi, 
                          min(self.policy.max_memory_mi, memory_limit))
        
        # Ensure limit >= request
        cpu_limit = max(cpu_limit, cpu_request)
        memory_limit = max(memory_limit, memory_request)
        
        return ResourceRecommendation(
            cpu_request_millicores=cpu_request,
            cpu_limit_millicores=cpu_limit,
            memory_request_mi=memory_request,
            memory_limit_mi=memory_limit
        )
    
    def _determine_scaling_action(
        self,
        cpu_predicted: float,
        memory_predicted: float
    ) -> Tuple[ScalingAction, str]:
        """
        Determine scaling action based on predictions.
        
        Args:
            cpu_predicted (float): Predicted CPU usage %
            memory_predicted (float): Predicted memory usage %
        
        Returns:
            Tuple[ScalingAction, str]: Action and reasoning
        """
        # Check scale-up conditions
        if cpu_predicted > self.policy.scale_up_cpu_threshold or \
           memory_predicted > self.policy.scale_up_memory_threshold:
            
            reasons = []
            if cpu_predicted > self.policy.scale_up_cpu_threshold:
                reasons.append(f"CPU prediction {cpu_predicted:.1f}% > {self.policy.scale_up_cpu_threshold}%")
            if memory_predicted > self.policy.scale_up_memory_threshold:
                reasons.append(f"Memory prediction {memory_predicted:.1f}% > {self.policy.scale_up_memory_threshold}%")
            
            reason = f"Scale UP: {', '.join(reasons)}"
            return ScalingAction.SCALE_UP, reason
        
        # Check scale-down conditions
        if cpu_predicted < self.policy.scale_down_cpu_threshold and \
           memory_predicted < self.policy.scale_down_memory_threshold:
            
            reasons = []
            if cpu_predicted < self.policy.scale_down_cpu_threshold:
                reasons.append(f"CPU prediction {cpu_predicted:.1f}% < {self.policy.scale_down_cpu_threshold}%")
            if memory_predicted < self.policy.scale_down_memory_threshold:
                reasons.append(f"Memory prediction {memory_predicted:.1f}% < {self.policy.scale_down_memory_threshold}%")
            
            reason = f"Scale DOWN: {', '.join(reasons)}"
            return ScalingAction.SCALE_DOWN, reason
        
        # Otherwise maintain
        return ScalingAction.MAINTAIN, "Usage within acceptable range"
    
    def recommend(
        self,
        recent_data: np.ndarray,
        use_model: str = 'ensemble'
    ) -> ScalingDecision:
        """
        Generate scaling recommendation based on recent data.
        
        Args:
            recent_data (np.ndarray): Recent data (N_timesteps, 2)
            use_model (str): Model to use ('lstm', 'baseline', 'ensemble')
        
        Returns:
            ScalingDecision: Scaling decision with recommendation
        """
        # Get predictions with confidence
        if use_model == 'ensemble':
            prediction = self.predictor.predict_with_confidence(recent_data)
            confidence = 1.0 - (prediction.get('cpu_std', 0) + prediction.get('memory_std', 0)) / 200.0
            confidence = max(0.0, min(1.0, confidence))
        elif use_model == 'lstm':
            prediction = self.predictor.predict_lstm(recent_data)
            confidence = 0.9  # LSTM is typically high confidence
        else:  # baseline
            prediction = self.predictor.predict_baseline(recent_data)
            confidence = 0.7  # Baseline less confident
        
        cpu_predicted = prediction['cpu']
        memory_predicted = prediction['memory']
        
        # Calculate utilization levels
        cpu_level = self._get_utilization_level(cpu_predicted)
        memory_level = self._get_utilization_level(memory_predicted)
        
        # Determine scaling action
        action, reason = self._determine_scaling_action(cpu_predicted, memory_predicted)
        
        # Calculate resource recommendation
        recommendation = self._calculate_recommendation(cpu_predicted, memory_predicted)
        
        return ScalingDecision(
            action=action,
            current_cpu_predicted=cpu_predicted,
            current_memory_predicted=memory_predicted,
            cpu_utilization_level=cpu_level,
            memory_utilization_level=memory_level,
            reason=reason,
            recommendation=recommendation,
            confidence_score=confidence
        )
    
    def recommend_batch(
        self,
        data_sequences: List[np.ndarray],
        use_model: str = 'ensemble'
    ) -> List[ScalingDecision]:
        """
        Generate recommendations for multiple data sequences.
        
        Args:
            data_sequences (List[np.ndarray]): List of data arrays
            use_model (str): Model to use
        
        Returns:
            List[ScalingDecision]: List of scaling decisions
        """
        return [self.recommend(data, use_model) for data in data_sequences]
    
    def generate_scaling_plan(
        self,
        current_cpu_millicores: int,
        current_memory_mi: int,
        decision: ScalingDecision
    ) -> Dict:
        """
        Generate a detailed scaling plan from current to recommended resources.
        
        Args:
            current_cpu_millicores (int): Current CPU request in millicores
            current_memory_mi (int): Current memory request in Mi
            decision (ScalingDecision): Scaling decision
        
        Returns:
            Dict: Detailed scaling plan
        """
        rec = decision.recommendation
        
        cpu_change = rec.cpu_request_millicores - current_cpu_millicores
        memory_change = rec.memory_request_mi - current_memory_mi
        
        cpu_percent_change = (cpu_change / current_cpu_millicores * 100) if current_cpu_millicores > 0 else 0
        memory_percent_change = (memory_change / current_memory_mi * 100) if current_memory_mi > 0 else 0
        
        return {
            'scaling_action': decision.action.value,
            'current_resources': {
                'cpu_millicores': current_cpu_millicores,
                'memory_mi': current_memory_mi
            },
            'recommended_resources': {
                'cpu_request_millicores': rec.cpu_request_millicores,
                'cpu_limit_millicores': rec.cpu_limit_millicores,
                'memory_request_mi': rec.memory_request_mi,
                'memory_limit_mi': rec.memory_limit_mi
            },
            'changes': {
                'cpu_millicores_change': cpu_change,
                'cpu_percent_change': round(cpu_percent_change, 2),
                'memory_mi_change': memory_change,
                'memory_percent_change': round(memory_percent_change, 2)
            },
            'reason': decision.reason,
            'confidence_score': decision.confidence_score
        }


if __name__ == "__main__":
    # Example usage
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize recommender
    recommender = ResourceRecommender(model_dir='models', window_size=10)
    
    # Simulate recent data
    recent_data = np.array([
        [25.0, 40.0], [26.5, 41.2], [28.0, 42.5], [30.0, 44.0], [31.5, 45.5],
        [33.0, 47.0], [34.5, 48.5], [36.0, 50.0], [37.5, 51.5], [39.0, 53.0]
    ])
    
    print("\n" + "="*70)
    print("RESOURCE RECOMMENDATION ENGINE - EXAMPLE")
    print("="*70)
    
    # Get recommendation
    decision = recommender.recommend(recent_data, use_model='ensemble')
    
    print(f"\n[PREDICTIONS]")
    print(f"  CPU:    {decision.current_cpu_predicted:.2f}% ({decision.cpu_utilization_level.value})")
    print(f"  Memory: {decision.current_memory_predicted:.2f}% ({decision.memory_utilization_level.value})")
    
    print(f"\n[SCALING DECISION]")
    print(f"  Action: {decision.action.value.upper()}")
    print(f"  Reason: {decision.reason}")
    print(f"  Confidence: {decision.confidence_score:.1%}")
    
    print(f"\n[RESOURCE RECOMMENDATION]")
    rec = decision.recommendation
    print(f"  CPU Request:  {rec.cpu_request_millicores}m")
    print(f"  CPU Limit:    {rec.cpu_limit_millicores}m")
    print(f"  Memory Request: {rec.memory_request_mi}Mi")
    print(f"  Memory Limit:   {rec.memory_limit_mi}Mi")
    
    print(f"\n[KUBERNETES YAML]")
    print(rec.to_kubernetes_yaml())
    
    print(f"\n[SCALING PLAN (from 500m/256Mi)]")
    plan = recommender.generate_scaling_plan(500, 256, decision)
    print(json.dumps(plan, indent=2))
