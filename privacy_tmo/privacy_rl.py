"""
Privacy-Aware Reinforcement Learning for TMO

Extends the original TMO RL models with:
1. Privacy Risk term in reward function
2. Privacy Budget constraint
3. Sensitivity-aware state representation

Key modification to reward function:
Original: R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost
Privacy:  R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost 
            - β₄·PrivacyRisk + γ·BudgetBonus

subject to: Σₜ PrivacyRisk(qₜ, aₜ) ≤ ε (Privacy Budget)
"""

import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch.nn import functional as F

from .sensitivity_classifier import SensitivityClassifier, SensitivityLevel
from .privacy_manager import PrivacyManager, PrivacyBudget


@dataclass
class PrivacyRLConfig:
    """Configuration for Privacy-Aware RL"""
    # Privacy reward weights
    beta_privacy: float = 1.0      # Weight for privacy risk penalty
    gamma_budget: float = 0.5      # Weight for budget conservation bonus
    
    # Privacy budget
    privacy_budget: float = 1.0    # Total privacy budget (epsilon)
    budget_penalty: float = 10.0   # Penalty for exceeding budget
    
    # Sensitivity thresholds
    high_sensitivity_threshold: float = 0.7
    
    # State augmentation
    include_sensitivity_in_state: bool = True
    include_budget_in_state: bool = True
    
    # Constraint handling
    use_lagrangian: bool = True    # Use Lagrangian relaxation for constraint
    lagrange_multiplier: float = 1.0


class PrivacyAwareEnv(gym.Wrapper):
    """
    Wrapper that adds privacy-awareness to the M4A1 environment
    
    Modifications:
    1. Augments state with sensitivity and budget information
    2. Modifies reward to include privacy risk
    3. Tracks privacy budget consumption
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[PrivacyRLConfig] = None,
        privacy_manager: Optional[PrivacyManager] = None,
    ):
        super().__init__(env)
        
        self.config = config or PrivacyRLConfig()
        self.privacy_manager = privacy_manager or PrivacyManager(
            enable_ner=False,
            enable_ml=False,
        )
        
        # Privacy budget tracker
        self.budget = PrivacyBudget(epsilon=self.config.privacy_budget)
        
        # Extend observation space if needed
        if self.config.include_sensitivity_in_state or self.config.include_budget_in_state:
            self._extend_observation_space()
        
        # Statistics
        self.episode_privacy_risk = 0.0
        self.episode_budget_violations = 0
        
        # Current query (for sensitivity analysis)
        self.current_query = ""
        self.current_sensitivity = None
    
    def _extend_observation_space(self):
        """Extend observation space with privacy-related features"""
        original_space = self.observation_space
        
        # Calculate additional dimensions
        extra_dims = 0
        if self.config.include_sensitivity_in_state:
            extra_dims += 3  # sensitivity level (one-hot), sensitivity score, num_entities
        if self.config.include_budget_in_state:
            extra_dims += 2  # remaining budget ratio, consumed budget
        
        if extra_dims > 0:
            low = np.concatenate([
                original_space.low,
                np.zeros(extra_dims)
            ])
            high = np.concatenate([
                original_space.high,
                np.ones(extra_dims) * 10  # Reasonable upper bound
            ])
            
            self.observation_space = gym.spaces.Box(
                low=low.astype(np.float32),
                high=high.astype(np.float32),
                dtype=np.float32
            )
    
    def set_current_query(self, query: str):
        """Set current query for sensitivity analysis"""
        self.current_query = query
        self.current_sensitivity = self.privacy_manager.analyze_query(query)
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Augment observation with privacy features"""
        if not (self.config.include_sensitivity_in_state or 
                self.config.include_budget_in_state):
            return obs
        
        extra_features = []
        
        if self.config.include_sensitivity_in_state:
            if self.current_sensitivity:
                # One-hot sensitivity level
                level_onehot = [0, 0, 0]
                level_onehot[self.current_sensitivity.level] = 1
                extra_features.extend(level_onehot)
                
                # Sensitivity score
                extra_features.append(self.current_sensitivity.score)
                
                # Number of detected entities (normalized)
                num_entities = min(len(self.current_sensitivity.detected_entities) / 5, 1.0)
                extra_features.append(num_entities)
            else:
                extra_features.extend([1, 0, 0, 0, 0])  # Default: PUBLIC, score=0
        
        if self.config.include_budget_in_state:
            # Remaining budget ratio (0 to 1)
            extra_features.append(self.budget.remaining / self.budget.epsilon)
            # Consumed budget (normalized)
            extra_features.append(min(self.budget.consumed, 1.0))
        
        return np.concatenate([obs, np.array(extra_features, dtype=np.float32)])
    
    def reset(self, **kwargs):
        """Reset environment and privacy budget"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset privacy tracking for new episode
        self.budget.reset()
        self.episode_privacy_risk = 0.0
        self.episode_budget_violations = 0
        self.current_sensitivity = None
        
        return self._augment_observation(obs), info
    
    def step(self, action):
        """
        Take action with privacy-aware reward modification
        
        Reward modification:
        R' = R_original - β₄·PrivacyRisk(q, a) + γ·BudgetBonus - BudgetPenalty
        """
        # Execute original step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate privacy risk for this action
        offload_to_cloud = action > 0
        privacy_risk = self._calculate_privacy_risk(offload_to_cloud)
        
        # Track budget
        if offload_to_cloud and privacy_risk > 0:
            self.budget.consume(privacy_risk, f"step_{self.episode_privacy_risk}")
        
        self.episode_privacy_risk += privacy_risk
        
        # Modify reward
        modified_reward = self._modify_reward(
            original_reward=reward,
            privacy_risk=privacy_risk,
            action=action,
        )
        
        # Add privacy info to info dict
        info['privacy_risk'] = privacy_risk
        info['cumulative_privacy_risk'] = self.episode_privacy_risk
        info['remaining_budget'] = self.budget.remaining
        info['budget_violated'] = self.budget.remaining < 0
        
        if self.budget.remaining < 0:
            self.episode_budget_violations += 1
        
        # Augment observation
        augmented_obs = self._augment_observation(obs)
        
        return augmented_obs, modified_reward, terminated, truncated, info
    
    def _calculate_privacy_risk(self, offload_to_cloud: bool) -> float:
        """Calculate privacy risk for current action"""
        if not offload_to_cloud:
            return 0.0
        
        if self.current_sensitivity is None:
            return 0.0
        
        return self.privacy_manager.calculate_privacy_risk(
            self.current_sensitivity,
            offload_to_cloud=True
        )
    
    def _modify_reward(
        self,
        original_reward: float,
        privacy_risk: float,
        action: int,
    ) -> float:
        """
        Modify reward with privacy considerations
        
        R' = R - β₄·PrivacyRisk + γ·BudgetBonus - BudgetPenalty
        """
        modified = original_reward
        
        # Privacy risk penalty
        modified -= self.config.beta_privacy * privacy_risk
        
        # Budget conservation bonus (reward for staying within budget)
        if self.budget.remaining > 0:
            budget_bonus = self.config.gamma_budget * (self.budget.remaining / self.budget.epsilon)
            modified += budget_bonus
        
        # Budget violation penalty
        if self.budget.remaining < 0:
            modified -= self.config.budget_penalty
        
        # Bonus for choosing local when sensitive
        if action == 0 and self.current_sensitivity:
            if self.current_sensitivity.level >= SensitivityLevel.SEMI_SENSITIVE:
                modified += 0.5  # Bonus for correct privacy-preserving decision
        
        return modified


class PrivacyConstrainedPPO(OnPolicyAlgorithm):
    """
    PPO with Privacy Budget Constraint using Lagrangian Relaxation
    
    Extends RC_PPO with:
    1. Privacy risk term in loss function
    2. Lagrangian multiplier for budget constraint
    3. Adaptive multiplier update
    """
    
    def __init__(
        self,
        policy,
        env: GymEnv,
        privacy_config: Optional[PrivacyRLConfig] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        **kwargs
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            **kwargs
        )
        
        self.privacy_config = privacy_config or PrivacyRLConfig()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        
        # Lagrangian multiplier for privacy constraint
        self.lagrange_multiplier = self.privacy_config.lagrange_multiplier
        self.lagrange_lr = 0.01  # Learning rate for multiplier update
        
        # Track constraint violations
        self.constraint_violations = []
    
    def _compute_privacy_penalty(self, rollout_data) -> torch.Tensor:
        """
        Compute privacy constraint penalty
        
        Penalty = λ · max(0, Σ PrivacyRisk - ε)
        """
        # This would need access to privacy risks from the environment
        # For now, we estimate based on actions (0 = local, >0 = cloud)
        cloud_actions = (rollout_data.actions > 0).float()
        
        # Estimate privacy risk (higher for cloud actions)
        estimated_risk = cloud_actions.mean() * 0.5  # Simple estimate
        
        # Constraint violation
        violation = torch.relu(
            estimated_risk - self.privacy_config.privacy_budget / self.n_steps
        )
        
        return self.lagrange_multiplier * violation
    
    def _update_lagrange_multiplier(self, constraint_violation: float):
        """
        Update Lagrangian multiplier based on constraint violation
        
        λ' = max(0, λ + lr · violation)
        """
        self.lagrange_multiplier = max(
            0.0,
            self.lagrange_multiplier + self.lagrange_lr * constraint_violation
        )
        self.constraint_violations.append(constraint_violation)
    
    def train(self) -> None:
        """
        Train with privacy-aware loss
        
        Loss = Policy Loss + VF Loss + Entropy Loss + Privacy Penalty
        """
        self.policy.set_training_mode(True)
        
        # Standard PPO training loop
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                
                # Evaluate actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, 
                    actions
                )
                values = values.flatten()
                
                # Advantages
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss (PPO clipping)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = F.mse_loss(rollout_data.returns, values)
                
                # Entropy loss
                entropy_loss = -torch.mean(entropy) if entropy is not None else 0.0
                
                # Privacy constraint penalty (Lagrangian)
                privacy_penalty = self._compute_privacy_penalty(rollout_data)
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.ent_coef * entropy_loss 
                    + self.vf_coef * value_loss
                    + privacy_penalty
                )
                
                # Optimize
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.max_grad_norm
                )
                self.policy.optimizer.step()
        
        # Update Lagrange multiplier after epoch
        if self.constraint_violations:
            avg_violation = np.mean(self.constraint_violations[-100:])
            self._update_lagrange_multiplier(avg_violation)


def create_privacy_reward_function(
    base_weights: Dict[str, float],
    privacy_config: PrivacyRLConfig,
    privacy_manager: PrivacyManager,
):
    """
    Factory function to create privacy-aware reward function
    
    Returns a function that can be used to calculate rewards.
    
    Args:
        base_weights: Original TMO weights {α, β₁, β₂, β₃}
        privacy_config: Privacy-specific configuration
        privacy_manager: Privacy manager for sensitivity analysis
    
    Returns:
        reward_fn(state, action, query) -> float
    """
    def reward_fn(
        response_score: float,
        association_score: float,
        latency_cost: float,
        usage_cost: float,
        query: str,
        action: int,
        budget: PrivacyBudget,
    ) -> Tuple[float, Dict]:
        """
        Calculate privacy-aware reward
        
        R = α·Quality + β₁·Association - β₂·Latency - β₃·Cost 
            - β₄·PrivacyRisk + γ·BudgetBonus
        """
        # Base reward (original TMO)
        reward = (
            base_weights.get('alpha', 1.0) * response_score
            + base_weights.get('beta_association', 0.33) * association_score
            - base_weights.get('beta_latency', 0.33) * latency_cost
            - base_weights.get('beta_usage', 0.33) * usage_cost
        )
        
        # Privacy risk
        sensitivity = privacy_manager.analyze_query(query)
        privacy_risk = privacy_manager.calculate_privacy_risk(
            sensitivity, 
            offload_to_cloud=(action > 0)
        )
        
        reward -= privacy_config.beta_privacy * privacy_risk
        
        # Budget bonus
        if budget.remaining > 0:
            budget_ratio = budget.remaining / budget.epsilon
            reward += privacy_config.gamma_budget * budget_ratio
        else:
            # Budget violation penalty
            reward -= privacy_config.budget_penalty
        
        # Info dict
        info = {
            'base_reward': reward + privacy_config.beta_privacy * privacy_risk,
            'privacy_risk': privacy_risk,
            'privacy_penalty': privacy_config.beta_privacy * privacy_risk,
            'budget_remaining': budget.remaining,
            'sensitivity_level': sensitivity.level.name,
        }
        
        return reward, info
    
    return reward_fn


# Extension to existing RC models
def add_privacy_constraint_to_loss(
    original_loss: torch.Tensor,
    rollout_data,
    privacy_lambda: float,
    env,
) -> torch.Tensor:
    """
    Add privacy constraint to existing RC model loss
    
    Can be used to extend RC_PPO, RC_A2C, RC_DQN
    
    Args:
        original_loss: Loss from original resource constraint
        rollout_data: Rollout data from buffer
        privacy_lambda: Lagrangian multiplier for privacy
        env: Environment with privacy tracking
        
    Returns:
        Modified loss with privacy constraint
    """
    # Estimate privacy risk from actions
    cloud_actions = (rollout_data.actions > 0).float()
    
    # Higher penalty for cloud actions on sensitive queries
    # (Would need sensitivity info from env)
    privacy_penalty = privacy_lambda * cloud_actions.mean()
    
    return original_loss + privacy_penalty


if __name__ == "__main__":
    print("🧪 Testing Privacy-Aware RL Components\n")
    
    # Test PrivacyRLConfig
    config = PrivacyRLConfig()
    print(f"Privacy Config:")
    print(f"  beta_privacy: {config.beta_privacy}")
    print(f"  privacy_budget: {config.privacy_budget}")
    
    # Test reward function
    base_weights = {
        'alpha': 1.0,
        'beta_association': 0.33,
        'beta_latency': 0.33,
        'beta_usage': 0.33,
    }
    
    privacy_manager = PrivacyManager(enable_ner=False, enable_ml=False)
    reward_fn = create_privacy_reward_function(
        base_weights, 
        config, 
        privacy_manager
    )
    
    # Test cases
    test_cases = [
        ("What is Python?", 1, "Public query to cloud"),
        ("My password is secret", 0, "Private query to local"),
        ("My password is secret", 1, "Private query to cloud (bad!)"),
    ]
    
    budget = PrivacyBudget(epsilon=1.0)
    
    print("\n📊 Reward Function Tests:")
    print("-" * 60)
    
    for query, action, description in test_cases:
        reward, info = reward_fn(
            response_score=0.8,
            association_score=0.5,
            latency_cost=0.3,
            usage_cost=0.1,
            query=query,
            action=action,
            budget=budget,
        )
        
        print(f"\n{description}")
        print(f"  Query: {query[:30]}...")
        print(f"  Action: {'Cloud' if action > 0 else 'Local'}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Privacy Risk: {info['privacy_risk']:.3f}")
        print(f"  Sensitivity: {info['sensitivity_level']}")
        
        # Consume budget if cloud
        if action > 0:
            budget.consume(info['privacy_risk'], "test")
    
    print("\n" + "-" * 60)
    print(f"\nFinal Budget: {budget.remaining:.3f} / {budget.epsilon}")
    print("\n✅ Privacy-Aware RL ready!")
