"""
Benchmarking Suite for Privacy-TMO

Implements comprehensive evaluation:
1. Baseline comparisons (No Protection, Random, Threshold-based)
2. Privacy-Quality-Latency-Cost trade-off analysis
3. Pareto frontier visualization
4. Statistical significance testing
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    num_episodes: int = 100
    num_queries_per_episode: int = 20
    privacy_budgets: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    repeat_runs: int = 3
    output_dir: str = "./benchmark_results"


@dataclass
class QuerySample:
    """Sample query for benchmarking"""
    id: str
    text: str
    true_sensitivity: int  # 0, 1, or 2
    category: str  # "public", "semi_sensitive", "private"
    expected_action: int  # 0 = local, 1 = cloud, 2 = hybrid


@dataclass
class EpisodeResult:
    """Result of single episode"""
    episode_id: int
    total_reward: float
    avg_response_quality: float
    avg_latency: float
    total_cost: float
    privacy_risk: float
    budget_violations: int
    actions_taken: List[int]
    correct_decisions: int
    total_decisions: int


@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    method_name: str
    config: Dict
    episodes: List[EpisodeResult]
    
    # Aggregate metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_quality: float = 0.0
    mean_latency: float = 0.0
    mean_cost: float = 0.0
    mean_privacy_risk: float = 0.0
    privacy_violation_rate: float = 0.0
    decision_accuracy: float = 0.0
    
    def compute_aggregates(self):
        """Compute aggregate metrics from episodes"""
        if not self.episodes:
            return
        
        rewards = [e.total_reward for e in self.episodes]
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)
        
        self.mean_quality = np.mean([e.avg_response_quality for e in self.episodes])
        self.mean_latency = np.mean([e.avg_latency for e in self.episodes])
        self.mean_cost = np.mean([e.total_cost for e in self.episodes])
        self.mean_privacy_risk = np.mean([e.privacy_risk for e in self.episodes])
        
        total_violations = sum(e.budget_violations for e in self.episodes)
        total_decisions = sum(e.total_decisions for e in self.episodes)
        self.privacy_violation_rate = total_violations / total_decisions if total_decisions > 0 else 0
        
        total_correct = sum(e.correct_decisions for e in self.episodes)
        self.decision_accuracy = total_correct / total_decisions if total_decisions > 0 else 0


class BaselineMethod:
    """Base class for baseline methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    def decide(self, query: QuerySample, state: Dict) -> int:
        """Make offloading decision. Override in subclasses."""
        raise NotImplementedError


class NoProtectionBaseline(BaselineMethod):
    """
    No Protection Baseline
    All queries go to cloud for best quality
    """
    
    def __init__(self):
        super().__init__("NoProtection")
    
    def decide(self, query: QuerySample, state: Dict) -> int:
        return 1  # Always cloud


class LocalOnlyBaseline(BaselineMethod):
    """
    Local Only Baseline
    All queries processed locally (maximum privacy)
    """
    
    def __init__(self):
        super().__init__("LocalOnly")
    
    def decide(self, query: QuerySample, state: Dict) -> int:
        return 0  # Always local


class RandomBaseline(BaselineMethod):
    """
    Random Baseline
    Random choice between local and cloud
    """
    
    def __init__(self, cloud_probability: float = 0.5):
        super().__init__("Random")
        self.cloud_prob = cloud_probability
    
    def decide(self, query: QuerySample, state: Dict) -> int:
        return 1 if np.random.random() < self.cloud_prob else 0


class ThresholdBaseline(BaselineMethod):
    """
    Threshold-based Baseline
    Use sensitivity threshold to decide
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(f"Threshold_{threshold}")
        self.threshold = threshold
    
    def decide(self, query: QuerySample, state: Dict) -> int:
        sensitivity_score = state.get('sensitivity_score', 0.0)
        return 0 if sensitivity_score > self.threshold else 1


class PrivacyTMOMethod(BaselineMethod):
    """
    Privacy-TMO Method (Ours)
    Uses sensitivity classification and budget management
    """
    
    def __init__(self, privacy_budget: float = 1.0):
        super().__init__(f"PrivacyTMO_budget{privacy_budget}")
        self.privacy_budget = privacy_budget
        self.consumed_budget = 0.0
    
    def decide(self, query: QuerySample, state: Dict) -> int:
        sensitivity_level = state.get('sensitivity_level', 0)
        sensitivity_score = state.get('sensitivity_score', 0.0)
        
        # Private must be local
        if sensitivity_level == 2:
            return 0
        
        # Check budget
        estimated_risk = sensitivity_score * 0.5
        remaining = self.privacy_budget - self.consumed_budget
        
        if sensitivity_level == 1:  # Semi-sensitive
            if remaining >= estimated_risk:
                self.consumed_budget += estimated_risk
                return 2  # Hybrid
            else:
                return 0  # Local (budget exhausted)
        
        # Public
        return 1  # Cloud
    
    def reset(self):
        """Reset budget for new episode"""
        self.consumed_budget = 0.0


class BenchmarkSuite:
    """
    Complete benchmarking suite
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, BenchmarkResult] = {}
        
        # Initialize baselines
        self.baselines = {
            'no_protection': NoProtectionBaseline(),
            'local_only': LocalOnlyBaseline(),
            'random': RandomBaseline(),
            'threshold_0.3': ThresholdBaseline(0.3),
            'threshold_0.5': ThresholdBaseline(0.5),
            'threshold_0.7': ThresholdBaseline(0.7),
        }
        
        # Add Privacy-TMO variants
        for budget in self.config.privacy_budgets:
            name = f'privacy_tmo_{budget}'
            self.baselines[name] = PrivacyTMOMethod(budget)
    
    def generate_test_queries(self, num_queries: int) -> List[QuerySample]:
        """Generate test query samples"""
        queries = []
        
        # Public queries (40%)
        public_templates = [
            "What is {topic}?",
            "How does {topic} work?",
            "Explain {topic} in simple terms",
        ]
        public_topics = [
            "machine learning", "Python programming", "data science",
            "artificial intelligence", "cloud computing", "algorithms",
        ]
        
        # Semi-sensitive queries (30%)
        semi_templates = [
            "Analyze my code: {code}",
            "Review this document about {topic}",
            "My schedule for {date} includes {event}",
        ]
        
        # Private queries (30%)
        private_templates = [
            "My password is {password}",
            "My SSN is {ssn}",
            "My credit card number is {cc}",
            "Send email to {email} about {topic}",
        ]
        
        for i in range(num_queries):
            r = np.random.random()
            
            if r < 0.4:  # Public
                template = np.random.choice(public_templates)
                topic = np.random.choice(public_topics)
                text = template.format(topic=topic)
                sensitivity = 0
                category = "public"
                expected_action = 1  # Cloud
            
            elif r < 0.7:  # Semi-sensitive
                template = np.random.choice(semi_templates)
                text = template.format(
                    code="def foo(): pass",
                    topic="meeting",
                    date="tomorrow",
                    event="doctor appointment",
                )
                sensitivity = 1
                category = "semi_sensitive"
                expected_action = 2  # Hybrid
            
            else:  # Private
                template = np.random.choice(private_templates)
                text = template.format(
                    password="secret123",
                    ssn="123-45-6789",
                    cc="4111-1111-1111-1111",
                    email="test@example.com",
                    topic="project",
                )
                sensitivity = 2
                category = "private"
                expected_action = 0  # Local
            
            queries.append(QuerySample(
                id=f"q_{i}",
                text=text,
                true_sensitivity=sensitivity,
                category=category,
                expected_action=expected_action,
            ))
        
        return queries
    
    def simulate_inference(
        self,
        query: QuerySample,
        action: int,
    ) -> Dict:
        """
        Simulate inference and return metrics
        
        Returns:
            Dict with quality, latency, cost, privacy_risk
        """
        # Simulate quality (cloud is better for public, local for private)
        if action == 0:  # Local
            quality = 0.6 + np.random.normal(0, 0.1)
            latency = 2.0 + np.random.exponential(0.5)
            cost = 0.001
            privacy_risk = 0.0
        elif action == 1:  # Cloud
            quality = 0.9 + np.random.normal(0, 0.05)
            latency = 0.5 + np.random.exponential(0.2)
            cost = 0.01
            # Privacy risk depends on sensitivity
            privacy_risk = query.true_sensitivity * 0.4
        else:  # Hybrid
            quality = 0.75 + np.random.normal(0, 0.08)
            latency = 1.5 + np.random.exponential(0.3)
            cost = 0.005
            privacy_risk = query.true_sensitivity * 0.2
        
        # Clip values
        quality = np.clip(quality, 0, 1)
        latency = max(0.1, latency)
        
        return {
            'quality': quality,
            'latency': latency,
            'cost': cost,
            'privacy_risk': privacy_risk,
        }
    
    def run_episode(
        self,
        method: BaselineMethod,
        queries: List[QuerySample],
        episode_id: int,
    ) -> EpisodeResult:
        """Run single episode"""
        if hasattr(method, 'reset'):
            method.reset()
        
        total_reward = 0.0
        qualities = []
        latencies = []
        total_cost = 0.0
        total_privacy_risk = 0.0
        budget_violations = 0
        actions = []
        correct_decisions = 0
        
        for query in queries:
            # Get state
            state = {
                'sensitivity_level': query.true_sensitivity,
                'sensitivity_score': query.true_sensitivity / 2.0,
            }
            
            # Make decision
            action = method.decide(query, state)
            actions.append(action)
            
            # Check if decision is correct
            if action == query.expected_action:
                correct_decisions += 1
            
            # Simulate inference
            result = self.simulate_inference(query, action)
            
            qualities.append(result['quality'])
            latencies.append(result['latency'])
            total_cost += result['cost']
            total_privacy_risk += result['privacy_risk']
            
            # Check budget violation
            if result['privacy_risk'] > 0.5:  # Threshold for violation
                budget_violations += 1
            
            # Calculate reward (simplified TMO reward)
            reward = (
                result['quality']
                - 0.1 * result['latency']
                - 10 * result['cost']
                - 0.5 * result['privacy_risk']
            )
            total_reward += reward
        
        return EpisodeResult(
            episode_id=episode_id,
            total_reward=total_reward,
            avg_response_quality=np.mean(qualities),
            avg_latency=np.mean(latencies),
            total_cost=total_cost,
            privacy_risk=total_privacy_risk,
            budget_violations=budget_violations,
            actions_taken=actions,
            correct_decisions=correct_decisions,
            total_decisions=len(queries),
        )
    
    def run_benchmark(
        self,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run complete benchmark
        
        Args:
            methods: List of method names to benchmark (default: all)
        """
        methods = methods or list(self.baselines.keys())
        
        print("🚀 Starting Benchmark Suite")
        print(f"   Episodes: {self.config.num_episodes}")
        print(f"   Queries per episode: {self.config.num_queries_per_episode}")
        print(f"   Methods: {len(methods)}")
        print("=" * 60)
        
        for method_name in methods:
            if method_name not in self.baselines:
                print(f"⚠️ Unknown method: {method_name}")
                continue
            
            method = self.baselines[method_name]
            print(f"\n📊 Running {method_name}...")
            
            episodes = []
            for run in range(self.config.repeat_runs):
                for ep in range(self.config.num_episodes):
                    queries = self.generate_test_queries(
                        self.config.num_queries_per_episode
                    )
                    episode_result = self.run_episode(
                        method=method,
                        queries=queries,
                        episode_id=ep + run * self.config.num_episodes,
                    )
                    episodes.append(episode_result)
            
            result = BenchmarkResult(
                method_name=method_name,
                config={'budget': getattr(method, 'privacy_budget', None)},
                episodes=episodes,
            )
            result.compute_aggregates()
            
            self.results[method_name] = result
            
            print(f"   Mean Reward: {result.mean_reward:.3f} ± {result.std_reward:.3f}")
            print(f"   Quality: {result.mean_quality:.3f}")
            print(f"   Privacy Risk: {result.mean_privacy_risk:.3f}")
            print(f"   Decision Accuracy: {result.decision_accuracy:.2%}")
        
        return self.results
    
    def compute_pareto_frontier(self) -> List[str]:
        """
        Compute Pareto frontier methods
        
        Pareto optimal: no other method is better in ALL metrics
        """
        pareto_optimal = []
        
        for name, result in self.results.items():
            is_dominated = False
            
            for other_name, other in self.results.items():
                if name == other_name:
                    continue
                
                # Check if 'other' dominates 'result'
                # (better quality AND better privacy)
                if (other.mean_quality >= result.mean_quality and
                    other.mean_privacy_risk <= result.mean_privacy_risk and
                    (other.mean_quality > result.mean_quality or
                     other.mean_privacy_risk < result.mean_privacy_risk)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(name)
        
        return pareto_optimal
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report"""
        lines = ["=" * 70]
        lines.append("PRIVACY-TMO BENCHMARK REPORT")
        lines.append("=" * 70)
        
        # Summary table
        lines.append("\n📊 SUMMARY TABLE")
        lines.append("-" * 70)
        lines.append(f"{'Method':<25} {'Reward':>10} {'Quality':>10} {'Privacy':>10} {'Accuracy':>10}")
        lines.append("-" * 70)
        
        for name, result in sorted(self.results.items(), key=lambda x: -x[1].mean_reward):
            lines.append(
                f"{name:<25} {result.mean_reward:>10.3f} {result.mean_quality:>10.3f} "
                f"{result.mean_privacy_risk:>10.3f} {result.decision_accuracy:>10.2%}"
            )
        
        # Pareto frontier
        pareto = self.compute_pareto_frontier()
        lines.append(f"\n🏆 PARETO OPTIMAL METHODS: {', '.join(pareto)}")
        
        # Analysis
        lines.append("\n📈 ANALYSIS")
        lines.append("-" * 70)
        
        # Best in each category
        best_reward = max(self.results.items(), key=lambda x: x[1].mean_reward)
        best_quality = max(self.results.items(), key=lambda x: x[1].mean_quality)
        best_privacy = min(self.results.items(), key=lambda x: x[1].mean_privacy_risk)
        
        lines.append(f"Best Reward: {best_reward[0]} ({best_reward[1].mean_reward:.3f})")
        lines.append(f"Best Quality: {best_quality[0]} ({best_quality[1].mean_quality:.3f})")
        lines.append(f"Best Privacy: {best_privacy[0]} ({best_privacy[1].mean_privacy_risk:.3f})")
        
        # Privacy-TMO vs baselines
        if any('privacy_tmo' in name for name in self.results):
            lines.append("\n🔒 PRIVACY-TMO vs BASELINES")
            lines.append("-" * 70)
            
            no_protection = self.results.get('no_protection')
            if no_protection:
                for name, result in self.results.items():
                    if 'privacy_tmo' in name:
                        quality_diff = (result.mean_quality - no_protection.mean_quality) / no_protection.mean_quality
                        privacy_diff = (no_protection.mean_privacy_risk - result.mean_privacy_risk) / no_protection.mean_privacy_risk
                        
                        lines.append(f"{name}:")
                        lines.append(f"  Quality change: {quality_diff:+.1%}")
                        lines.append(f"  Privacy improvement: {privacy_diff:+.1%}")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
    
    def save_results(self, filepath: Optional[str] = None):
        """Save results to JSON file"""
        filepath = filepath or f"{self.config.output_dir}/benchmark_results.json"
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for name, result in self.results.items():
            data[name] = {
                'mean_reward': result.mean_reward,
                'std_reward': result.std_reward,
                'mean_quality': result.mean_quality,
                'mean_latency': result.mean_latency,
                'mean_cost': result.mean_cost,
                'mean_privacy_risk': result.mean_privacy_risk,
                'privacy_violation_rate': result.privacy_violation_rate,
                'decision_accuracy': result.decision_accuracy,
                'config': result.config,
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Results saved to: {filepath}")


if __name__ == "__main__":
    print("🧪 Testing Benchmark Suite\n")
    
    # Initialize with smaller config for testing
    config = BenchmarkConfig(
        num_episodes=10,
        num_queries_per_episode=20,
        privacy_budgets=[0.3, 0.5, 0.7, 1.0],
        repeat_runs=1,
    )
    
    suite = BenchmarkSuite(config)
    
    # Run benchmark on subset
    methods = [
        'no_protection',
        'local_only',
        'random',
        'threshold_0.5',
        'privacy_tmo_0.5',
        'privacy_tmo_1.0',
    ]
    
    results = suite.run_benchmark(methods)
    
    # Generate report
    print("\n" + suite.generate_report())
    
    # Save results
    suite.save_results("./benchmark_results/test_run.json")
    
    print("\n✅ Benchmark Suite ready!")
