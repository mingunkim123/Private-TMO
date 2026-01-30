"""
Privacy Manager for Privacy-TMO

Central component that orchestrates:
1. Sensitivity classification
2. Privacy budget management
3. Offloading decisions
4. Privacy risk calculation

This is the main interface for privacy-aware inference.
"""

import time
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

from .config import PrivacyTMOConfig, PrivacyConfig
from .sensitivity_classifier import SensitivityClassifier, SensitivityLevel, SensitivityResult
from .image_sensitivity import ImageSensitivityClassifier

# TMO action -> modality indices mapping (0=local, 1+=cloud with image combos)
ACTION_TO_MODALITY_INDICES = {
    0: [], 1: [], 2: [0], 3: [1], 4: [2],
    5: [0, 1], 6: [0, 2], 7: [1, 2], 8: [0, 1, 2],
}


@dataclass
class MultimodalSensitivity:
    """Result of multimodal sensitivity analysis (text + images)"""
    text: SensitivityResult
    images: Dict[int, SensitivityResult]  # {0: ..., 1: ..., 2: ...}
    
    def get_modality_risk(self, modality_indices: List[int]) -> float:
        """
        Compute privacy risk for selected image modalities sent to cloud.
        Text risk is handled separately (privacy_risk) to avoid double-counting.
        
        Args:
            modality_indices: Image indices sent to cloud (0, 1, 2)
            
        Returns:
            Normalized risk (0.0 to 1.0)
        """
        if not modality_indices or not self.images:
            return 0.0
        risk = sum(self.images[idx].score for idx in modality_indices if idx in self.images)
        return min(risk / len(modality_indices), 1.0)


class OffloadingDecision(Enum):
    """Decision for where to process the query"""
    LOCAL_ONLY = "local"        # Process entirely on device
    CLOUD_ONLY = "cloud"        # Process entirely on cloud
    HYBRID = "hybrid"           # Split processing


@dataclass
class PrivacyBudget:
    """
    Privacy Budget Tracker
    
    Tracks cumulative privacy risk and enforces budget constraints.
    Based on the concept: Σ PrivacyRisk(qₜ, aₜ) ≤ ε
    """
    epsilon: float = 1.0  # Total privacy budget
    consumed: float = 0.0  # Consumed budget so far
    history: List[Dict] = field(default_factory=list)
    
    @property
    def remaining(self) -> float:
        """Remaining privacy budget"""
        return max(0.0, self.epsilon - self.consumed)
    
    @property
    def utilization(self) -> float:
        """Budget utilization ratio (0.0 to 1.0)"""
        return self.consumed / self.epsilon if self.epsilon > 0 else 1.0
    
    def can_afford(self, risk: float) -> bool:
        """Check if we can afford this privacy risk"""
        return self.remaining >= risk
    
    def consume(self, risk: float, query_id: str, details: Dict = None):
        """Consume privacy budget"""
        self.consumed += risk
        self.history.append({
            'query_id': query_id,
            'risk': risk,
            'timestamp': time.time(),
            'remaining': self.remaining,
            'details': details or {},
        })
    
    def reset(self):
        """Reset budget (e.g., at start of new session)"""
        self.consumed = 0.0
        self.history = []


@dataclass
class OffloadingResult:
    """Result of offloading decision"""
    decision: OffloadingDecision
    sensitivity: SensitivityResult
    privacy_risk: float
    can_use_cloud: bool
    recommended_action: int  # Action for RL environment (0=local, 1+=cloud)
    explanation: str
    
    def to_dict(self) -> Dict:
        return {
            'decision': self.decision.value,
            'sensitivity_level': self.sensitivity.level.name,
            'sensitivity_score': self.sensitivity.score,
            'privacy_risk': self.privacy_risk,
            'can_use_cloud': self.can_use_cloud,
            'recommended_action': self.recommended_action,
        }


class PrivacyManager:
    """
    Central Privacy Manager for Privacy-TMO
    
    Responsibilities:
    1. Classify query sensitivity
    2. Calculate privacy risk for offloading decisions
    3. Manage privacy budget
    4. Make offloading recommendations
    """
    
    def __init__(
        self,
        config: Optional[PrivacyTMOConfig] = None,
        enable_ner: bool = True,
        enable_ml: bool = False,  # Disabled by default (requires training)
        enable_image_sensitivity: bool = False,
    ):
        self.config = config or PrivacyTMOConfig()
        self.privacy_config = self.config.privacy
        
        # Initialize sensitivity classifier
        self.classifier = SensitivityClassifier(
            use_ner=enable_ner,
            use_ml=enable_ml,
            device="cpu",  # Use CPU for classifier (save GPU for LLM)
        )
        
        # Initialize image sensitivity classifier (optional)
        self.image_classifier = ImageSensitivityClassifier(
            use_face_detection=enable_image_sensitivity,
            use_ocr=False,
        ) if enable_image_sensitivity else None
        
        # Initialize privacy budget
        self.budget = PrivacyBudget(epsilon=self.privacy_config.privacy_budget)
        
        # Query counter for IDs
        self._query_counter = 0
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'local_decisions': 0,
            'cloud_decisions': 0,
            'hybrid_decisions': 0,
            'blocked_by_budget': 0,
        }
    
    def analyze_query(self, query: str) -> SensitivityResult:
        """
        Analyze query sensitivity
        
        Args:
            query: User's input query
            
        Returns:
            SensitivityResult with classification details
        """
        return self.classifier.classify(query)
    
    def analyze_multimodal(
        self,
        text: str,
        images: Optional[Dict[int, Any]] = None,
        simulate_image_sensitivity: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> MultimodalSensitivity:
        """
        Analyze sensitivity of text + images for modality-aware offloading.
        
        Args:
            text: User query text
            images: Dict of {image_index: image_path_or_array} (optional)
            simulate_image_sensitivity: Use simulated scores when images unavailable
            context: Context for simulation (task_cat, prompt, etc.)
            
        Returns:
            MultimodalSensitivity with text and per-image sensitivity
        """
        text_sens = self.analyze_query(text)
        image_sens: Dict[int, SensitivityResult] = {}
        
        if images:
            for idx, img in images.items():
                if self.image_classifier:
                    image_sens[idx] = self.image_classifier.classify(img)
                else:
                    image_sens[idx] = SensitivityResult(
                        level=SensitivityLevel.PUBLIC,
                        score=0.0,
                        confidence=0.5,
                        detected_entities=[],
                        explanation="Image classifier not enabled",
                    )
        elif simulate_image_sensitivity:
            from .image_sensitivity import ImageSensitivityClassifier
            sim_classifier = ImageSensitivityClassifier(use_face_detection=False, use_ocr=False)
            ctx = context or {"prompt": text}
            for i in [0, 1, 2]:
                image_sens[i] = sim_classifier.classify_simulated(i, ctx)
        
        return MultimodalSensitivity(text=text_sens, images=image_sens)
    
    def calculate_modality_privacy_risk(
        self,
        mm_sensitivity: MultimodalSensitivity,
        action: int,
    ) -> float:
        """
        Compute privacy risk for given action based on modality selection.
        
        Args:
            mm_sensitivity: Result from analyze_multimodal
            action: TMO action (0=local, 1-8=cloud with modality combos)
            
        Returns:
            Privacy risk (0.0 to 1.0)
        """
        if action == 0:
            return 0.0
        modality_indices = ACTION_TO_MODALITY_INDICES.get(action, [])
        return mm_sensitivity.get_modality_risk(modality_indices)
    
    def calculate_privacy_risk(
        self, 
        sensitivity: SensitivityResult,
        offload_to_cloud: bool,
    ) -> float:
        """
        Calculate privacy risk for a given offloading decision
        
        PrivacyRisk(q, a) = sensitivity(q) × I(a routes q to cloud)
        
        Args:
            sensitivity: Sensitivity classification result
            offload_to_cloud: Whether query would be sent to cloud
            
        Returns:
            Privacy risk value (0.0 to 1.0)
        """
        if not offload_to_cloud:
            # Local processing has zero privacy risk
            return 0.0
        
        # Risk is proportional to sensitivity score
        base_risk = sensitivity.score
        
        # Apply entity-based adjustment
        num_entities = len(sensitivity.detected_entities)
        entity_multiplier = 1.0 + (num_entities * 0.1)
        
        # Private level queries have maximum risk
        if sensitivity.level == SensitivityLevel.PRIVATE:
            base_risk = 1.0
        
        return min(base_risk * entity_multiplier, 1.0)
    
    def make_offloading_decision(
        self,
        query: str,
        force_local: bool = False,
        ignore_budget: bool = False,
    ) -> OffloadingResult:
        """
        Make offloading decision for a query
        
        Decision logic:
        1. Classify sensitivity
        2. Calculate privacy risk for cloud offloading
        3. Check privacy budget
        4. Return recommendation
        
        Args:
            query: User's input query
            force_local: Force local processing regardless of analysis
            ignore_budget: Ignore privacy budget constraints
            
        Returns:
            OffloadingResult with decision and details
        """
        self._query_counter += 1
        self.stats['total_queries'] += 1
        
        # Step 1: Classify sensitivity
        sensitivity = self.analyze_query(query)
        
        # Step 2: Calculate privacy risk if sent to cloud
        cloud_risk = self.calculate_privacy_risk(sensitivity, offload_to_cloud=True)
        
        # Step 3: Make decision
        decision = OffloadingDecision.LOCAL_ONLY
        can_use_cloud = True
        recommended_action = 0  # Default: local
        explanation_parts = []
        
        if force_local:
            decision = OffloadingDecision.LOCAL_ONLY
            explanation_parts.append("Forced local processing")
        
        elif sensitivity.level == SensitivityLevel.PRIVATE:
            # Private data must stay local
            decision = OffloadingDecision.LOCAL_ONLY
            can_use_cloud = False
            recommended_action = 0
            explanation_parts.append(f"Private data detected (score: {sensitivity.score:.2f})")
            self.stats['local_decisions'] += 1
        
        elif sensitivity.level == SensitivityLevel.SEMI_SENSITIVE:
            # Check budget for semi-sensitive
            if not ignore_budget and not self.budget.can_afford(cloud_risk):
                decision = OffloadingDecision.LOCAL_ONLY
                can_use_cloud = False
                recommended_action = 0
                explanation_parts.append("Privacy budget exhausted")
                self.stats['blocked_by_budget'] += 1
            else:
                # Hybrid processing recommended
                decision = OffloadingDecision.HYBRID
                can_use_cloud = True
                recommended_action = 1  # Cloud with caution
                explanation_parts.append("Semi-sensitive: hybrid processing recommended")
                self.stats['hybrid_decisions'] += 1
        
        else:
            # Public data can go to cloud
            decision = OffloadingDecision.CLOUD_ONLY
            can_use_cloud = True
            recommended_action = 1  # Cloud
            explanation_parts.append("Public query: cloud processing safe")
            self.stats['cloud_decisions'] += 1
        
        # Generate explanation
        explanation = " | ".join(explanation_parts)
        if sensitivity.detected_entities:
            entity_types = list(set(e['type'] for e in sensitivity.detected_entities))
            explanation += f"\nDetected: {entity_types}"
        
        return OffloadingResult(
            decision=decision,
            sensitivity=sensitivity,
            privacy_risk=cloud_risk,
            can_use_cloud=can_use_cloud,
            recommended_action=recommended_action,
            explanation=explanation,
        )
    
    def record_offloading(
        self,
        query: str,
        actual_action: int,
        sensitivity: Optional[SensitivityResult] = None,
    ):
        """
        Record actual offloading decision and update budget
        
        Call this after actual inference to track privacy consumption.
        
        Args:
            query: The processed query
            actual_action: 0 for local, 1+ for cloud
            sensitivity: Pre-computed sensitivity (optional)
        """
        if sensitivity is None:
            sensitivity = self.analyze_query(query)
        
        offloaded_to_cloud = actual_action > 0
        risk = self.calculate_privacy_risk(sensitivity, offloaded_to_cloud)
        
        if offloaded_to_cloud and risk > 0:
            self.budget.consume(
                risk=risk,
                query_id=f"q_{self._query_counter}",
                details={
                    'query_preview': query[:50],
                    'sensitivity_level': sensitivity.level.name,
                    'action': actual_action,
                }
            )
    
    def get_security_score(self, query: str, action: int) -> float:
        """
        Get security score for RL reward function
        
        Compatible with existing tmo_interface.get_security_score()
        
        Args:
            query: User query
            action: 0 for local, 1+ for cloud
            
        Returns:
            Security score (1.0 = safe, 0.0 = risky)
        """
        if action == 0:
            # Local processing is always safe
            return 1.0
        
        sensitivity = self.analyze_query(query)
        
        # Private data to cloud = 0.0
        if sensitivity.level == SensitivityLevel.PRIVATE:
            return 0.0
        
        # Semi-sensitive to cloud = reduced score
        if sensitivity.level == SensitivityLevel.SEMI_SENSITIVE:
            return 0.5
        
        # Public to cloud = safe
        return 1.0
    
    def get_privacy_risk_for_rl(
        self,
        query: str,
        action: int,
    ) -> float:
        """
        Get privacy risk value for RL reward function
        
        This is β₄ × PrivacyRisk(q, a) in the reward function
        
        Args:
            query: User query
            action: 0 for local, 1+ for cloud
            
        Returns:
            Privacy risk penalty (0.0 to 1.0)
        """
        sensitivity = self.analyze_query(query)
        risk = self.calculate_privacy_risk(sensitivity, offload_to_cloud=(action > 0))
        return risk * self.privacy_config.privacy_risk_weight
    
    def get_budget_status(self) -> Dict:
        """Get current privacy budget status"""
        return {
            'epsilon': self.budget.epsilon,
            'consumed': self.budget.consumed,
            'remaining': self.budget.remaining,
            'utilization': self.budget.utilization,
            'num_queries': len(self.budget.history),
        }
    
    def get_statistics(self) -> Dict:
        """Get privacy manager statistics"""
        return {
            **self.stats,
            'budget': self.get_budget_status(),
        }
    
    def reset_session(self):
        """Reset for new session"""
        self.budget.reset()
        self._query_counter = 0
        self.stats = {
            'total_queries': 0,
            'local_decisions': 0,
            'cloud_decisions': 0,
            'hybrid_decisions': 0,
            'blocked_by_budget': 0,
        }


# Convenience function for drop-in replacement
def get_security_score(prompt: str, action: int) -> float:
    """
    Drop-in replacement for tmo_interface.get_security_score()
    
    Uses Privacy Manager for more sophisticated analysis.
    """
    manager = PrivacyManager(enable_ner=False, enable_ml=False)
    return manager.get_security_score(prompt, action)


if __name__ == "__main__":
    print("🧪 Testing Privacy Manager\n")
    
    # Initialize manager
    manager = PrivacyManager(enable_ner=False, enable_ml=False)
    
    # Test queries
    test_cases = [
        ("What is Python?", "Should allow cloud"),
        ("How to sort a list?", "Should allow cloud"),
        ("My email is test@example.com", "Semi-sensitive"),
        ("Call me at 010-1234-5678", "Semi-sensitive"),
        ("My password is secret123", "Must be local"),
        ("My SSN is 123-45-6789", "Must be local"),
        ("내 비밀번호는 test123", "Must be local (Korean)"),
    ]
    
    print("=" * 70)
    for query, expected in test_cases:
        result = manager.make_offloading_decision(query)
        
        emoji = {
            OffloadingDecision.LOCAL_ONLY: "🔒",
            OffloadingDecision.CLOUD_ONLY: "☁️",
            OffloadingDecision.HYBRID: "🔀",
        }[result.decision]
        
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Decision: {emoji} {result.decision.value}")
        print(f"Sensitivity: {result.sensitivity.level.name} (score: {result.sensitivity.score:.2f})")
        print(f"Privacy Risk: {result.privacy_risk:.2f}")
        print(f"Can use cloud: {result.can_use_cloud}")
    
    print("\n" + "=" * 70)
    print("\n📊 Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Privacy Manager ready!")
