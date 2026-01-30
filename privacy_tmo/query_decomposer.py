"""
Query Decomposition for Selective Offloading

Implements the key contribution: Partial Offloading
- Decomposes queries into sensitive and non-sensitive parts
- Routes each part to appropriate processor (local/cloud)
- Aggregates responses into coherent output

This enables fine-grained privacy control beyond binary decisions.
"""

import re
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from .sensitivity_classifier import (
    SensitivityClassifier, 
    SensitivityLevel, 
    SensitivityResult
)

if TYPE_CHECKING:
    from .privacy_manager import MultimodalSensitivity


class QueryPartType(Enum):
    """Type of query part for routing"""
    SENSITIVE = "sensitive"      # Must be processed locally
    NON_SENSITIVE = "non_sensitive"  # Can be processed in cloud
    CONTEXT = "context"          # Context needed for both


@dataclass
class QueryPart:
    """A decomposed part of the original query"""
    text: str
    part_type: QueryPartType
    sensitivity: SensitivityResult
    start_idx: int  # Position in original query
    end_idx: int
    placeholder: str = ""  # Placeholder for aggregation
    
    def __repr__(self):
        return f"QueryPart({self.part_type.value}: '{self.text[:30]}...')"


@dataclass
class DecomposedQuery:
    """Result of query decomposition"""
    original: str
    parts: List[QueryPart]
    local_query: str      # Query for local processing
    cloud_query: str      # Query for cloud processing (sanitized)
    has_sensitive: bool
    decomposition_strategy: str
    
    def get_routing_summary(self) -> Dict:
        """Get summary of how query will be routed"""
        return {
            'total_parts': len(self.parts),
            'sensitive_parts': sum(1 for p in self.parts if p.part_type == QueryPartType.SENSITIVE),
            'non_sensitive_parts': sum(1 for p in self.parts if p.part_type == QueryPartType.NON_SENSITIVE),
            'has_sensitive': self.has_sensitive,
            'strategy': self.decomposition_strategy,
        }


class QueryDecomposer:
    """
    Query Decomposition Engine
    
    Strategies:
    1. Sentence-level: Split by sentences, classify each
    2. Entity-level: Identify and mask sensitive entities
    3. Clause-level: Split by clauses for finer granularity
    """
    
    def __init__(self, classifier: Optional[SensitivityClassifier] = None):
        self.classifier = classifier or SensitivityClassifier(
            use_ner=False,  # Faster without NER
            use_ml=False,
        )
        
        # Sentence splitting patterns
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+|(?<=[。！？])\s*'  # English + Korean/Chinese
        )
        
        # Clause splitting patterns
        self.clause_pattern = re.compile(
            r'(?<=,)\s+|(?<=;)\s+|(?<=:)\s+|(?<=[，；：])\s*'
        )
        
        # Placeholder template
        self.placeholder_template = "[REDACTED_{idx}]"
    
    def decompose(
        self, 
        query: str,
        strategy: str = "auto"
    ) -> DecomposedQuery:
        """
        Decompose query into sensitive and non-sensitive parts
        
        Args:
            query: Original user query
            strategy: "sentence", "entity", "clause", or "auto"
            
        Returns:
            DecomposedQuery with routing information
        """
        # First, check overall sensitivity
        overall_sensitivity = self.classifier.classify(query)
        
        # If completely public or completely private, no decomposition needed
        if overall_sensitivity.level == SensitivityLevel.PUBLIC:
            return self._create_single_part_result(
                query, 
                QueryPartType.NON_SENSITIVE,
                overall_sensitivity,
                "none_needed"
            )
        
        if overall_sensitivity.level == SensitivityLevel.PRIVATE:
            # Check if we can extract any non-sensitive parts
            if len(overall_sensitivity.detected_entities) == 0:
                return self._create_single_part_result(
                    query,
                    QueryPartType.SENSITIVE,
                    overall_sensitivity,
                    "fully_sensitive"
                )
        
        # Auto-select strategy based on query characteristics
        if strategy == "auto":
            strategy = self._select_strategy(query, overall_sensitivity)
        
        # Apply decomposition strategy
        if strategy == "sentence":
            return self._decompose_by_sentence(query, overall_sensitivity)
        elif strategy == "entity":
            return self._decompose_by_entity(query, overall_sensitivity)
        elif strategy == "clause":
            return self._decompose_by_clause(query, overall_sensitivity)
        else:
            return self._decompose_by_sentence(query, overall_sensitivity)
    
    def _select_strategy(
        self, 
        query: str, 
        sensitivity: SensitivityResult
    ) -> str:
        """Auto-select decomposition strategy"""
        # If detected entities, use entity-level masking
        if sensitivity.detected_entities:
            return "entity"
        
        # If multiple sentences, use sentence-level
        sentences = self.sentence_pattern.split(query)
        if len(sentences) > 1:
            return "sentence"
        
        # Otherwise, try clause-level
        return "clause"
    
    def _decompose_by_sentence(
        self, 
        query: str,
        overall_sensitivity: SensitivityResult
    ) -> DecomposedQuery:
        """
        Sentence-level decomposition
        
        Example:
        Input: "My password is secret123. What is Python?"
        Output: 
          - Sensitive: "My password is secret123."
          - Non-sensitive: "What is Python?"
        """
        sentences = self.sentence_pattern.split(query)
        if not sentences:
            sentences = [query]
        
        parts = []
        local_parts = []
        cloud_parts = []
        current_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Find position in original
            start_idx = query.find(sentence, current_idx)
            end_idx = start_idx + len(sentence)
            current_idx = end_idx
            
            # Classify this sentence
            sent_sensitivity = self.classifier.classify(sentence)
            
            # Determine part type
            if sent_sensitivity.level == SensitivityLevel.PRIVATE:
                part_type = QueryPartType.SENSITIVE
                local_parts.append(sentence)
                cloud_parts.append(self.placeholder_template.format(idx=i))
            elif sent_sensitivity.level == SensitivityLevel.SEMI_SENSITIVE:
                part_type = QueryPartType.SENSITIVE
                local_parts.append(sentence)
                cloud_parts.append(self.placeholder_template.format(idx=i))
            else:
                part_type = QueryPartType.NON_SENSITIVE
                cloud_parts.append(sentence)
            
            parts.append(QueryPart(
                text=sentence,
                part_type=part_type,
                sensitivity=sent_sensitivity,
                start_idx=start_idx,
                end_idx=end_idx,
                placeholder=self.placeholder_template.format(idx=i),
            ))
        
        return DecomposedQuery(
            original=query,
            parts=parts,
            local_query=" ".join(local_parts) if local_parts else "",
            cloud_query=" ".join(cloud_parts) if cloud_parts else "",
            has_sensitive=bool(local_parts),
            decomposition_strategy="sentence",
        )
    
    def _decompose_by_entity(
        self, 
        query: str,
        overall_sensitivity: SensitivityResult
    ) -> DecomposedQuery:
        """
        Entity-level decomposition (masking)
        
        Example:
        Input: "Send email to john@example.com about the meeting"
        Output:
          - Original with mask: "Send email to [EMAIL_0] about the meeting"
          - Sensitive entities: {"[EMAIL_0]": "john@example.com"}
        """
        parts = []
        masked_query = query
        entity_map = {}
        
        # Sort entities by position (reverse to maintain indices)
        entities = overall_sensitivity.detected_entities
        
        # Group entities by their approximate position
        for i, entity in enumerate(entities):
            entity_value = entity.get('value', '')
            if not entity_value or entity_value not in query:
                continue
            
            # Find entity in query
            start_idx = query.find(entity_value)
            if start_idx == -1:
                continue
            
            end_idx = start_idx + len(entity_value)
            
            # Create placeholder
            placeholder = f"[{entity['type']}_{i}]"
            entity_map[placeholder] = entity_value
            
            # Create part
            parts.append(QueryPart(
                text=entity_value,
                part_type=QueryPartType.SENSITIVE,
                sensitivity=SensitivityResult(
                    level=SensitivityLevel(entity.get('level', 1)),
                    score=0.8,
                    confidence=0.8,
                    detected_entities=[entity],
                    explanation=f"Sensitive entity: {entity['type']}"
                ),
                start_idx=start_idx,
                end_idx=end_idx,
                placeholder=placeholder,
            ))
            
            # Mask in query
            masked_query = masked_query.replace(entity_value, placeholder, 1)
        
        # Create non-sensitive context part
        if masked_query != query:
            parts.append(QueryPart(
                text=masked_query,
                part_type=QueryPartType.CONTEXT,
                sensitivity=SensitivityResult(
                    level=SensitivityLevel.PUBLIC,
                    score=0.0,
                    confidence=0.9,
                    detected_entities=[],
                    explanation="Context after entity masking"
                ),
                start_idx=0,
                end_idx=len(query),
                placeholder="",
            ))
        
        return DecomposedQuery(
            original=query,
            parts=parts,
            local_query=query,  # Full query needed locally for context
            cloud_query=masked_query,  # Masked version for cloud
            has_sensitive=bool(entity_map),
            decomposition_strategy="entity",
        )
    
    def _decompose_by_clause(
        self, 
        query: str,
        overall_sensitivity: SensitivityResult
    ) -> DecomposedQuery:
        """
        Clause-level decomposition
        
        Finer granularity than sentence-level
        """
        clauses = self.clause_pattern.split(query)
        if len(clauses) <= 1:
            # Fall back to entity-level if no clauses
            return self._decompose_by_entity(query, overall_sensitivity)
        
        parts = []
        local_parts = []
        cloud_parts = []
        current_idx = 0
        
        for i, clause in enumerate(clauses):
            clause = clause.strip()
            if not clause:
                continue
            
            start_idx = query.find(clause, current_idx)
            end_idx = start_idx + len(clause)
            current_idx = end_idx
            
            clause_sensitivity = self.classifier.classify(clause)
            
            if clause_sensitivity.level >= SensitivityLevel.SEMI_SENSITIVE:
                part_type = QueryPartType.SENSITIVE
                local_parts.append(clause)
                cloud_parts.append(self.placeholder_template.format(idx=i))
            else:
                part_type = QueryPartType.NON_SENSITIVE
                cloud_parts.append(clause)
            
            parts.append(QueryPart(
                text=clause,
                part_type=part_type,
                sensitivity=clause_sensitivity,
                start_idx=start_idx,
                end_idx=end_idx,
                placeholder=self.placeholder_template.format(idx=i),
            ))
        
        return DecomposedQuery(
            original=query,
            parts=parts,
            local_query=", ".join(local_parts) if local_parts else "",
            cloud_query=", ".join(cloud_parts) if cloud_parts else "",
            has_sensitive=bool(local_parts),
            decomposition_strategy="clause",
        )
    
    def _create_single_part_result(
        self,
        query: str,
        part_type: QueryPartType,
        sensitivity: SensitivityResult,
        strategy: str
    ) -> DecomposedQuery:
        """Create result for queries that don't need decomposition"""
        part = QueryPart(
            text=query,
            part_type=part_type,
            sensitivity=sensitivity,
            start_idx=0,
            end_idx=len(query),
        )
        
        if part_type == QueryPartType.SENSITIVE:
            return DecomposedQuery(
                original=query,
                parts=[part],
                local_query=query,
                cloud_query="",
                has_sensitive=True,
                decomposition_strategy=strategy,
            )
        else:
            return DecomposedQuery(
                original=query,
                parts=[part],
                local_query="",
                cloud_query=query,
                has_sensitive=False,
                decomposition_strategy=strategy,
            )


class SelectiveOffloader:
    """
    Selective Offloading Engine
    
    Routes decomposed query parts to appropriate processors
    and manages the hybrid processing workflow.
    """
    
    def __init__(
        self,
        decomposer: Optional[QueryDecomposer] = None,
        local_inference_fn=None,
        cloud_inference_fn=None,
    ):
        self.decomposer = decomposer or QueryDecomposer()
        self.local_inference_fn = local_inference_fn
        self.cloud_inference_fn = cloud_inference_fn
    
    def process_query(
        self,
        query: str,
        strategy: str = "auto",
    ) -> Dict:
        """
        Process query with selective offloading
        
        Args:
            query: User query
            strategy: Decomposition strategy
            
        Returns:
            Dict with processing results and routing info
        """
        # Decompose query
        decomposed = self.decomposer.decompose(query, strategy)
        
        results = {
            'original_query': query,
            'decomposition': decomposed.get_routing_summary(),
            'local_query': decomposed.local_query,
            'cloud_query': decomposed.cloud_query,
            'local_response': None,
            'cloud_response': None,
            'final_response': None,
            'processing_path': [],
        }
        
        # Route based on decomposition
        if not decomposed.has_sensitive:
            # All public - cloud only
            results['processing_path'].append('cloud_only')
            if self.cloud_inference_fn:
                results['cloud_response'], _ = self.cloud_inference_fn(query)
                results['final_response'] = results['cloud_response']
        
        elif decomposed.cloud_query == "" or decomposed.cloud_query == query:
            # All sensitive - local only
            results['processing_path'].append('local_only')
            if self.local_inference_fn:
                results['local_response'], _ = self.local_inference_fn(query)
                results['final_response'] = results['local_response']
        
        else:
            # Hybrid processing
            results['processing_path'].append('hybrid')
            
            # Process sensitive parts locally
            if decomposed.local_query and self.local_inference_fn:
                results['local_response'], _ = self.local_inference_fn(decomposed.local_query)
            
            # Process masked query in cloud
            if decomposed.cloud_query and self.cloud_inference_fn:
                results['cloud_response'], _ = self.cloud_inference_fn(decomposed.cloud_query)
            
            # Aggregate responses
            results['final_response'] = self._aggregate_responses(
                decomposed,
                results['local_response'],
                results['cloud_response']
            )
        
        return results
    
    def _aggregate_responses(
        self,
        decomposed: DecomposedQuery,
        local_response: Optional[str],
        cloud_response: Optional[str],
    ) -> str:
        """
        Aggregate local and cloud responses
        
        Strategy depends on decomposition type:
        - Sentence: Concatenate responses
        - Entity: Replace placeholders in cloud response
        """
        if not local_response and not cloud_response:
            return ""
        
        if not cloud_response:
            return local_response or ""
        
        if not local_response:
            return cloud_response
        
        # For entity-level: replace placeholders
        if decomposed.decomposition_strategy == "entity":
            result = cloud_response
            for part in decomposed.parts:
                if part.part_type == QueryPartType.SENSITIVE and part.placeholder:
                    # Don't expose sensitive data in final response
                    # Instead, use local response for those parts
                    pass
            return f"{cloud_response}\n\n[Local context processed separately]"
        
        # For sentence-level: combine intelligently
        return f"{cloud_response}\n\n{local_response}"
    
    def get_routing_decision(self, query: str) -> Dict:
        """
        Get routing decision without executing inference
        
        Useful for RL environment to determine action
        """
        decomposed = self.decomposer.decompose(query)
        
        if not decomposed.has_sensitive:
            action = 1  # Cloud
            decision = "cloud_only"
        elif decomposed.cloud_query == "" or decomposed.cloud_query == query:
            action = 0  # Local
            decision = "local_only"
        else:
            action = 2  # Hybrid (could be mapped to specific modality combination)
            decision = "hybrid"
        
        return {
            'action': action,
            'decision': decision,
            'decomposition': decomposed.get_routing_summary(),
            'local_query': decomposed.local_query,
            'cloud_query': decomposed.cloud_query,
        }


@dataclass
class MultimodalDecomposedQuery:
    """Decomposed query with modality routing decisions"""
    text_decomposition: DecomposedQuery
    modality_routing: Dict[int, str]  # {0: "local", 1: "cloud", ...}
    local_modalities: List[int]
    cloud_modalities: List[int]


class MultimodalQueryDecomposer:
    """
    Multimodal query decomposer that adds modality routing.
    
    - Text is decomposed using existing QueryDecomposer
    - Image modalities are routed based on per-image sensitivity
    """
    
    def __init__(
        self,
        text_decomposer: Optional[QueryDecomposer] = None,
        image_risk_threshold: float = 0.5,
    ):
        self.text_decomposer = text_decomposer or QueryDecomposer()
        self.image_risk_threshold = image_risk_threshold
    
    def decompose(
        self,
        query: str,
        mm_sensitivity: Optional["MultimodalSensitivity"] = None,
        strategy: str = "auto",
    ) -> MultimodalDecomposedQuery:
        text_decomp = self.text_decomposer.decompose(query, strategy=strategy)
        modality_routing: Dict[int, str] = {}
        
        if mm_sensitivity and mm_sensitivity.images:
            for idx, result in mm_sensitivity.images.items():
                if result.level == SensitivityLevel.PRIVATE or result.score >= self.image_risk_threshold:
                    modality_routing[idx] = "local"
                else:
                    modality_routing[idx] = "cloud"
        else:
            # Default: allow all modalities to cloud if no sensitivity info
            modality_routing = {0: "cloud", 1: "cloud", 2: "cloud"}
        
        local_modalities = [i for i, v in modality_routing.items() if v == "local"]
        cloud_modalities = [i for i, v in modality_routing.items() if v == "cloud"]
        
        return MultimodalDecomposedQuery(
            text_decomposition=text_decomp,
            modality_routing=modality_routing,
            local_modalities=local_modalities,
            cloud_modalities=cloud_modalities,
        )


if __name__ == "__main__":
    print("🧪 Testing Query Decomposition\n")
    
    decomposer = QueryDecomposer()
    
    test_queries = [
        # Public - no decomposition needed
        "What is machine learning?",
        
        # Sentence-level decomposition
        "My password is secret123. What is Python programming?",
        
        # Entity-level masking
        "Send an email to john@example.com about the meeting tomorrow.",
        
        # Complex query
        "My SSN is 123-45-6789. Please check my tax records. What's the weather today?",
        
        # Korean
        "내 비밀번호는 test123이야. 오늘 날씨 어때?",
    ]
    
    print("=" * 70)
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        
        result = decomposer.decompose(query)
        
        print(f"   Strategy: {result.decomposition_strategy}")
        print(f"   Has sensitive: {result.has_sensitive}")
        print(f"   Parts: {len(result.parts)}")
        
        for part in result.parts:
            emoji = "🔴" if part.part_type == QueryPartType.SENSITIVE else "🟢"
            print(f"     {emoji} [{part.part_type.value}] {part.text[:40]}...")
        
        print(f"   Local query: {result.local_query[:50]}..." if result.local_query else "   Local query: (none)")
        print(f"   Cloud query: {result.cloud_query[:50]}..." if result.cloud_query else "   Cloud query: (none)")
    
    print("\n" + "=" * 70)
    print("\n✅ Query Decomposition ready!")
