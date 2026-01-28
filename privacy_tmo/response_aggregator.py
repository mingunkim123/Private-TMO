"""
Response Aggregation for Hybrid Processing

Combines responses from local and cloud LLMs into coherent output.

Strategies:
1. Concatenation: Simple join of responses
2. Merge: Intelligent merging with overlap detection
3. Template: Fill cloud response with local details
4. Priority: Select best response based on quality
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .query_decomposer import DecomposedQuery, QueryPartType


class AggregationStrategy(Enum):
    """Response aggregation strategies"""
    CONCATENATE = "concatenate"  # Simple concatenation
    MERGE = "merge"              # Intelligent merge
    TEMPLATE = "template"        # Template filling
    PRIORITY = "priority"        # Select best
    INTERLEAVE = "interleave"    # Interleave by topic


@dataclass
class AggregatedResponse:
    """Result of response aggregation"""
    final_response: str
    local_contribution: str
    cloud_contribution: str
    strategy_used: AggregationStrategy
    metadata: Dict
    
    def __str__(self):
        return self.final_response


class ResponseAggregator:
    """
    Aggregates responses from local and cloud processing
    
    Ensures:
    1. Coherent final output
    2. No sensitive data leakage in aggregated response
    3. Consistent tone and style
    """
    
    def __init__(self):
        # Sentence boundary pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        
        # Placeholder pattern
        self.placeholder_pattern = re.compile(r'\[REDACTED_\d+\]|\[[A-Z_]+_\d+\]')
    
    def aggregate(
        self,
        decomposed: DecomposedQuery,
        local_response: Optional[str],
        cloud_response: Optional[str],
        strategy: AggregationStrategy = AggregationStrategy.MERGE,
    ) -> AggregatedResponse:
        """
        Aggregate local and cloud responses
        
        Args:
            decomposed: Original query decomposition
            local_response: Response from local LLM
            cloud_response: Response from cloud LLM
            strategy: Aggregation strategy to use
            
        Returns:
            AggregatedResponse with combined output
        """
        # Handle edge cases
        if not local_response and not cloud_response:
            return AggregatedResponse(
                final_response="Sorry, I couldn't generate a response.",
                local_contribution="",
                cloud_contribution="",
                strategy_used=strategy,
                metadata={'error': 'no_responses'},
            )
        
        if not local_response:
            return AggregatedResponse(
                final_response=self._sanitize_response(cloud_response),
                local_contribution="",
                cloud_contribution=cloud_response,
                strategy_used=AggregationStrategy.PRIORITY,
                metadata={'source': 'cloud_only'},
            )
        
        if not cloud_response:
            return AggregatedResponse(
                final_response=local_response,
                local_contribution=local_response,
                cloud_contribution="",
                strategy_used=AggregationStrategy.PRIORITY,
                metadata={'source': 'local_only'},
            )
        
        # Apply selected strategy
        if strategy == AggregationStrategy.CONCATENATE:
            result = self._concatenate(decomposed, local_response, cloud_response)
        elif strategy == AggregationStrategy.MERGE:
            result = self._merge(decomposed, local_response, cloud_response)
        elif strategy == AggregationStrategy.TEMPLATE:
            result = self._template_fill(decomposed, local_response, cloud_response)
        elif strategy == AggregationStrategy.PRIORITY:
            result = self._priority_select(decomposed, local_response, cloud_response)
        elif strategy == AggregationStrategy.INTERLEAVE:
            result = self._interleave(decomposed, local_response, cloud_response)
        else:
            result = self._merge(decomposed, local_response, cloud_response)
        
        return AggregatedResponse(
            final_response=result['final'],
            local_contribution=local_response,
            cloud_contribution=cloud_response,
            strategy_used=strategy,
            metadata=result.get('metadata', {}),
        )
    
    def _sanitize_response(self, response: str) -> str:
        """Remove any remaining placeholders from response"""
        return self.placeholder_pattern.sub('[information processed locally]', response)
    
    def _concatenate(
        self,
        decomposed: DecomposedQuery,
        local_response: str,
        cloud_response: str,
    ) -> Dict:
        """
        Simple concatenation with section markers
        
        Format:
        [Cloud Response]
        ...
        
        [Local Details]
        ...
        """
        # Sanitize cloud response (remove placeholders)
        sanitized_cloud = self._sanitize_response(cloud_response)
        
        # Combine with clear separation
        if decomposed.decomposition_strategy == "entity":
            # For entity masking, cloud response is main, local adds details
            final = f"{sanitized_cloud}\n\n---\n\n[Additional context processed locally]"
        else:
            # For sentence-level, both are important
            final = f"{sanitized_cloud}\n\n---\n\n{local_response}"
        
        return {
            'final': final,
            'metadata': {'method': 'concatenation'},
        }
    
    def _merge(
        self,
        decomposed: DecomposedQuery,
        local_response: str,
        cloud_response: str,
    ) -> Dict:
        """
        Intelligent merge with overlap detection
        
        1. Identify overlapping content
        2. Keep unique parts from each
        3. Merge into coherent response
        """
        # Split into sentences
        local_sentences = self.sentence_pattern.split(local_response)
        cloud_sentences = self.sentence_pattern.split(cloud_response)
        
        # Sanitize cloud sentences
        cloud_sentences = [self._sanitize_response(s) for s in cloud_sentences]
        
        # Find unique sentences from local
        unique_local = []
        for sent in local_sentences:
            sent_lower = sent.lower().strip()
            is_duplicate = any(
                self._sentence_similarity(sent_lower, c.lower().strip()) > 0.7
                for c in cloud_sentences
            )
            if not is_duplicate and sent.strip():
                unique_local.append(sent)
        
        # Combine: cloud response + unique local additions
        if unique_local:
            cloud_text = " ".join(cloud_sentences)
            local_additions = " ".join(unique_local)
            final = f"{cloud_text}\n\nAdditionally: {local_additions}"
        else:
            final = " ".join(cloud_sentences)
        
        return {
            'final': final,
            'metadata': {
                'method': 'merge',
                'unique_local_sentences': len(unique_local),
            },
        }
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Simple word-overlap based similarity"""
        words1 = set(sent1.split())
        words2 = set(sent2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _template_fill(
        self,
        decomposed: DecomposedQuery,
        local_response: str,
        cloud_response: str,
    ) -> Dict:
        """
        Use cloud response as template, fill with local details
        
        Best for entity-level decomposition where cloud got
        the structure but not the specific values.
        """
        result = cloud_response
        
        # For entity-level: cloud might reference placeholders
        # We don't want to expose actual values, so we:
        # 1. Keep placeholders as "[processed locally]"
        # 2. Add note that sensitive data was handled securely
        
        result = self._sanitize_response(result)
        
        # If local response has specific information, add summary
        if local_response and len(local_response) > 50:
            result += "\n\n[Note: Sensitive details were processed securely on your device.]"
        
        return {
            'final': result,
            'metadata': {
                'method': 'template',
                'placeholders_sanitized': True,
            },
        }
    
    def _priority_select(
        self,
        decomposed: DecomposedQuery,
        local_response: str,
        cloud_response: str,
    ) -> Dict:
        """
        Select best response based on quality heuristics
        
        Factors:
        1. Length (within reason)
        2. Completeness
        3. Privacy preservation
        """
        scores = {
            'local': 0,
            'cloud': 0,
        }
        
        # Length score (moderate length preferred)
        local_len = len(local_response)
        cloud_len = len(cloud_response)
        optimal_len = 500  # Characters
        
        scores['local'] += max(0, 1 - abs(local_len - optimal_len) / optimal_len)
        scores['cloud'] += max(0, 1 - abs(cloud_len - optimal_len) / optimal_len)
        
        # Privacy score (local always better for sensitive)
        if decomposed.has_sensitive:
            scores['local'] += 1.0
            scores['cloud'] -= 0.5
        
        # Completeness (check for truncation markers)
        if "..." in local_response or local_response.endswith(","):
            scores['local'] -= 0.3
        if "..." in cloud_response or cloud_response.endswith(","):
            scores['cloud'] -= 0.3
        
        # Select winner
        if scores['local'] >= scores['cloud']:
            selected = local_response
            source = 'local'
        else:
            selected = self._sanitize_response(cloud_response)
            source = 'cloud'
        
        return {
            'final': selected,
            'metadata': {
                'method': 'priority',
                'selected': source,
                'scores': scores,
            },
        }
    
    def _interleave(
        self,
        decomposed: DecomposedQuery,
        local_response: str,
        cloud_response: str,
    ) -> Dict:
        """
        Interleave responses by matching parts to query parts
        
        Attempts to create natural flow by ordering responses
        according to query structure.
        """
        result_parts = []
        
        # Match response parts to query parts
        local_sentences = self.sentence_pattern.split(local_response)
        cloud_sentences = self.sentence_pattern.split(cloud_response)
        
        local_idx = 0
        cloud_idx = 0
        
        for part in decomposed.parts:
            if part.part_type == QueryPartType.SENSITIVE:
                # Use local response for sensitive parts
                if local_idx < len(local_sentences):
                    result_parts.append(local_sentences[local_idx].strip())
                    local_idx += 1
            else:
                # Use cloud response for non-sensitive parts
                if cloud_idx < len(cloud_sentences):
                    sanitized = self._sanitize_response(cloud_sentences[cloud_idx])
                    result_parts.append(sanitized.strip())
                    cloud_idx += 1
        
        # Add remaining sentences
        while local_idx < len(local_sentences):
            result_parts.append(local_sentences[local_idx].strip())
            local_idx += 1
        
        while cloud_idx < len(cloud_sentences):
            sanitized = self._sanitize_response(cloud_sentences[cloud_idx])
            result_parts.append(sanitized.strip())
            cloud_idx += 1
        
        final = " ".join(p for p in result_parts if p)
        
        return {
            'final': final,
            'metadata': {
                'method': 'interleave',
                'parts_used': len(result_parts),
            },
        }


class HybridInferenceEngine:
    """
    Complete hybrid inference engine combining all components
    
    This is the main entry point for Privacy-TMO inference.
    """
    
    def __init__(
        self,
        local_inference_fn=None,
        cloud_inference_fn=None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.MERGE,
    ):
        from .query_decomposer import QueryDecomposer
        from .privacy_manager import PrivacyManager
        
        self.decomposer = QueryDecomposer()
        self.privacy_manager = PrivacyManager(enable_ner=False, enable_ml=False)
        self.aggregator = ResponseAggregator()
        
        self.local_inference_fn = local_inference_fn
        self.cloud_inference_fn = cloud_inference_fn
        self.default_strategy = aggregation_strategy
    
    def infer(
        self,
        query: str,
        strategy: Optional[AggregationStrategy] = None,
    ) -> Dict:
        """
        Perform privacy-preserving hybrid inference
        
        Args:
            query: User's input query
            strategy: Aggregation strategy (default from init)
            
        Returns:
            Dict with response and processing details
        """
        strategy = strategy or self.default_strategy
        
        # Step 1: Make offloading decision
        decision = self.privacy_manager.make_offloading_decision(query)
        
        # Step 2: Decompose query
        decomposed = self.decomposer.decompose(query)
        
        # Step 3: Process based on decision
        local_response = None
        cloud_response = None
        local_latency = 0.0
        cloud_latency = 0.0
        
        if decision.decision.value in ['local', 'hybrid']:
            if self.local_inference_fn and decomposed.local_query:
                local_response, local_latency = self.local_inference_fn(
                    decomposed.local_query if decomposed.has_sensitive else query
                )
        
        if decision.can_use_cloud and decision.decision.value in ['cloud', 'hybrid']:
            if self.cloud_inference_fn and decomposed.cloud_query:
                cloud_response, cloud_latency = self.cloud_inference_fn(
                    decomposed.cloud_query if decomposed.has_sensitive else query
                )
        
        # Step 4: Aggregate responses
        aggregated = self.aggregator.aggregate(
            decomposed=decomposed,
            local_response=local_response,
            cloud_response=cloud_response,
            strategy=strategy,
        )
        
        # Step 5: Record privacy consumption
        self.privacy_manager.record_offloading(
            query=query,
            actual_action=1 if cloud_response else 0,
            sensitivity=decision.sensitivity,
        )
        
        return {
            'response': aggregated.final_response,
            'decision': decision.decision.value,
            'sensitivity': {
                'level': decision.sensitivity.level.name,
                'score': decision.sensitivity.score,
            },
            'privacy': {
                'risk': decision.privacy_risk,
                'budget_remaining': self.privacy_manager.budget.remaining,
            },
            'latency': {
                'local': local_latency,
                'cloud': cloud_latency,
                'total': max(local_latency, cloud_latency),  # Parallel
            },
            'aggregation': {
                'strategy': aggregated.strategy_used.value,
                'metadata': aggregated.metadata,
            },
        }


if __name__ == "__main__":
    print("🧪 Testing Response Aggregation\n")
    
    from .query_decomposer import QueryDecomposer, DecomposedQuery
    
    aggregator = ResponseAggregator()
    decomposer = QueryDecomposer()
    
    # Test case: Sentence-level decomposition
    query = "My password is secret123. What is machine learning?"
    decomposed = decomposer.decompose(query)
    
    local_response = "I've noted your password securely. Please use a stronger password for better security."
    cloud_response = "Machine learning is a subset of AI that enables systems to learn from data. [REDACTED_0] It uses algorithms to identify patterns."
    
    print("=" * 60)
    print("Test 1: Sentence-level decomposition")
    print(f"Query: {query}")
    print(f"\nLocal response: {local_response[:50]}...")
    print(f"Cloud response: {cloud_response[:50]}...")
    
    for strategy in AggregationStrategy:
        result = aggregator.aggregate(
            decomposed=decomposed,
            local_response=local_response,
            cloud_response=cloud_response,
            strategy=strategy,
        )
        print(f"\n📋 Strategy: {strategy.value}")
        print(f"   Result: {result.final_response[:100]}...")
    
    # Test case: Entity-level decomposition
    print("\n" + "=" * 60)
    print("Test 2: Entity-level decomposition")
    
    query2 = "Send email to john@example.com about the project"
    decomposed2 = decomposer.decompose(query2)
    
    local_response2 = "Email will be sent to the specified address."
    cloud_response2 = "I'll draft an email to [EMAIL_0] about the project. Here's a suggested template..."
    
    result2 = aggregator.aggregate(
        decomposed=decomposed2,
        local_response=local_response2,
        cloud_response=cloud_response2,
        strategy=AggregationStrategy.TEMPLATE,
    )
    
    print(f"Query: {query2}")
    print(f"Result: {result2.final_response}")
    
    print("\n" + "=" * 60)
    print("\n✅ Response Aggregation ready!")
