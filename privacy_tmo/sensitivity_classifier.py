"""
Multi-level Sensitivity Classifier for Privacy-TMO

Classifies user queries into three sensitivity levels:
- Level 0 (Public): General knowledge questions, safe for cloud
- Level 1 (Semi-sensitive): Context-dependent, hybrid processing
- Level 2 (Private): Contains PII, must stay on device

Uses a combination of:
1. ML-based classifier (DistilBERT fine-tuned)
2. Rule-based PII detection (Presidio / regex patterns)
3. Named Entity Recognition (NER)
"""

import re
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

if TYPE_CHECKING:
    from .image_sensitivity import ImageSensitivityClassifier


class SensitivityLevel(IntEnum):
    """Sensitivity levels for query classification"""
    PUBLIC = 0       # Safe for cloud processing
    SEMI_SENSITIVE = 1  # Hybrid processing recommended
    PRIVATE = 2      # Must stay on device


@dataclass
class SensitivityResult:
    """Result of sensitivity classification"""
    level: SensitivityLevel
    score: float  # 0.0 to 1.0
    confidence: float  # Model confidence
    detected_entities: List[Dict]  # Detected PII entities
    explanation: str  # Human-readable explanation
    
    def is_safe_for_cloud(self) -> bool:
        """Check if this query can be sent to cloud"""
        return self.level == SensitivityLevel.PUBLIC
    
    def requires_local_only(self) -> bool:
        """Check if this query must be processed locally"""
        return self.level == SensitivityLevel.PRIVATE


class PIIPatterns:
    """Regular expression patterns for PII detection"""
    
    # Korean patterns
    KOREAN_NAME = r'[가-힣]{2,4}(?:씨|님|에게|한테|의)'
    KOREAN_PHONE = r'(?:010|011|016|017|018|019)[-\s]?\d{3,4}[-\s]?\d{4}'
    KOREAN_RRN = r'\d{6}[-\s]?[1-4]\d{6}'  # 주민등록번호
    
    # English patterns
    EMAIL = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    PHONE_US = r'(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}'
    SSN = r'\d{3}[-\s]?\d{2}[-\s]?\d{4}'
    CREDIT_CARD = r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'
    
    # Common patterns
    IP_ADDRESS = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    DATE_OF_BIRTH = r'(?:19|20)\d{2}[-/년]\s*(?:0?[1-9]|1[0-2])[-/월]\s*(?:0?[1-9]|[12]\d|3[01])일?'
    
    # Sensitive keywords (high risk)
    HIGH_RISK_KEYWORDS = [
        # English
        'password', 'passwd', 'secret', 'api key', 'apikey', 'token',
        'credit card', 'bank account', 'ssn', 'social security',
        'private key', 'secret key', 'auth token',
        # Korean
        '비밀번호', '패스워드', '계좌번호', '카드번호', '주민번호',
        '신용카드', '비밀', '개인정보',
    ]
    
    # Medium risk keywords
    MEDIUM_RISK_KEYWORDS = [
        # English
        'address', 'phone number', 'email', 'birthday', 'salary',
        'medical', 'diagnosis', 'prescription', 'health record',
        # Korean  
        '주소', '전화번호', '이메일', '생년월일', '연봉', '급여',
        '진단', '처방', '병원', '의료기록',
    ]


class RuleBasedDetector:
    """Rule-based PII detector using regex patterns"""
    
    def __init__(self):
        self.patterns = {
            'EMAIL': (PIIPatterns.EMAIL, SensitivityLevel.SEMI_SENSITIVE),
            'PHONE_KR': (PIIPatterns.KOREAN_PHONE, SensitivityLevel.SEMI_SENSITIVE),
            'PHONE_US': (PIIPatterns.PHONE_US, SensitivityLevel.SEMI_SENSITIVE),
            'SSN': (PIIPatterns.SSN, SensitivityLevel.PRIVATE),
            'RRN': (PIIPatterns.KOREAN_RRN, SensitivityLevel.PRIVATE),
            'CREDIT_CARD': (PIIPatterns.CREDIT_CARD, SensitivityLevel.PRIVATE),
            'IP_ADDRESS': (PIIPatterns.IP_ADDRESS, SensitivityLevel.SEMI_SENSITIVE),
            'DOB': (PIIPatterns.DATE_OF_BIRTH, SensitivityLevel.SEMI_SENSITIVE),
            'KOREAN_NAME': (PIIPatterns.KOREAN_NAME, SensitivityLevel.SEMI_SENSITIVE),
        }
        
        self.high_risk_keywords = [kw.lower() for kw in PIIPatterns.HIGH_RISK_KEYWORDS]
        self.medium_risk_keywords = [kw.lower() for kw in PIIPatterns.MEDIUM_RISK_KEYWORDS]
    
    def detect(self, text: str) -> Tuple[SensitivityLevel, List[Dict], float]:
        """
        Detect PII using regex patterns and keywords
        
        Returns:
            (sensitivity_level, detected_entities, confidence_score)
        """
        detected = []
        max_level = SensitivityLevel.PUBLIC
        text_lower = text.lower()
        
        # Check regex patterns
        for entity_type, (pattern, level) in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                detected.append({
                    'type': entity_type,
                    'value': match if isinstance(match, str) else match[0],
                    'source': 'regex',
                    'level': level.value,
                })
                max_level = max(max_level, level)
        
        # Check high-risk keywords
        for keyword in self.high_risk_keywords:
            if keyword in text_lower:
                detected.append({
                    'type': 'HIGH_RISK_KEYWORD',
                    'value': keyword,
                    'source': 'keyword',
                    'level': SensitivityLevel.PRIVATE.value,
                })
                max_level = SensitivityLevel.PRIVATE
        
        # Check medium-risk keywords (only if not already private)
        if max_level < SensitivityLevel.PRIVATE:
            for keyword in self.medium_risk_keywords:
                if keyword in text_lower:
                    detected.append({
                        'type': 'MEDIUM_RISK_KEYWORD',
                        'value': keyword,
                        'source': 'keyword',
                        'level': SensitivityLevel.SEMI_SENSITIVE.value,
                    })
                    max_level = max(max_level, SensitivityLevel.SEMI_SENSITIVE)
        
        # Calculate confidence based on number of detections
        confidence = min(0.5 + len(detected) * 0.1, 1.0) if detected else 0.3
        
        return max_level, detected, confidence


class NERBasedDetector:
    """Named Entity Recognition based detector using transformers"""
    
    def __init__(self, model_name: str = "dslim/bert-base-NER", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        
        # Entity types that indicate PII
        self.sensitive_entity_types = {
            'PER': SensitivityLevel.SEMI_SENSITIVE,  # Person names
            'LOC': SensitivityLevel.PUBLIC,          # Locations (context-dependent)
            'ORG': SensitivityLevel.PUBLIC,          # Organizations
            'MISC': SensitivityLevel.PUBLIC,         # Miscellaneous
        }
    
    def load_model(self):
        """Lazy load the NER model"""
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=self.device if self.device != "cpu" else -1,
            )
    
    def detect(self, text: str) -> Tuple[SensitivityLevel, List[Dict], float]:
        """
        Detect named entities that might be PII
        
        Returns:
            (sensitivity_level, detected_entities, confidence_score)
        """
        self.load_model()
        
        detected = []
        max_level = SensitivityLevel.PUBLIC
        
        try:
            results = self.pipeline(text)
            
            for entity in results:
                entity_type = entity['entity_group']
                score = entity['score']
                
                # Only consider high-confidence predictions
                if score < 0.85:
                    continue
                
                level = self.sensitive_entity_types.get(
                    entity_type, 
                    SensitivityLevel.PUBLIC
                )
                
                # Person names are sensitive in certain contexts
                if entity_type == 'PER':
                    # Check if it's in a sensitive context
                    context_keywords = ['my', 'me', 'i am', 'contact', '내', '나의', '저의']
                    text_lower = text.lower()
                    if any(kw in text_lower for kw in context_keywords):
                        level = SensitivityLevel.SEMI_SENSITIVE
                
                detected.append({
                    'type': f'NER_{entity_type}',
                    'value': entity['word'],
                    'source': 'ner',
                    'level': level.value,
                    'score': score,
                })
                
                max_level = max(max_level, level)
            
            confidence = max([e.get('score', 0.5) for e in detected]) if detected else 0.3
            
        except Exception as e:
            print(f"⚠️ NER detection failed: {e}")
            confidence = 0.0
        
        return max_level, detected, confidence


class MLClassifier:
    """
    ML-based sensitivity classifier using fine-tuned DistilBERT
    
    Classifies queries directly into sensitivity levels
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
        # Default model for feature extraction
        self.base_model_name = "distilbert-base-uncased"
    
    def load_or_train(self, training_data: Optional[List[Dict]] = None):
        """Load pre-trained model or train new one"""
        if self.model_path and Path(self.model_path).exists():
            self._load_model()
        elif training_data:
            self._train_model(training_data)
        else:
            # Use feature extraction with simple classifier
            self._setup_feature_extractor()
    
    def _setup_feature_extractor(self):
        """Setup DistilBERT for feature extraction"""
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModel.from_pretrained(self.base_model_name)
        self.model.eval()
        
        # Simple linear classifier on top
        self.classifier = torch.nn.Linear(768, 3)  # 3 classes
        self.is_trained = False
    
    def _load_model(self):
        """Load pre-trained classifier"""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        self.is_trained = True
    
    def _train_model(self, training_data: List[Dict]):
        """Train sensitivity classifier"""
        from transformers import (
            AutoTokenizer, 
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from datasets import Dataset
        
        # Prepare dataset
        texts = [d['text'] for d in training_data]
        labels = [d['label'] for d in training_data]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        def tokenize(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=True,
                max_length=256,
            )
        
        dataset = Dataset.from_dict({'text': texts, 'label': labels})
        dataset = dataset.map(tokenize, batched=True)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,  # PUBLIC, SEMI_SENSITIVE, PRIVATE
        )
        
        # Training
        training_args = TrainingArguments(
            output_dir="./sensitivity_classifier",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            logging_steps=10,
            save_strategy="no",
            report_to="none",  # Privacy: no external reporting
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        trainer.train()
        self.is_trained = True
    
    def predict(self, text: str) -> Tuple[SensitivityLevel, float]:
        """
        Predict sensitivity level for a query
        
        Returns:
            (predicted_level, confidence)
        """
        if self.model is None:
            self._setup_feature_extractor()
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            max_length=256,
        )
        
        with torch.no_grad():
            if self.is_trained:
                outputs = self.model(**inputs)
                logits = outputs.logits
            else:
                # Feature extraction mode
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
                logits = self.classifier(features)
        
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()
        
        return SensitivityLevel(predicted_class), confidence


class SensitivityClassifier:
    """
    Main Sensitivity Classifier combining multiple detection methods
    
    Combines:
    1. Rule-based PII detection (fast, interpretable)
    2. NER-based entity detection (good for names, organizations)
    3. ML-based classification (learned patterns)
    """
    
    def __init__(
        self,
        use_ner: bool = True,
        use_ml: bool = True,
        ml_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.rule_detector = RuleBasedDetector()
        
        self.use_ner = use_ner
        if use_ner:
            self.ner_detector = NERBasedDetector(device=device)
        
        self.use_ml = use_ml
        if use_ml:
            self.ml_classifier = MLClassifier(model_path=ml_model_path, device=device)
    
    def classify(self, text: str) -> SensitivityResult:
        """
        Classify the sensitivity level of a query
        
        Combines results from multiple detectors with weighted voting
        
        Args:
            text: User query to classify
            
        Returns:
            SensitivityResult with level, score, and explanations
        """
        all_entities = []
        level_votes = []
        confidences = []
        
        # 1. Rule-based detection (always run - fast and reliable)
        rule_level, rule_entities, rule_conf = self.rule_detector.detect(text)
        all_entities.extend(rule_entities)
        level_votes.append((rule_level, rule_conf, 0.4))  # weight: 0.4
        confidences.append(rule_conf)
        
        # 2. NER-based detection
        if self.use_ner:
            try:
                ner_level, ner_entities, ner_conf = self.ner_detector.detect(text)
                all_entities.extend(ner_entities)
                level_votes.append((ner_level, ner_conf, 0.3))  # weight: 0.3
                confidences.append(ner_conf)
            except Exception as e:
                print(f"⚠️ NER failed: {e}")
        
        # 3. ML-based classification
        if self.use_ml:
            try:
                ml_level, ml_conf = self.ml_classifier.predict(text)
                level_votes.append((ml_level, ml_conf, 0.3))  # weight: 0.3
                confidences.append(ml_conf)
            except Exception as e:
                print(f"⚠️ ML classifier failed: {e}")
        
        # Combine votes (weighted by confidence and method weight)
        final_level = self._combine_votes(level_votes)
        
        # Calculate overall confidence
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Calculate sensitivity score (0.0 to 1.0)
        score = self._calculate_score(final_level, all_entities, avg_confidence)
        
        # Generate explanation
        explanation = self._generate_explanation(final_level, all_entities)
        
        return SensitivityResult(
            level=final_level,
            score=score,
            confidence=avg_confidence,
            detected_entities=all_entities,
            explanation=explanation,
        )
    
    def _combine_votes(
        self, 
        votes: List[Tuple[SensitivityLevel, float, float]]
    ) -> SensitivityLevel:
        """
        Combine votes from different detectors
        
        Uses weighted voting with a conservative approach:
        - If any detector says PRIVATE with high confidence, result is PRIVATE
        - Otherwise, weighted average
        """
        if not votes:
            return SensitivityLevel.PUBLIC
        
        # Check for high-confidence PRIVATE votes
        for level, conf, weight in votes:
            if level == SensitivityLevel.PRIVATE and conf > 0.7:
                return SensitivityLevel.PRIVATE
        
        # Weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for level, conf, method_weight in votes:
            vote_weight = conf * method_weight
            weighted_sum += level.value * vote_weight
            total_weight += vote_weight
        
        if total_weight > 0:
            avg_level = weighted_sum / total_weight
            # Round to nearest level
            return SensitivityLevel(round(avg_level))
        
        return SensitivityLevel.PUBLIC


@dataclass
class MultimodalSensitivityResult:
    """Result of multimodal sensitivity classification"""
    text: SensitivityResult
    images: Dict[int, SensitivityResult]
    combined_level: SensitivityLevel
    combined_score: float
    explanation: str


class MultimodalSensitivityClassifier:
    """
    Multimodal sensitivity classifier (text + images).
    
    This class aggregates text sensitivity with per-image sensitivity and
    provides a combined score/level for downstream routing or analysis.
    """
    
    def __init__(
        self,
        text_classifier: Optional[SensitivityClassifier] = None,
        image_classifier: Optional["ImageSensitivityClassifier"] = None,
        use_image_sensitivity: bool = False,
    ):
        self.text_classifier = text_classifier or SensitivityClassifier(
            use_ner=False,
            use_ml=False,
        )
        if image_classifier is not None:
            self.image_classifier = image_classifier
        elif use_image_sensitivity:
            from .image_sensitivity import ImageSensitivityClassifier
            self.image_classifier = ImageSensitivityClassifier(use_face_detection=True, use_ocr=False)
        else:
            self.image_classifier = None
    
    def classify(
        self,
        text: str,
        images: Optional[Dict[int, Any]] = None,
        simulate_images: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> MultimodalSensitivityResult:
        """
        Classify sensitivity for text + images.
        
        Args:
            text: Input query text
            images: Dict of {image_index: image_path_or_array}
            simulate_images: Use simulated image sensitivity when images unavailable
            context: Optional context for simulation
        """
        text_result = self.text_classifier.classify(text)
        image_results: Dict[int, SensitivityResult] = {}
        
        if images and self.image_classifier:
            for idx, img in images.items():
                image_results[idx] = self.image_classifier.classify(img)
        elif simulate_images:
            from .image_sensitivity import ImageSensitivityClassifier
            sim = ImageSensitivityClassifier(use_face_detection=False, use_ocr=False)
            ctx = context or {"prompt": text}
            for idx in [0, 1, 2]:
                image_results[idx] = sim.classify_simulated(idx, ctx)
        
        # Combine scores (average over available modalities)
        scores = [text_result.score] + [r.score for r in image_results.values()]
        combined_score = float(np.mean(scores)) if scores else 0.0
        
        if combined_score >= 0.7:
            combined_level = SensitivityLevel.PRIVATE
        elif combined_score >= 0.4:
            combined_level = SensitivityLevel.SEMI_SENSITIVE
        else:
            combined_level = SensitivityLevel.PUBLIC
        
        explanation = f"text={text_result.level.name}"
        if image_results:
            explanation += ", images=" + ",".join(
                f"{idx}:{res.level.name}" for idx, res in image_results.items()
            )
        
        return MultimodalSensitivityResult(
            text=text_result,
            images=image_results,
            combined_level=combined_level,
            combined_score=combined_score,
            explanation=explanation,
        )
    
    def _calculate_score(
        self, 
        level: SensitivityLevel,
        entities: List[Dict],
        confidence: float
    ) -> float:
        """Calculate overall sensitivity score (0.0 to 1.0)"""
        # Base score from level
        base_score = level.value / 2.0  # 0.0, 0.5, 1.0
        
        # Adjust based on number of entities
        entity_bonus = min(len(entities) * 0.05, 0.2)
        
        # Adjust based on entity severity
        has_private = any(e.get('level', 0) == 2 for e in entities)
        if has_private:
            entity_bonus += 0.2
        
        # Combine with confidence
        score = base_score + entity_bonus
        score = score * (0.5 + confidence * 0.5)  # Scale by confidence
        
        return min(max(score, 0.0), 1.0)
    
    def _generate_explanation(
        self, 
        level: SensitivityLevel, 
        entities: List[Dict]
    ) -> str:
        """Generate human-readable explanation"""
        level_names = {
            SensitivityLevel.PUBLIC: "Public (safe for cloud)",
            SensitivityLevel.SEMI_SENSITIVE: "Semi-sensitive (hybrid recommended)",
            SensitivityLevel.PRIVATE: "Private (local only)",
        }
        
        explanation = f"Classification: {level_names[level]}\n"
        
        if entities:
            explanation += f"Detected {len(entities)} sensitive element(s):\n"
            for entity in entities[:5]:  # Limit to 5
                explanation += f"  - {entity['type']}: '{entity['value']}'\n"
            if len(entities) > 5:
                explanation += f"  ... and {len(entities) - 5} more\n"
        else:
            explanation += "No sensitive elements detected."
        
        return explanation


def create_sample_training_data(output_path: str, num_samples: int = 500):
    """
    Create sample training data for sensitivity classifier
    
    Args:
        output_path: Path to save training data
        num_samples: Number of samples to generate
    """
    import json
    import random
    
    samples = []
    
    # Level 0: Public queries
    public_templates = [
        "What is {topic}?",
        "How do I {action}?",
        "Explain {concept} in simple terms",
        "What are the benefits of {topic}?",
        "Compare {thing1} and {thing2}",
    ]
    public_topics = [
        "machine learning", "Python programming", "climate change",
        "the solar system", "artificial intelligence", "blockchain",
    ]
    
    # Level 1: Semi-sensitive queries  
    semi_sensitive_templates = [
        "My address is {address}, can you find nearby {place}?",
        "I was born on {date}, what's my zodiac sign?",
        "Send an email to {name} about {topic}",
        "My phone number is {phone}, add it to {service}",
    ]
    
    # Level 2: Private queries
    private_templates = [
        "My password for {service} is {password}",
        "My SSN is {ssn}, check my {service}",
        "My credit card number is {cc}, pay for {item}",
        "Here's my bank account: {account}, transfer {amount}",
        "비밀번호는 {password}입니다",
        "내 주민번호 {rrn}로 조회해줘",
    ]
    
    # Generate samples
    for _ in range(num_samples // 3):
        # Public
        template = random.choice(public_templates)
        topic = random.choice(public_topics)
        text = template.format(
            topic=topic, action="learn " + topic, concept=topic,
            thing1=topic, thing2=random.choice(public_topics)
        )
        samples.append({"text": text, "label": 0})
        
        # Semi-sensitive
        template = random.choice(semi_sensitive_templates)
        text = template.format(
            address="123 Main St", date="1990-05-15",
            name="John", topic="meeting", phone="555-1234",
            place="restaurants", service="contacts"
        )
        samples.append({"text": text, "label": 1})
        
        # Private
        template = random.choice(private_templates)
        text = template.format(
            service="email", password="secret123",
            ssn="123-45-6789", cc="4111-1111-1111-1111",
            account="123456789", amount="$100", item="subscription",
            rrn="901234-1234567"
        )
        samples.append({"text": text, "label": 2})
    
    random.shuffle(samples)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training data saved to: {output_path}")
    print(f"   Total samples: {len(samples)}")
    return output_path


if __name__ == "__main__":
    print("🧪 Testing Sensitivity Classifier\n")
    
    # Initialize classifier (NER and ML disabled for quick test)
    classifier = SensitivityClassifier(use_ner=False, use_ml=False)
    
    # Test queries
    test_queries = [
        "What is machine learning?",  # Public
        "How do I sort a list in Python?",  # Public
        "My email is john@example.com, send a reminder",  # Semi-sensitive
        "Call me at 010-1234-5678",  # Semi-sensitive
        "My password is secret123",  # Private
        "내 비밀번호는 qwerty야",  # Private (Korean)
        "My SSN is 123-45-6789",  # Private
        "Transfer $1000 to account 123456789",  # Private
    ]
    
    print("=" * 60)
    for query in test_queries:
        result = classifier.classify(query)
        level_emoji = ["🟢", "🟡", "🔴"][result.level]
        print(f"\nQuery: {query[:50]}...")
        print(f"  {level_emoji} Level: {result.level.name}")
        print(f"  Score: {result.score:.2f}, Confidence: {result.confidence:.2f}")
        if result.detected_entities:
            print(f"  Detected: {[e['type'] for e in result.detected_entities]}")
    print("=" * 60)
    
    print("\n✅ Sensitivity Classifier ready!")
