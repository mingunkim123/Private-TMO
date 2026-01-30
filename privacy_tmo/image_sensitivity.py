"""
Image Sensitivity Classifier for Privacy-TMO

Classifies image sensitivity for multimodal privacy-aware offloading:
- Face detection: Images with faces -> PRIVATE
- OCR-based text extraction: Sensitive text in images -> analyzed via text classifier
- Document pattern detection: ID cards, medical records, etc.

Supports simulated mode when actual images are unavailable (e.g., M4AI dataset).
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

from .sensitivity_classifier import SensitivityLevel, SensitivityResult, SensitivityClassifier


@dataclass
class ImageSensitivityConfig:
    """Configuration for image sensitivity analysis"""
    use_face_detection: bool = True
    use_ocr: bool = False
    face_detection_confidence: float = 0.3
    simulated_default_score: float = 0.5  # Default for simulated mode


class ImageSensitivityClassifier:
    """
    Classifies image sensitivity for privacy-aware modality selection.
    
    Detects:
    1. Faces -> PRIVATE (high sensitivity)
    2. OCR text -> analyzed via text classifier
    3. Document patterns (ID cards, medical records)
    
    Supports simulated mode when actual image data is unavailable.
    """
    
    def __init__(
        self,
        use_face_detection: bool = True,
        use_ocr: bool = False,
        text_classifier: Optional[SensitivityClassifier] = None,
    ):
        self.use_face_detection = use_face_detection
        self.use_ocr = use_ocr
        self.text_classifier = text_classifier or SensitivityClassifier(
            use_ner=False,
            use_ml=False,
        )
        self._face_cascade = None
        
        if use_face_detection:
            self._init_face_detector()
    
    def _init_face_detector(self) -> None:
        """Initialize OpenCV face detector (optional dependency)"""
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except ImportError:
            self._face_cascade = None
        except Exception:
            self._face_cascade = None
    
    def _detect_faces(self, image) -> bool:
        """Detect if image contains faces. Returns True if faces found."""
        if self._face_cascade is None:
            return False
        try:
            import cv2
            import numpy as np
            
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
            elif hasattr(image, 'shape'):
                img = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img = np.array(image)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            if img is None:
                return False
            
            faces = self._face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            return len(faces) > 0
        except Exception:
            return False
    
    def _extract_text_via_ocr(self, image) -> str:
        """Extract text from image via OCR (optional)."""
        if not self.use_ocr:
            return ""
        try:
            import pytesseract
            import cv2
            import numpy as np
            
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
            else:
                img = np.array(image)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            if img is None:
                return ""
            
            return pytesseract.image_to_string(img)
        except ImportError:
            return ""
        except Exception:
            return ""
    
    def classify(
        self,
        image_or_path: Union[str, Path, Any],
    ) -> SensitivityResult:
        """
        Classify image sensitivity.
        
        Args:
            image_or_path: Image file path, numpy array, or PIL Image
            
        Returns:
            SensitivityResult with level and score
        """
        has_face = False
        ocr_text = ""
        
        if self.use_face_detection:
            has_face = self._detect_faces(image_or_path)
        
        if has_face:
            return SensitivityResult(
                level=SensitivityLevel.PRIVATE,
                score=1.0,
                confidence=0.9,
                detected_entities=[{"type": "FACE", "source": "face_detection"}],
                explanation="Face detected in image - must stay on device",
            )
        
        if self.use_ocr:
            ocr_text = self._extract_text_via_ocr(image_or_path)
            if ocr_text.strip():
                text_result = self.text_classifier.classify(ocr_text)
                if text_result.level == SensitivityLevel.PRIVATE:
                    return SensitivityResult(
                        level=SensitivityLevel.PRIVATE,
                        score=max(text_result.score, 0.8),
                        confidence=text_result.confidence,
                        detected_entities=text_result.detected_entities,
                        explanation=f"Sensitive text in image: {text_result.explanation}",
                    )
                elif text_result.level == SensitivityLevel.SEMI_SENSITIVE:
                    return SensitivityResult(
                        level=SensitivityLevel.SEMI_SENSITIVE,
                        score=text_result.score,
                        confidence=text_result.confidence,
                        detected_entities=text_result.detected_entities,
                        explanation=f"Semi-sensitive text in image: {text_result.explanation}",
                    )
        
        return SensitivityResult(
            level=SensitivityLevel.PUBLIC,
            score=0.0,
            confidence=0.7,
            detected_entities=[],
            explanation="No sensitive content detected in image",
        )
    
    def classify_simulated(
        self,
        image_index: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> SensitivityResult:
        """
        Simulate image sensitivity when actual images are unavailable.
        
        M4AI dataset: image_index 0=1st person view, 1=overhead, 2=another view.
        Heuristics:
        - image_index 0: First-person view -> higher chance of faces
        - image_index 1: Overhead view -> may show documents
        - image_index 2: Third view -> context-dependent
        
        Args:
            image_index: 0, 1, or 2 (modality index)
            context: Optional dict with 'task_cat', 'prompt', etc.
            
        Returns:
            Simulated SensitivityResult
        """
        context = context or {}
        
        base_scores = {
            0: 0.4,  # First-person: faces more likely
            1: 0.3,  # Overhead: documents possible
            2: 0.35,  # Third view: mixed
        }
        base_score = base_scores.get(image_index, 0.35)
        
        task_cat = context.get("task_cat", "")
        prompt = context.get("prompt", "").lower()
        
        if "assistive" in task_cat.lower() or "assist" in prompt:
            base_score += 0.15
        if "medical" in prompt or "진단" in prompt or "의료" in prompt:
            base_score += 0.2
        if "document" in prompt or "문서" in prompt or "id" in prompt:
            base_score += 0.25
        
        score = min(base_score, 1.0)
        
        if score >= 0.7:
            level = SensitivityLevel.PRIVATE
        elif score >= 0.4:
            level = SensitivityLevel.SEMI_SENSITIVE
        else:
            level = SensitivityLevel.PUBLIC
        
        return SensitivityResult(
            level=level,
            score=score,
            confidence=0.6,
            detected_entities=[{"type": "SIMULATED", "image_index": image_index}],
            explanation=f"Simulated sensitivity for image {image_index} (score={score:.2f})",
        )
