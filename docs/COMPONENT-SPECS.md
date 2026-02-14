# Component Specifications

## 🎯 Component 1: Bubble Detector

### Overview
The Bubble Detector component identifies speech bubbles, thought bubbles, and text regions within manga panels using a combination of traditional computer vision techniques and machine learning models.

### Input/Output Specification

**Input**:
```python
class DetectionInput:
    image: np.ndarray          # RGB image data (H, W, 3)
    detection_mode: str        # "fast", "balanced", "accurate"
    confidence_threshold: float # 0.0-1.0, minimum confidence for detection
    min_bubble_size: int       # Minimum pixel area for valid bubble
    max_bubble_size: int       # Maximum pixel area for valid bubble
```

**Output**:
```python
class BubbleDetection:
    bubbles: List[BubbleRegion]
    processing_time: float
    confidence_scores: List[float]

class BubbleRegion:
    contour: np.ndarray        # OpenCV contour points
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    mask: np.ndarray          # Binary mask of bubble interior
    bubble_type: str          # "speech", "thought", "narration", "sfx"
    confidence: float         # Detection confidence 0.0-1.0
    text_direction: str       # "horizontal", "vertical", "mixed"
```

### Libraries and Models

#### Primary Technologies
- **OpenCV 4.8+**: Contour detection and image processing
- **scikit-image**: Advanced image analysis and morphological operations
- **NumPy**: Numerical operations and array processing

#### Machine Learning Models

1. **Custom Bubble Detection CNN**:
   ```
   Model Architecture:
   Input: 512x512 RGB patch
   ├── Conv Block 1: 32 filters, 3x3, ReLU
   ├── Conv Block 2: 64 filters, 3x3, ReLU  
   ├── Conv Block 3: 128 filters, 3x3, ReLU
   ├── Global Average Pooling
   └── Dense Layer: 4 classes (speech/thought/narration/sfx)
   
   Training Dataset: 50,000+ annotated manga panels
   Performance: 94% accuracy on test set
   Size: 15MB
   ```

2. **Fallback Algorithm** (Traditional CV):
   ```python
   def traditional_detection(image):
       # Edge detection
       edges = cv2.Canny(image, 50, 150)
       
       # Contour finding
       contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
       # Shape analysis
       bubbles = []
       for contour in contours:
           # Filter by area
           area = cv2.contourArea(contour)
           if area < min_area or area > max_area:
               continue
               
           # Analyze shape characteristics
           convexity = cv2.isContourConvex(contour)
           aspect_ratio = get_aspect_ratio(contour)
           circularity = get_circularity(contour)
           
           # Classify as bubble if meets criteria
           if is_bubble_like(convexity, aspect_ratio, circularity):
               bubbles.append(create_bubble_region(contour))
       
       return bubbles
   ```

### Configuration Options

```python
class BubbleDetectorConfig:
    # Detection Parameters
    confidence_threshold: float = 0.7    # Minimum confidence for ML detection
    use_ml_model: bool = True           # Use neural network vs traditional CV
    
    # Size Filtering
    min_bubble_area: int = 500          # Minimum pixels for valid bubble
    max_bubble_area: int = 50000        # Maximum pixels for valid bubble
    min_aspect_ratio: float = 0.2       # Minimum width/height ratio
    max_aspect_ratio: float = 5.0       # Maximum width/height ratio
    
    # Shape Analysis
    min_circularity: float = 0.3        # How round bubble must be (0-1)
    max_convexity_defects: int = 5      # Maximum allowed shape irregularities
    
    # Text Direction Detection
    enable_direction_detection: bool = True
    vertical_text_threshold: float = 0.6  # Confidence for vertical text
    
    # Performance
    max_processing_time: float = 30.0   # Timeout in seconds
    use_gpu: bool = True               # GPU acceleration if available
    batch_size: int = 4                # Batch size for ML inference
```

### Error Handling

**Exception Types**:
```python
class BubbleDetectionError(Exception):
    pass

class ImageProcessingError(BubbleDetectionError):
    """Image format or size issues"""
    pass

class ModelLoadError(BubbleDetectionError):
    """ML model loading failures"""
    pass

class TimeoutError(BubbleDetectionError):
    """Processing exceeded time limit"""
    pass
```

**Recovery Strategies**:
- **Model Load Failure** → Fall back to traditional CV methods
- **GPU Memory Error** → Retry with CPU processing
- **Processing Timeout** → Return partial results with warning
- **Invalid Image** → Attempt format conversion or request user action

### Edge Cases

#### Complex Layouts
- **Overlapping Bubbles**: Use depth analysis and bubble priority
- **Irregular Shapes**: Adjust convexity and circularity thresholds
- **Connected Bubbles**: Implement watershed segmentation
- **Nested Bubbles**: Hierarchical contour analysis

#### Art Style Variations
- **Realistic vs Cartoon**: Adaptive thresholds based on art style detection
- **Black & White vs Color**: Different preprocessing pipelines
- **Hand-drawn vs Digital**: Noise filtering and edge smoothing

#### Text Placement Edge Cases
```python
# Outside bubble text (signs, narration)
def detect_outside_text(image, bubble_regions):
    # Mask out detected bubbles
    masked_image = mask_bubble_regions(image, bubble_regions)
    
    # Detect remaining text regions
    text_regions = detect_text_regions(masked_image)
    
    # Classify as signs, narration, or SFX
    classified_regions = classify_text_type(text_regions)
    
    return classified_regions

# Vertical text handling
def detect_vertical_text(bubble_region):
    # Analyze text orientation within bubble
    text_lines = detect_text_lines(bubble_region)
    
    orientation_scores = []
    for line in text_lines:
        h_score = calculate_horizontal_score(line)
        v_score = calculate_vertical_score(line)
        orientation_scores.append((h_score, v_score))
    
    # Determine overall text direction
    if mean([v for h, v in orientation_scores]) > threshold:
        return "vertical"
    return "horizontal"
```

---

## 📖 Component 2: OCR Engine

### Overview
Multi-model OCR system optimized for manga text extraction with automatic language detection and model selection.

### Input/Output Specification

**Input**:
```python
class OCRInput:
    image_region: np.ndarray   # Cropped text region (H, W, 3)
    language_hint: str         # "auto", "ja", "zh", "ko", "en"
    text_direction: str        # "horizontal", "vertical", "auto"
    confidence_threshold: float # Minimum confidence for acceptance
    preprocessing_level: str   # "none", "basic", "aggressive"
```

**Output**:
```python
class OCRResult:
    text: str                  # Extracted text content
    confidence: float          # Overall confidence score 0.0-1.0
    character_confidences: List[float]  # Per-character confidence
    language_detected: str     # Detected language code
    model_used: str           # Which OCR model was used
    processing_time: float    # Time taken for extraction
    bounding_boxes: List[BoundingBox]  # Character-level positions

class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    character: str
    confidence: float
```

### Libraries and Models

#### 1. manga-ocr (Primary for Japanese)

**Installation**: `pip install manga-ocr`

**Usage**:
```python
from manga_ocr import MangaOcr

class MangaOCREngine:
    def __init__(self):
        self.mocr = MangaOcr()
    
    def extract_text(self, image_region):
        # manga-ocr expects PIL Image
        pil_image = Image.fromarray(image_region)
        
        # Extract text (returns string only)
        text = self.mocr(pil_image)
        
        # Estimate confidence (manga-ocr doesn't provide this)
        confidence = estimate_confidence(text, image_region)
        
        return OCRResult(
            text=text,
            confidence=confidence,
            model_used="manga-ocr",
            language_detected="ja"
        )
```

**Strengths**:
- Excellent accuracy on manga-style Japanese text
- Handles mixed hiragana/katakana/kanji
- Trained specifically on manga fonts
- Good with stylized and decorative text

**Limitations**:
- Japanese only
- No confidence scores
- No character-level bounding boxes
- Fixed model (not customizable)

#### 2. PaddleOCR (CJK Languages)

**Installation**: `pip install paddleocr`

**Usage**:
```python
from paddleocr import PaddleOCR

class PaddleOCREngine:
    def __init__(self):
        # Initialize for multiple languages
        self.ocr_ja = PaddleOCR(use_angle_cls=True, lang='japan')
        self.ocr_ch = PaddleOCR(use_angle_cls=True, lang='ch')
        self.ocr_ko = PaddleOCR(use_angle_cls=True, lang='korean')
    
    def extract_text(self, image_region, language_hint="auto"):
        # Select appropriate model
        if language_hint == "ja":
            ocr_engine = self.ocr_ja
        elif language_hint == "zh":
            ocr_engine = self.ocr_ch
        elif language_hint == "ko":
            ocr_engine = self.ocr_ko
        else:
            # Auto-detect language
            ocr_engine = self.detect_and_select_engine(image_region)
        
        # Perform OCR
        result = ocr_engine.ocr(image_region, cls=True)
        
        # Parse result
        return self.parse_paddle_result(result)
    
    def parse_paddle_result(self, paddle_result):
        text_lines = []
        confidences = []
        bounding_boxes = []
        
        for line in paddle_result:
            for word_info in line:
                bbox, (text, confidence) = word_info
                text_lines.append(text)
                confidences.append(confidence)
                bounding_boxes.append(BoundingBox.from_paddle_bbox(bbox))
        
        return OCRResult(
            text=" ".join(text_lines),
            confidence=np.mean(confidences),
            character_confidences=confidences,
            bounding_boxes=bounding_boxes,
            model_used="paddleocr"
        )
```

#### 3. Tesseract (Fallback)

**Installation**: `pip install pytesseract`

**Usage**:
```python
import pytesseract
from pytesseract import Output

class TesseractEngine:
    def __init__(self):
        # Configure Tesseract
        self.config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    
    def extract_text(self, image_region, language_hint="eng"):
        # Preprocess image for better OCR
        processed_image = self.preprocess_for_tesseract(image_region)
        
        # Extract text with detailed information
        data = pytesseract.image_to_data(
            processed_image, 
            output_type=Output.DICT,
            lang=language_hint,
            config=self.config
        )
        
        # Filter and aggregate results
        return self.parse_tesseract_result(data)
```

### Configuration Options

```python
class OCRConfig:
    # Model Selection
    preferred_models: List[str] = ["manga-ocr", "paddleocr", "tesseract"]
    auto_language_detection: bool = True
    fallback_on_low_confidence: bool = True
    
    # Quality Thresholds
    min_confidence_threshold: float = 0.7
    character_confidence_threshold: float = 0.5
    
    # Preprocessing Options
    enable_denoising: bool = True
    enable_deskewing: bool = True
    enable_contrast_enhancement: bool = True
    upscale_small_text: bool = True
    upscale_factor: float = 2.0
    
    # Language Settings
    supported_languages: List[str] = ["ja", "zh-cn", "zh-tw", "ko", "en"]
    language_detection_confidence: float = 0.8
    
    # Performance
    max_processing_time: float = 15.0
    use_gpu: bool = True
    batch_processing: bool = True
```

### Error Handling

**Exception Handling**:
```python
class OCRError(Exception):
    pass

class ModelNotAvailableError(OCRError):
    """Requested OCR model not installed/available"""
    pass

class LanguageNotSupportedError(OCRError):
    """Language not supported by available models"""
    pass

class LowConfidenceError(OCRError):
    """OCR confidence below acceptable threshold"""
    pass

def robust_ocr_pipeline(image_region, config):
    """Robust OCR with multiple fallback strategies"""
    
    models_to_try = config.preferred_models.copy()
    last_error = None
    
    for model_name in models_to_try:
        try:
            # Attempt OCR with current model
            result = ocr_engines[model_name].extract_text(image_region)
            
            # Check confidence threshold
            if result.confidence >= config.min_confidence_threshold:
                return result
            else:
                # Low confidence - try next model
                last_error = LowConfidenceError(f"Confidence {result.confidence} below threshold")
                continue
                
        except ModelNotAvailableError:
            # Model not installed - try next
            continue
        except Exception as e:
            last_error = e
            continue
    
    # All models failed
    if last_error:
        raise last_error
    else:
        raise OCRError("No OCR models available")
```

### Edge Cases

#### Vertical Text Handling
```python
def handle_vertical_text(image_region, detected_direction):
    if detected_direction == "vertical":
        # Rotate image for horizontal-optimized OCR models
        if model_name == "tesseract":
            # Rotate 90 degrees for Tesseract
            rotated = cv2.rotate(image_region, cv2.ROTATE_90_CLOCKWISE)
            result = ocr_engine.extract_text(rotated)
            # Restore original text orientation in result
            return restore_vertical_layout(result)
        else:
            # PaddleOCR and manga-ocr handle vertical text natively
            return ocr_engine.extract_text(image_region)
    
    return ocr_engine.extract_text(image_region)
```

#### Stylized and Decorative Text
```python
def handle_stylized_text(image_region):
    """Special preprocessing for decorative manga fonts"""
    
    # Detect if text is stylized (outlined, shadowed, etc.)
    style_features = analyze_text_style(image_region)
    
    if style_features.has_outline:
        # Remove outline to improve OCR
        cleaned_image = remove_text_outline(image_region)
    elif style_features.has_shadow:
        # Remove drop shadow
        cleaned_image = remove_text_shadow(image_region)
    else:
        cleaned_image = image_region
    
    # Enhance contrast for stylized text
    enhanced_image = enhance_contrast(cleaned_image)
    
    return enhanced_image

def remove_text_outline(image):
    """Remove outline from outlined text"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Morphological operations to remove thin outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # Threshold to get clean text
    _, thresh = cv2.threshold(opened, 127, 255, cv2.THRESH_BINARY)
    
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
```

#### Mixed Language Text
```python
def handle_mixed_language_text(image_region):
    """Handle text with multiple languages in one bubble"""
    
    # Segment into individual character/word regions
    char_regions = segment_characters(image_region)
    
    results = []
    for region in char_regions:
        # Detect language for each segment
        detected_lang = detect_language_visual(region)
        
        # Use appropriate OCR model
        if detected_lang == "ja":
            result = manga_ocr_engine.extract_text(region)
        elif detected_lang in ["zh", "ko"]:
            result = paddle_ocr_engine.extract_text(region, detected_lang)
        else:
            result = tesseract_engine.extract_text(region, detected_lang)
        
        results.append(result)
    
    # Combine results maintaining spatial order
    return combine_segmented_results(results, char_regions)
```

#### Furigana (Ruby Text) Handling
```python
def detect_and_handle_furigana(image_region):
    """Detect and properly handle furigana (small reading text above kanji)"""
    
    # Detect text size variations
    text_lines = detect_text_lines(image_region)
    
    furigana_lines = []
    main_text_lines = []
    
    for line in text_lines:
        avg_char_height = calculate_average_character_height(line)
        
        if avg_char_height < FURIGANA_HEIGHT_THRESHOLD:
            furigana_lines.append(line)
        else:
            main_text_lines.append(line)
    
    # Process each type separately
    main_text = ""
    furigana_text = ""
    
    if main_text_lines:
        main_text = manga_ocr_engine.extract_text(combine_regions(main_text_lines))
    
    if furigana_lines:
        furigana_text = manga_ocr_engine.extract_text(combine_regions(furigana_lines))
    
    # Combine results with proper formatting
    if furigana_text:
        return f"{main_text}({furigana_text})"
    return main_text
```

---

## 🌐 Component 3: Translator Engine

### Overview
Unified translation interface supporting multiple translation backends with context awareness and manga-specific terminology handling.

### Input/Output Specification

**Input**:
```python
class TranslationInput:
    source_text: str           # Text to translate
    source_language: str       # Source language code
    target_language: str       # Target language code ("en" default)
    context_type: str          # "dialogue", "narration", "sfx", "sign"
    character_name: str        # Character speaking (for context)
    previous_translations: List[str]  # Previous bubble translations for context
    terminology_dict: Dict[str, str]  # Custom term translations
    preserve_formatting: bool  # Keep special formatting (**, etc.)
```

**Output**:
```python
class TranslationResult:
    translated_text: str       # Final translated text
    confidence: float          # Translation confidence 0.0-1.0
    engine_used: str          # Which translation engine was used
    processing_time: float    # Time taken for translation
    alternatives: List[str]   # Alternative translations (if available)
    detected_context: str     # Inferred context (formal/casual/emotional)
    terminology_applied: List[Tuple[str, str]]  # Applied custom terms
```

### Translation Engines

#### 1. OpenAI GPT (Primary for Context-Aware Translation)

**Installation**: `pip install openai`

**Usage**:
```python
import openai
from typing import List, Dict

class OpenAITranslatorEngine:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.manga_prompt_template = """
You are a professional manga translator. Translate the following {context_type} from {source_lang} to {target_lang}.

Context:
- Character: {character_name}
- Scene context: {scene_context}
- Previous dialogue: {previous_context}

Custom terminology:
{terminology}

Source text: "{source_text}"

Requirements:
- Maintain the tone and personality of the character
- Use appropriate English that fits the manga style
- Keep cultural references when they add value
- Use natural, flowing English dialogue
- Preserve any sound effects in a way that English readers understand

Translation:
"""
    
    def translate(self, input_data: TranslationInput) -> TranslationResult:
        # Build context-aware prompt
        prompt = self.build_manga_prompt(input_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert manga translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=200
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Calculate confidence based on response quality
            confidence = self.calculate_confidence(response)
            
            return TranslationResult(
                translated_text=translated_text,
                confidence=confidence,
                engine_used=f"openai-{self.model}",
                processing_time=response.usage.total_tokens / 1000  # Approximate
            )
            
        except Exception as e:
            raise TranslationError(f"OpenAI translation failed: {str(e)}")
    
    def build_manga_prompt(self, input_data: TranslationInput) -> str:
        # Apply custom terminology
        text_with_terms = self.apply_terminology(
            input_data.source_text, 
            input_data.terminology_dict
        )
        
        # Build context
        context_parts = []
        if input_data.previous_translations:
            context_parts.append(f"Previous: {' → '.join(input_data.previous_translations[-2:])}")
        
        return self.manga_prompt_template.format(
            context_type=input_data.context_type,
            source_lang=input_data.source_language,
            target_lang=input_data.target_language,
            character_name=input_data.character_name or "Unknown",
            scene_context=" ".join(context_parts),
            previous_context=" → ".join(input_data.previous_translations[-3:]),
            terminology=self.format_terminology(input_data.terminology_dict),
            source_text=text_with_terms
        )
```

#### 2. DeepL API (High-Quality Neural Translation)

**Installation**: `pip install deepl`

**Usage**:
```python
import deepl

class DeepLTranslatorEngine:
    def __init__(self, auth_key: str):
        self.translator = deepl.Translator(auth_key)
    
    def translate(self, input_data: TranslationInput) -> TranslationResult:
        # Apply preprocessing for manga text
        preprocessed_text = self.preprocess_manga_text(input_data.source_text)
        
        try:
            # DeepL translation
            result = self.translator.translate_text(
                preprocessed_text,
                source_lang=input_data.source_language.upper(),
                target_lang=input_data.target_language.upper(),
                formality="less"  # More casual for manga dialogue
            )
            
            # Post-process for manga style
            final_text = self.postprocess_manga_text(
                result.text, 
                input_data.context_type
            )
            
            return TranslationResult(
                translated_text=final_text,
                confidence=0.85,  # DeepL generally high quality
                engine_used="deepl",
                processing_time=result.billed_characters / 1000
            )
            
        except Exception as e:
            raise TranslationError(f"DeepL translation failed: {str(e)}")
    
    def preprocess_manga_text(self, text: str) -> str:
        """Prepare manga text for DeepL translation"""
        # Handle common manga text patterns
        text = re.sub(r'！+', '!', text)  # Multiple exclamation marks
        text = re.sub(r'？+', '?', text)  # Multiple question marks
        text = re.sub(r'…+', '...', text)  # Ellipsis normalization
        
        # Preserve sound effects markers
        text = re.sub(r'(\*[^*]+\*)', r'SFX:\1', text)
        
        return text
    
    def postprocess_manga_text(self, translated_text: str, context_type: str) -> str:
        """Post-process DeepL output for manga style"""
        # Restore sound effects
        translated_text = re.sub(r'SFX:(\*[^*]+\*)', r'\1', translated_text)
        
        # Adjust formality based on context
        if context_type == "dialogue":
            # Make dialogue more natural/casual
            translated_text = self.make_dialogue_casual(translated_text)
        
        return translated_text
```

#### 3. Argos Translate (Offline)

**Installation**: `pip install argostranslate`

**Usage**:
```python
import argostranslate.package
import argostranslate.translate

class ArgosTranslatorEngine:
    def __init__(self):
        # Download required language packages on first use
        self.ensure_language_packages()
    
    def ensure_language_packages(self):
        """Ensure required language packages are installed"""
        available_packages = argostranslate.package.get_available_packages()
        
        required_pairs = [
            ("ja", "en"),
            ("zh", "en"), 
            ("ko", "en")
        ]
        
        for from_code, to_code in required_pairs:
            package = next(
                (p for p in available_packages 
                 if p.from_code == from_code and p.to_code == to_code),
                None
            )
            if package:
                argostranslate.package.install_from_path(package.download())
    
    def translate(self, input_data: TranslationInput) -> TranslationResult:
        try:
            # Perform offline translation
            translated_text = argostranslate.translate.translate(
                input_data.source_text,
                input_data.source_language,
                input_data.target_language
            )
            
            # Apply manga-specific post-processing
            final_text = self.apply_manga_style(translated_text, input_data)
            
            return TranslationResult(
                translated_text=final_text,
                confidence=0.7,  # Generally lower quality than cloud APIs
                engine_used="argos-translate",
                processing_time=0.1  # Very fast offline processing
            )
            
        except Exception as e:
            raise TranslationError(f"Argos translation failed: {str(e)}")
```

### Configuration Options

```python
class TranslatorConfig:
    # Engine Selection
    primary_engine: str = "openai"
    fallback_engines: List[str] = ["deepl", "argos"]
    engine_selection_strategy: str = "quality_first"  # "speed_first", "cost_first"
    
    # API Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    deepl_auth_key: str = ""
    enable_offline_fallback: bool = True
    
    # Quality Settings
    min_confidence_threshold: float = 0.6
    request_alternatives: bool = True
    max_alternatives: int = 3
    
    # Context Handling
    enable_context_awareness: bool = True
    context_window_size: int = 3  # Previous translations to include
    character_consistency: bool = True
    
    # Terminology
    terminology_dictionaries: List[str] = []  # Paths to custom dictionaries
    auto_detect_names: bool = True
    preserve_honorifics: bool = True
    
    # Performance
    max_translation_time: float = 30.0
    batch_processing: bool = True
    cache_translations: bool = True
```

### Error Handling

```python
class TranslationError(Exception):
    pass

class APIKeyError(TranslationError):
    """Invalid or missing API key"""
    pass

class QuotaExceededError(TranslationError):
    """Translation service quota exceeded"""
    pass

class LanguageNotSupportedError(TranslationError):
    """Language pair not supported by engine"""
    pass

class TranslationQualityError(TranslationError):
    """Translation quality below acceptable threshold"""
    pass

def robust_translation_pipeline(input_data: TranslationInput, config: TranslatorConfig):
    """Robust translation with fallback strategies"""
    
    engines_to_try = [config.primary_engine] + config.fallback_engines
    last_error = None
    
    for engine_name in engines_to_try:
        try:
            engine = translation_engines[engine_name]
            result = engine.translate(input_data)
            
            # Quality check
            if result.confidence >= config.min_confidence_threshold:
                return result
            else:
                # Try next engine for better quality
                continue
                
        except APIKeyError:
            # Missing API key - skip this engine
            continue
        except QuotaExceededError:
            # Quota exceeded - try different engine
            continue
        except Exception as e:
            last_error = e
            continue
    
    # All engines failed
    if last_error:
        raise last_error
    else:
        raise TranslationError("No translation engines available")
```

### Edge Cases

#### Context-Aware Character Dialogue
```python
def maintain_character_consistency(translation_input: TranslationInput) -> TranslationInput:
    """Ensure character speech patterns are maintained"""
    
    character = translation_input.character_name
    if not character:
        return translation_input
    
    # Load character speech pattern database
    character_patterns = load_character_patterns(character)
    
    if character_patterns:
        # Adjust translation prompt to include character traits
        original_context = translation_input.context_type
        enhanced_context = f"{original_context} (Character: {character} - {character_patterns.speech_style})"
        
        translation_input.context_type = enhanced_context
    
    return translation_input

def load_character_patterns(character_name: str) -> Optional[CharacterPattern]:
    """Load known character speech patterns"""
    # Example patterns for common archetypes
    patterns = {
        "tsundere": CharacterPattern(
            speech_style="Initially cold/hostile, occasionally showing warmth",
            common_phrases=["It's not like I...", "Hmph!", "Whatever!"],
            formality_level="casual_defensive"
        ),
        "kuudere": CharacterPattern(
            speech_style="Cool, calm, emotionally detached",
            common_phrases=["I see.", "How troublesome.", "If you insist."],
            formality_level="formal_distant"
        ),
        # ... more patterns
    }
    
    # Try to match character name or look up in series database
    character_type = detect_character_archetype(character_name)
    return patterns.get(character_type)
```

#### Sound Effects (SFX) Translation
```python
def handle_sound_effects(text: str, context_type: str) -> str:
    """Handle Japanese onomatopoeia translation"""
    
    if context_type != "sfx":
        return text
    
    # Common Japanese SFX mappings
    sfx_mappings = {
        # Impact sounds
        "ドン": "THUD",
        "バン": "BANG",
        "ガン": "CLANG",
        
        # Movement sounds  
        "ザッ": "DASH",
        "スッ": "WHOOSH",
        "パタパタ": "FLUTTER",
        
        # Emotional sounds
        "ドキドキ": "THUMP THUMP",
        "ハァハァ": "PANT PANT",
        "シーン": "SILENCE",
        
        # Repeat patterns
        r"(.)\1+": r"\1 \1 \1",  # ああああ → A A A
    }
    
    translated_sfx = text
    for japanese_sfx, english_sfx in sfx_mappings.items():
        if isinstance(japanese_sfx, str):
            translated_sfx = translated_sfx.replace(japanese_sfx, english_sfx)
        else:
            # Regex pattern
            translated_sfx = re.sub(japanese_sfx, english_sfx, translated_sfx)
    
    return translated_sfx
```

#### Honorifics and Cultural Elements
```python
def handle_honorifics(text: str, preserve_honorifics: bool = True) -> str:
    """Handle Japanese honorifics in translation"""
    
    honorifics_map = {
        "さん": "-san",
        "ちゃん": "-chan", 
        "くん": "-kun",
        "様": "-sama",
        "先生": "-sensei",
        "先輩": "-senpai",
        "後輩": "-kohai"
    }
    
    if preserve_honorifics:
        # Keep honorifics for cultural authenticity
        for jp_honorific, en_honorific in honorifics_map.items():
            text = text.replace(jp_honorific, en_honorific)
    else:
        # Remove or adapt honorifics for Western audience
        text = re.sub(r'[さちく様先生輩後]+', '', text)
        # Add appropriate English titles where needed
        text = add_appropriate_english_titles(text)
    
    return text

def handle_cultural_references(text: str) -> str:
    """Handle cultural references that need explanation or adaptation"""
    
    cultural_adaptations = {
        "お弁当": "lunch box",
        "部活": "club activities", 
        "文化祭": "school festival",
        "お疲れ様": "good work",
        "いただきます": "thanks for the meal",
        "お帰りなさい": "welcome home",
    }
    
    for japanese_term, english_adaptation in cultural_adaptations.items():
        if japanese_term in text:
            text = text.replace(japanese_term, english_adaptation)
    
    return text
```

---

## 🎨 Component 4: Inpainter

### Overview
Intelligent text removal system that seamlessly removes original manga text while preserving the underlying artwork through neural inpainting and traditional image processing techniques.

### Input/Output Specification

**Input**:
```python
class InpaintingInput:
    original_image: np.ndarray     # Full image (H, W, 3)
    text_masks: List[np.ndarray]   # Binary masks for each text region
    bubble_regions: List[BubbleRegion]  # Associated bubble information
    inpainting_method: str         # "neural", "traditional", "hybrid"
    quality_level: str            # "fast", "balanced", "high_quality"
    preserve_style: bool          # Match surrounding art style
```

**Output**:
```python
class InpaintingResult:
    cleaned_image: np.ndarray      # Image with text removed
    quality_score: float           # Inpainting quality assessment 0.0-1.0
    processing_time: float         # Time taken for inpainting
    method_used: str              # Which inpainting method was used
    artifacts_detected: bool       # Whether artifacts were detected
    regions_processed: int         # Number of text regions inpainted
```

### Inpainting Methods

#### 1. LaMa (Large Mask Inpainting) - Primary Neural Method

**Installation**: 
```bash
# Install dependencies
pip install torch torchvision
pip install opencv-python pillow numpy

# Download LaMa model
wget https://github.com/saic-mdal/lama/releases/download/models/big-lama.zip
```

**Implementation**:
```python
import torch
import cv2
import numpy as np
from PIL import Image

class LaMaInpainter:
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self.select_device(device)
        self.model = self.load_lama_model(model_path)
        self.model.eval()
    
    def load_lama_model(self, model_path: str):
        """Load pre-trained LaMa model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = LaMaModel(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(self.device)
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Perform neural inpainting using LaMa"""
        
        # Preprocess inputs
        image_tensor = self.preprocess_image(image)
        mask_tensor = self.preprocess_mask(mask)
        
        with torch.no_grad():
            # Run inference
            inpainted_tensor = self.model(image_tensor, mask_tensor)
            
            # Postprocess result
            inpainted_image = self.postprocess_result(inpainted_tensor)
        
        return inpainted_image
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input format"""
        # Normalize to [-1, 1]
        image_normalized = (image.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Pad to multiple of 8 (model requirement)
        image_tensor = self.pad_to_multiple(image_tensor, 8)
        
        return image_tensor.to(self.device)
    
    def preprocess_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Convert binary mask to model input format"""
        # Ensure binary mask (0 or 1)
        mask_binary = (mask > 127).astype(np.float32)
        
        # Dilate mask slightly to ensure complete text removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(mask_dilated).unsqueeze(0).unsqueeze(0)
        mask_tensor = self.pad_to_multiple(mask_tensor, 8)
        
        return mask_tensor.to(self.device)
    
    def postprocess_result(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output back to image"""
        # Remove batch dimension and convert to numpy
        tensor = tensor.squeeze(0).cpu()
        image = tensor.permute(1, 2, 0).numpy()
        
        # Denormalize from [-1, 1] to [0, 255]
        image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        return image
```

#### 2. Traditional GIMP Methods - Fallback

**Implementation**:
```python
class GimpInpainter:
    def __init__(self, gimp_image, gimp_layer):
        self.image = gimp_image
        self.layer = gimp_layer
    
    def inpaint_with_resynthesizer(self, mask_layer):
        """Use GIMP's Resynthesizer plugin for inpainting"""
        try:
            # Ensure Resynthesizer is available
            if not self.check_resynthesizer_available():
                raise InpaintingError("Resynthesizer plugin not available")
            
            # Apply heal selection
            pdb.python_fu_heal_selection(
                self.image,
                self.layer,
                50,     # Heal radius
                0,      # Direction
                True    # Random seed
            )
            
            return True
            
        except Exception as e:
            raise InpaintingError(f"Resynthesizer inpainting failed: {str(e)}")
    
    def inpaint_with_clone_tool(self, text_regions):
        """Use GIMP's clone tool for manual inpainting"""
        
        for region in text_regions:
            # Find good source region for cloning
            source_region = self.find_clone_source(region)
            
            if source_region:
                # Apply clone tool
                pdb.gimp_clone(
                    self.layer,
                    self.layer,          # Source drawable
                    CLONE_IMAGE,         # Clone type
                    source_region.x,     # Source x
                    source_region.y,     # Source y
                    region.width,        # Number of stroke coordinates
                    region.stroke_coords # Stroke coordinates
                )
    
    def find_clone_source(self, text_region):
        """Find appropriate source region for cloning"""
        
        # Analyze surrounding regions
        surrounding_regions = self.get_surrounding_regions(text_region)
        
        best_source = None
        best_similarity = 0
        
        for candidate in surrounding_regions:
            # Calculate texture similarity
            similarity = self.calculate_texture_similarity(
                text_region.surrounding_texture,
                candidate.texture
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_source = candidate
        
        return best_source if best_similarity > 0.7 else None
```

#### 3. Hybrid Approach - Combining Methods

**Implementation**:
```python
class HybridInpainter:
    def __init__(self):
        self.lama_inpainter = LaMaInpainter()
        self.gimp_inpainter = GimpInpainter()
    
    def inpaint(self, image: np.ndarray, masks: List[np.ndarray]) -> InpaintingResult:
        """Combine neural and traditional methods for optimal results"""
        
        results = []
        for mask in masks:
            mask_area = np.sum(mask > 0)
            mask_complexity = self.analyze_mask_complexity(mask)
            
            if mask_area < 1000 and mask_complexity < 0.5:
                # Small, simple regions - use traditional methods
                result = self.gimp_inpainter.inpaint(image, mask)
                method = "traditional"
            else:
                # Large or complex regions - use neural inpainting
                result = self.lama_inpainter.inpaint(image, mask)
                method = "neural"
                
                # Quality check - fallback if artifacts detected
                if self.detect_artifacts(result, mask):
                    result = self.gimp_inpainter.inpaint(image, mask)
                    method = "traditional_fallback"
            
            results.append((result, method))
        
        # Combine all inpainted regions
        final_image = self.combine_inpainted_regions(image, results)
        
        return InpaintingResult(
            cleaned_image=final_image,
            quality_score=self.assess_quality(final_image, masks),
            method_used="hybrid",
            regions_processed=len(masks)
        )
```

### Configuration Options

```python
class InpainterConfig:
    # Method Selection
    preferred_method: str = "neural"  # "neural", "traditional", "hybrid"
    fallback_methods: List[str] = ["traditional", "hybrid"]
    
    # Neural Inpainting
    neural_model_path: str = "models/lama_big.pth"
    use_gpu: bool = True
    gpu_memory_limit: float = 4.0  # GB
    batch_processing: bool = True
    
    # Quality Settings
    mask_dilation: int = 2          # Pixels to expand text mask
    quality_threshold: float = 0.8   # Minimum acceptable quality
    artifact_detection: bool = True
    
    # Traditional Methods
    enable_resynthesizer: bool = True
    heal_radius: int = 50
    clone_tool_fallback: bool = True
    
    # Performance
    max_processing_time: float = 60.0
    max_region_size: int = 500000   # Max pixels per region
    
    # Style Preservation
    preserve_art_style: bool = True
    edge_blending: bool = True
    color_matching: bool = True
```

### Error Handling

```python
class InpaintingError(Exception):
    pass

class ModelLoadError(InpaintingError):
    """Neural model loading failed"""
    pass

class GPUMemoryError(InpaintingError):
    """Insufficient GPU memory"""
    pass

class QualityError(InpaintingError):
    """Inpainting quality below threshold"""
    pass

def robust_inpainting_pipeline(input_data: InpaintingInput, config: InpainterConfig):
    """Robust inpainting with fallback strategies"""
    
    methods_to_try = [config.preferred_method] + config.fallback_methods
    last_error = None
    
    for method in methods_to_try:
        try:
            inpainter = inpainting_engines[method]
            result = inpainter.inpaint(input_data)
            
            # Quality check
            if result.quality_score >= config.quality_threshold:
                return result
            else:
                # Try next method for better quality
                continue
                
        except GPUMemoryError:
            # GPU memory issue - try CPU or traditional methods
            if "traditional" not in methods_to_try:
                methods_to_try.append("traditional")
            continue
        except ModelLoadError:
            # Neural model unavailable - skip to traditional
            continue
        except Exception as e:
            last_error = e
            continue
    
    # All methods failed
    if last_error:
        raise last_error
    else:
        raise InpaintingError("No inpainting methods available")
```

### Edge Cases

#### Complex Background Patterns
```python
def handle_complex_backgrounds(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Special handling for complex artistic backgrounds"""
    
    # Analyze background complexity
    background_features = analyze_background_complexity(image, mask)
    
    if background_features.has_gradients:
        # Preserve gradients during inpainting
        result = inpaint_with_gradient_preservation(image, mask)
    elif background_features.has_patterns:
        # Use pattern-aware inpainting
        result = inpaint_with_pattern_matching(image, mask)
    elif background_features.has_textures:
        # Texture synthesis approach
        result = inpaint_with_texture_synthesis(image, mask)
    else:
        # Standard inpainting
        result = standard_inpainting(image, mask)
    
    return result

def inpaint_with_gradient_preservation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Preserve color gradients during inpainting"""
    
    # Extract gradient information from surrounding area
    gradient_info = extract_local_gradients(image, mask)
    
    # Perform inpainting
    base_result = lama_inpainter.inpaint(image, mask)
    
    # Apply gradient correction
    corrected_result = apply_gradient_correction(base_result, mask, gradient_info)
    
    return corrected_result
```

#### Overlapping Text and Art Elements
```python
def handle_overlapping_elements(image: np.ndarray, text_mask: np.ndarray, 
                              bubble_region: BubbleRegion) -> np.ndarray:
    """Handle text that overlaps with important art elements"""
    
    # Detect art elements in the text region
    art_elements = detect_art_elements(image, text_mask)
    
    if art_elements:
        # Create selective mask that preserves art elements
        selective_mask = create_selective_mask(text_mask, art_elements)
        
        # Inpaint only the text portions
        result = lama_inpainter.inpaint(image, selective_mask)
        
        # Restore art elements if damaged
        result = restore_art_elements(result, art_elements, image)
    else:
        # Standard inpainting
        result = lama_inpainter.inpaint(image, text_mask)
    
    return result

def detect_art_elements(image: np.ndarray, text_mask: np.ndarray) -> List[ArtElement]:
    """Detect important art elements within text region"""
    
    # Extract region
    region = extract_masked_region(image, text_mask)
    
    # Detect various art elements
    elements = []
    
    # Character hair/clothing lines
    hair_lines = detect_hair_lines(region)
    elements.extend(hair_lines)
    
    # Background detail lines
    bg_lines = detect_background_lines(region)
    elements.extend(bg_lines)
    
    # Important color regions
    color_regions = detect_important_colors(region)
    elements.extend(color_regions)
    
    return elements
```

#### Large Text Regions
```python
def handle_large_text_regions(image: np.ndarray, large_mask: np.ndarray) -> np.ndarray:
    """Handle very large text regions that might exceed model capabilities"""
    
    mask_area = np.sum(large_mask > 0)
    
    if mask_area > MAX_INPAINT_AREA:
        # Split large mask into smaller, overlapping regions
        sub_masks = split_large_mask(large_mask, overlap_ratio=0.2)
        
        # Inpaint each region separately
        partial_results = []
        current_image = image.copy()
        
        for sub_mask in sub_masks:
            partial_result = lama_inpainter.inpaint(current_image, sub_mask)
            
            # Apply result to current image
            current_image = apply_partial_result(current_image, partial_result, sub_mask)
            partial_results.append(partial_result)
        
        # Blend overlapping regions for seamless result
        final_result = blend_overlapping_regions(current_image, partial_results, sub_masks)
        
    else:
        # Standard inpainting for manageable size
        final_result = lama_inpainter.inpaint(image, large_mask)
    
    return final_result
```

---

## ✍️ Component 5: Typesetter

### Overview
Intelligent text rendering system that places English translations into cleaned speech bubbles with appropriate fonts, sizing, and layout to match manga typography standards.

### Input/Output Specification

**Input**:
```python
class TypesettingInput:
    cleaned_image: np.ndarray      # Image with text removed
    translated_texts: List[str]    # Translated text for each bubble
    bubble_regions: List[BubbleRegion]  # Bubble boundaries and info
    original_text_style: List[TextStyle]  # Original text styling info
    font_preferences: FontPreferences   # User font preferences
    layout_mode: str              # "auto", "preserve_original", "optimize_english"
```

**Output**:
```python
class TypesettingResult:
    final_image: np.ndarray        # Image with translated text rendered
    text_layers: List[GimpTextLayer]  # Individual GIMP text layers
    font_choices: List[str]        # Fonts used for each text region
    layout_info: List[LayoutInfo]  # Detailed layout information
    rendering_quality: float      # Overall rendering quality score
    processing_time: float        # Time taken for typesetting
```

**Supporting Classes**:
```python
class TextStyle:
    font_family: str
    font_size: int
    is_bold: bool
    is_italic: bool
    has_outline: bool
    outline_color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    alignment: str                 # "left", "center", "right", "justify"
    line_spacing: float
    character_spacing: float

class LayoutInfo:
    text_bounds: BoundingBox
    line_breaks: List[int]         # Character positions of line breaks
    font_size_used: int
    lines_count: int
    fit_quality: float             # How well text fits in bubble (0.0-1.0)
```

### Font Management System

**Font Selection Engine**:
```python
class FontManager:
    def __init__(self):
        self.available_fonts = self.scan_available_fonts()
        self.manga_fonts = self.load_manga_font_database()
        self.font_cache = {}
    
    def scan_available_fonts(self) -> List[FontInfo]:
        """Scan system and plugin fonts"""
        system_fonts = self.get_system_fonts()
        plugin_fonts = self.get_plugin_fonts()
        
        all_fonts = []
        for font_path in system_fonts + plugin_fonts:
            font_info = self.analyze_font(font_path)
            if font_info.is_suitable_for_manga():
                all_fonts.append(font_info)
        
        return all_fonts
    
    def select_optimal_font(self, text_style: TextStyle, 
                          bubble_context: str) -> FontInfo:
        """Select best font for given context"""
        
        # Score all available fonts
        font_scores = []
        for font in self.available_fonts:
            score = self.calculate_font_score(font, text_style, bubble_context)
            font_scores.append((font, score))
        
        # Return highest scoring font
        best_font, _ = max(font_scores, key=lambda x: x[1])
        return best_font
    
    def calculate_font_score(self, font: FontInfo, text_style: TextStyle, 
                           context: str) -> float:
        """Score font suitability for specific use case"""
        
        score = 0.0
        
        # Context-based scoring
        if context == "dialogue":
            if font.is_comic_style:
                score += 3.0
            if font.readability_score > 0.8:
                score += 2.0
        elif context == "narration":
            if font.is_formal:
                score += 2.0
            if font.has_clean_lines:
                score += 1.5
        elif context == "sfx":
            if font.is_bold_style:
                score += 3.0
            if font.has_impact:
                score += 2.0
        
        # Style matching
        if text_style.is_bold and font.supports_bold:
            score += 1.0
        if text_style.has_outline and font.supports_outline:
            score += 1.0
        
        # Technical factors
        score += font.unicode_support * 0.5
        score += font.size_scalability * 0.3
        
        return score
```

**Recommended Fonts Database**:
```python
MANGA_FONTS_DATABASE = {
    "dialogue": {
        "primary": [
            {
                "name": "CC Wild Words Roman",
                "url": "https://fonts.google.com/specimen/CC+Wild+Words",
                "features": ["comic_style", "readable", "free"],
                "fallbacks": ["Comic Sans MS", "Trebuchet MS"]
            },
            {
                "name": "Manga Temple",
                "url": "https://www.dafont.com/manga-temple.font", 
                "features": ["manga_optimized", "multiple_weights"],
                "fallbacks": ["Arial Bold", "Helvetica Bold"]
            }
        ]
    },
    "sfx": {
        "primary": [
            {
                "name": "Action Man",
                "url": "https://www.dafont.com/action-man.font",
                "features": ["impact_style", "bold", "condensed"],
                "fallbacks": ["Impact", "Arial Black"]
            },
            {
                "name": "Crash Bang Wallop",
                "url": "https://www.dafont.com/crash-bang-wallop.font",
                "features": ["comic_sfx", "varied_weights"],
                "fallbacks": ["Cooper Black", "Franklin Gothic Heavy"]
            }
        ]
    },
    "narration": {
        "primary": [
            {
                "name": "Lato",
                "url": "https://fonts.google.com/specimen/Lato",
                "features": ["clean", "readable", "professional"],
                "fallbacks": ["Times New Roman", "Georgia"]
            }
        ]
    }
}
```

### Text Layout Engine

**Auto-sizing Algorithm**:
```python
class TextLayoutEngine:
    def __init__(self):
        self.font_manager = FontManager()
        self.line_breaker = LineBreaker()
    
    def fit_text_to_bubble(self, text: str, bubble_region: BubbleRegion,
                          font_info: FontInfo) -> LayoutInfo:
        """Automatically size and layout text within bubble"""
        
        # Calculate available space
        available_area = self.calculate_text_area(bubble_region)
        
        # Binary search for optimal font size
        min_size, max_size = 8, 72
        optimal_size = None
        best_layout = None
        
        for size in range(max_size, min_size - 1, -1):
            layout = self.attempt_layout(text, available_area, font_info, size)
            
            if layout.fits_completely:
                optimal_size = size
                best_layout = layout
                break
            elif layout.fit_quality > 0.8:
                # Acceptable fit even if not perfect
                optimal_size = size
                best_layout = layout
                # Continue to see if we can do better
        
        if not best_layout:
            # Text doesn't fit - try emergency strategies
            best_layout = self.emergency_fit_strategies(text, available_area, font_info)
        
        return best_layout
    
    def calculate_text_area(self, bubble_region: BubbleRegion) -> TextArea:
        """Calculate usable area within bubble for text"""
        
        # Get bubble interior mask
        bubble_mask = bubble_region.mask
        
        # Find largest inscribed rectangle
        largest_rect = self.find_largest_rectangle(bubble_mask)
        
        # Apply margins for better appearance
        margin_factor = 0.85  # 85% of available space
        adjusted_rect = self.apply_margins(largest_rect, margin_factor)
        
        return TextArea(
            x=adjusted_rect.x,
            y=adjusted_rect.y,
            width=adjusted_rect.width,
            height=adjusted_rect.height,
            shape="rectangle"  # Could be "ellipse" for round bubbles
        )
    
    def attempt_layout(self, text: str, text_area: TextArea,
                      font_info: FontInfo, font_size: int) -> LayoutAttempt:
        """Attempt to layout text at given font size"""
        
        # Create font metrics
        font_metrics = FontMetrics(font_info, font_size)
        
        # Break text into lines
        lines = self.line_breaker.break_lines(text, text_area.width, font_metrics)
        
        # Calculate total text dimensions
        total_height = len(lines) * font_metrics.line_height
        max_width = max([font_metrics.get_text_width(line) for line in lines])
        
        # Check if it fits
        fits_width = max_width <= text_area.width
        fits_height = total_height <= text_area.height
        fits_completely = fits_width and fits_height
        
        # Calculate fit quality
        width_ratio = max_width / text_area.width
        height_ratio = total_height / text_area.height
        fit_quality = min(1.0 / max(width_ratio, 1.0), 1.0 / max(height_ratio, 1.0))
        
        return LayoutAttempt(
            lines=lines,
            font_size=font_size,
            total_width=max_width,
            total_height=total_height,
            fits_completely=fits_completely,
            fit_quality=fit_quality
        )
```

**Line Breaking Algorithm**:
```python
class LineBreaker:
    def __init__(self):
        self.word_break_penalties = self.load_word_break_penalties()
    
    def break_lines(self, text: str, max_width: int, 
                   font_metrics: FontMetrics) -> List[str]:
        """Break text into lines using optimal line breaking"""
        
        words = self.tokenize_text(text)
        
        # Dynamic programming approach to optimal line breaking
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = font_metrics.get_text_width(word)
            space_width = font_metrics.get_text_width(" ")
            
            # Check if word fits on current line
            if current_line and current_width + space_width + word_width > max_width:
                # Start new line
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                # Add to current line
                if current_line:
                    current_width += space_width
                current_line.append(word)
                current_width += word_width
        
        # Add final line
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def tokenize_text(self, text: str) -> List[str]:
        """Smart tokenization for manga text"""
        
        # Handle common manga punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)  # Normalize multiple punctuation
        
        # Split on spaces but preserve punctuation attachment
        words = []
        for word in text.split():
            # Handle attached punctuation
            if re.match(r'.*[!?.,]$', word):
                # Punctuation stays with word
                words.append(word)
            else:
                words.append(word)
        
        return words
```

### Rendering Engine

**GIMP Text Layer Creation**:
```python
class GimpTextRenderer:
    def __init__(self, gimp_image):
        self.image = gimp_image
    
    def create_text_layer(self, text: str, layout_info: LayoutInfo,
                         font_info: FontInfo, style: TextStyle) -> GimpTextLayer:
        """Create GIMP text layer with proper styling"""
        
        # Create text layer
        text_layer = pdb.gimp_text_layer_new(
            self.image,
            text,
            font_info.name,
            layout_info.font_size_used,
            UNIT_PIXEL
        )
        
        # Apply styling
        if style.is_bold:
            pdb.gimp_text_layer_set_font_size(text_layer, layout_info.font_size_used)
            # Note: GIMP handles bold through font selection
        
        if style.text_color:
            pdb.gimp_text_layer_set_color(text_layer, style.text_color)
        
        # Set alignment
        alignment_map = {
            "left": TEXT_JUSTIFY_LEFT,
            "center": TEXT_JUSTIFY_CENTER,
            "right": TEXT_JUSTIFY_RIGHT,
            "justify": TEXT_JUSTIFY_FILL
        }
        pdb.gimp_text_layer_set_justification(
            text_layer, 
            alignment_map.get(style.alignment, TEXT_JUSTIFY_CENTER)
        )
        
        # Position the text layer
        pdb.gimp_layer_set_offsets(
            text_layer,
            layout_info.text_bounds.x,
            layout_info.text_bounds.y
        )
        
        # Apply text effects if needed
        if style.has_outline:
            self.apply_text_outline(text_layer, style.outline_color)
        
        return text_layer
    
    def apply_text_outline(self, text_layer, outline_color):
        """Apply outline effect to text"""
        
        # Create outline using stroke path
        pdb.gimp_image_select_item(self.image, CHANNEL_OP_REPLACE, text_layer)
        pdb.gimp_selection_border(self.image, 2)  # 2px outline
        
        # Create new layer for outline
        outline_layer = pdb.gimp_layer_new(
            self.image,
            text_layer.width,
            text_layer.height,
            RGBA_IMAGE,
            "Text Outline",
            100,
            LAYER_MODE_NORMAL
        )
        
        pdb.gimp_image_insert_layer(self.image, outline_layer, None, -1)
        
        # Fill selection with outline color
        pdb.gimp_context_set_foreground(outline_color)
        pdb.gimp_edit_fill(outline_layer, FILL_FOREGROUND)
        
        # Move outline behind text
        pdb.gimp_image_lower_item(self.image, outline_layer)
        
        pdb.gimp_selection_none(self.image)
```

### Configuration Options

```python
class TypesetterConfig:
    # Font Management
    font_directories: List[str] = ["./fonts/", "~/.fonts/", "/usr/share/fonts/"]
    preferred_fonts: Dict[str, List[str]] = {
        "dialogue": ["CC Wild Words", "Comic Sans MS", "Trebuchet MS"],
        "narration": ["Lato", "Open Sans", "Arial"],
        "sfx": ["Impact", "Arial Black", "Franklin Gothic Heavy"]
    }
    download_missing_fonts: bool = True
    
    # Layout Settings
    auto_sizing: bool = True
    min_font_size: int = 8
    max_font_size: int = 72
    margin_factor: float = 0.85      # Percentage of bubble to use for text
    line_spacing_factor: float = 1.2  # Multiplier for line spacing
    
    # Quality Settings
    fit_quality_threshold: float = 0.8
    enable_emergency_strategies: bool = True
    prefer_smaller_text_over_overflow: bool = True
    
    # Styling
    auto_detect_text_style: bool = True
    preserve_original_styling: bool = True
    default_text_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    default_outline_color: Tuple[int, int, int] = (255, 255, 255)  # White
    
    # Performance
    max_processing_time: float = 30.0
    cache_font_metrics: bool = True
    parallel_processing: bool = True
```

### Error Handling

```python
class TypesettingError(Exception):
    pass

class FontNotFoundError(TypesettingError):
    """Required font not available"""
    pass

class TextTooLongError(TypesettingError):
    """Text cannot fit in available space"""
    pass

class RenderingError(TypesettingError):
    """Error during text rendering"""
    pass

def robust_typesetting_pipeline(input_data: TypesettingInput, 
                              config: TypesetterConfig):
    """Robust typesetting with fallback strategies"""
    
    results = []
    
    for i, (text, bubble, style) in enumerate(zip(
        input_data.translated_texts,
        input_data.bubble_regions, 
        input_data.original_text_style
    )):
        try:
            # Attempt normal typesetting
            result = typesetter.render_text(text, bubble, style)
            results.append(result)
            
        except TextTooLongError:
            # Text too long - try emergency strategies
            try:
                result = apply_emergency_strategies(text, bubble, style)
                results.append(result)
            except Exception:
                # Last resort - create basic text layer
                result = create_basic_text_layer(text, bubble)
                results.append(result)
                
        except FontNotFoundError:
            # Font not available - use fallback
            fallback_style = style.copy()
            fallback_style.font_family = config.preferred_fonts["dialogue"][0]
            result = typesetter.render_text(text, bubble, fallback_style)
            results.append(result)
            
        except Exception as e:
            # Unexpected error - log and continue
            logging.error(f"Typesetting failed for region {i}: {str(e)}")
            continue
    
    return TypesettingResult(
        text_layers=results,
        processing_time=time.time() - start_time
    )
```

### Edge Cases

#### Emergency Text Fitting Strategies
```python
def apply_emergency_strategies(text: str, bubble_region: BubbleRegion,
                             style: TextStyle) -> GimpTextLayer:
    """Emergency strategies when normal layout fails"""
    
    strategies = [
        abbreviate_common_phrases,
        remove_unnecessary_words,
        use_smaller_font_aggressively,
        split_into_multiple_bubbles,
        use_vertical_layout,
        apply_extreme_compression
    ]
    
    for strategy in strategies:
        try:
            modified_text = strategy(text)
            result = typesetter.render_text(modified_text, bubble_region, style)
            if result.layout_info.fit_quality > 0.6:
                return result
        except Exception:
            continue
    
    # Last resort - just render what fits
    return render_truncated_text(text, bubble_region, style)

def abbreviate_common_phrases(text: str) -> str:
    """Abbreviate common English phrases for manga"""
    
    abbreviations = {
        "what are you": "what're you",
        "I am going to": "I'm gonna",
        "you are": "you're", 
        "cannot": "can't",
        "should not": "shouldn't",
        "would not": "wouldn't",
        "something": "somethin'",
        "nothing": "nothin'",
        "because": "'cause"
    }
    
    for phrase, abbrev in abbreviations.items():
        text = text.replace(phrase, abbrev)
    
    return text
```

#### Vertical Text Layout
```python
def create_vertical_text_layout(text: str, bubble_region: BubbleRegion) -> LayoutInfo:
    """Create vertical text layout for tall, narrow bubbles"""
    
    # Check if bubble is suitable for vertical text
    aspect_ratio = bubble_region.height / bubble_region.width
    
    if aspect_ratio > 2.0:
        # Very tall bubble - vertical layout might work better
        
        # Calculate character-per-line for vertical layout
        available_height = bubble_region.height * 0.85
        char_height = estimate_character_height(font_size)
        max_chars_vertical = int(available_height / char_height)
        
        # Break into vertical lines
        lines = []
        for i in range(0, len(text), max_chars_vertical):
            line = text[i:i + max_chars_vertical]
            lines.append(line)
        
        return create_vertical_layout_info(lines, bubble_region)
    
    return None  # Not suitable for vertical layout
```

#### Multi-Bubble Text Splitting
```python
def split_text_across_bubbles(long_text: str, 
                            connected_bubbles: List[BubbleRegion]) -> List[str]:
    """Split long text across multiple connected speech bubbles"""
    
    # Estimate capacity of each bubble
    bubble_capacities = []
    for bubble in connected_bubbles:
        area = bubble.width * bubble.height
        estimated_chars = area / AVERAGE_CHAR_AREA
        bubble_capacities.append(int(estimated_chars))
    
    # Split text intelligently at sentence/phrase boundaries
    sentences = split_into_sentences(long_text)
    
    bubble_texts = [""] * len(connected_bubbles)
    current_bubble = 0
    
    for sentence in sentences:
        if (len(bubble_texts[current_bubble]) + len(sentence) <= 
            bubble_capacities[current_bubble]):
            # Fits in current bubble
            bubble_texts[current_bubble] += sentence + " "
        else:
            # Move to next bubble
            current_bubble += 1
            if current_bubble < len(connected_bubbles):
                bubble_texts[current_bubble] = sentence + " "
            else:
                # No more bubbles - truncate
                break
    
    return [text.strip() for text in bubble_texts if text.strip()]
```

---

## 🔧 Component 6: GIMP Integration Layer

### Overview
Bridge component that handles all interactions with GIMP's API, manages plugin lifecycle, handles user interface, and coordinates between the manga translation components and GIMP's layer system.

### Input/Output Specification

**Input**:
```python
class GimpIntegrationInput:
    gimp_image: GimpImage         # Current GIMP image
    active_layer: GimpLayer       # Currently selected layer
    user_selections: List[Selection]  # User-made selections (if any)
    plugin_mode: str             # "auto", "semi-auto", "manual"
    user_preferences: UserPreferences  # Plugin settings
```

**Output**:
```python
class GimpIntegrationOutput:
    result_layers: List[GimpLayer]     # Created/modified layers
    undo_group_id: int                # GIMP undo group for rollback
    processing_report: ProcessingReport  # Summary of operations
    user_notifications: List[str]      # Messages for user
    modified_image_state: ImageState   # Updated image state
```

### GIMP API Compatibility Layer

#### Python-Fu Integration (GIMP 2.10)

**Plugin Registration**:
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gimpfu import *
import sys
import os

# Add plugin directory to Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

from manga_translator.core import MangaTranslatorCore
from manga_translator.ui.dialogs import MangaTranslatorDialog

def manga_translate_auto(image, drawable):
    """Auto translation mode - one-click translation"""
    try:
        # Start undo group
        pdb.gimp_image_undo_group_start(image)
        
        # Initialize translator core
        translator = MangaTranslatorCore(image, drawable)
        
        # Run auto translation pipeline
        result = translator.auto_translate()
        
        # Update GIMP interface
        gimp.displays_flush()
        
        # End undo group
        pdb.gimp_image_undo_group_end(image)
        
        return result
        
    except Exception as e:
        # Ensure undo group is properly closed
        pdb.gimp_image_undo_group_end(image)
        gimp.message(f"Translation failed: {str(e)}")
        raise

def manga_translate_interactive(image, drawable):
    """Interactive translation mode with dialog"""
    try:
        # Create and show dialog
        dialog = MangaTranslatorDialog(image, drawable)
        response = dialog.run()
        
        if response == gtk.RESPONSE_OK:
            # Get user settings from dialog
            settings = dialog.get_settings()
            
            # Start undo group
            pdb.gimp_image_undo_group_start(image)
            
            # Initialize translator with settings
            translator = MangaTranslatorCore(image, drawable, settings)
            
            # Run translation based on mode
            if settings.mode == "semi-auto":
                result = translator.semi_auto_translate()
            elif settings.mode == "manual":
                result = translator.manual_translate()
            else:
                result = translator.auto_translate()
            
            # Update interface
            gimp.displays_flush()
            pdb.gimp_image_undo_group_end(image)
            
        dialog.destroy()
        
    except Exception as e:
        pdb.gimp_image_undo_group_end(image)
        gimp.message(f"Translation failed: {str(e)}")
        raise

# Register auto mode
register(
    "python-fu-manga-translate-auto",
    "Automatically translate manga page",
    "One-click translation of manga panels using AI",
    "Manga Translator Plugin Team",
    "GPL v3",
    "2024",
    "_Auto Translate Page",
    "RGB*, GRAY*",
    [],
    [],
    manga_translate_auto,
    menu="<Image>/Filters/Manga",
    domain=("gimp20-python", gimp.locale_directory)
)

# Register interactive mode
register(
    "python-fu-manga-translate-interactive",
    "Interactive manga translation",
    "Translate manga with full control and preview",
    "Manga Translator Plugin Team", 
    "GPL v3",
    "2024",
    "_Translate Page...",
    "RGB*, GRAY*",
    [],
    [],
    manga_translate_interactive,
    menu="<Image>/Filters/Manga",
    domain=("gimp20-python", gimp.locale_directory)
)

# Register settings dialog
register(
    "python-fu-manga-translate-settings",
    "Manga translation settings",
    "Configure manga translation preferences",
    "Manga Translator Plugin Team",
    "GPL v3", 
    "2024",
    "_Settings...",
    "",
    [],
    [],
    manga_translate_settings,
    menu="<Image>/Filters/Manga",
    domain=("gimp20-python", gimp.locale_directory)
)

main()
```

#### LibGIMP Integration (GIMP 3.0+)

**Modern Plugin Architecture**:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gi
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')
gi.require_version('Gtk', '3.0')

from gi.repository import Gimp, GimpUi, GObject, Gtk
import sys
import os

# Add plugin path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manga_translator.core import MangaTranslatorCore
from manga_translator.ui.gtk_dialogs import MangaTranslatorGtkDialog

class MangaTranslatePlugin(Gimp.PlugIn):
    ## Plugin properties ##
    __gproperties__ = {
        "mode": (str, "Translation mode", "Auto, Semi-auto, or Manual", 
                "auto", GObject.ParamFlags.READWRITE),
        "source-lang": (str, "Source language", "Source text language",
                       "auto", GObject.ParamFlags.READWRITE),
        "target-lang": (str, "Target language", "Target translation language",
                       "en", GObject.ParamFlags.READWRITE),
    }

    def do_query_procedures(self):
        """Register plugin procedures"""
        return ["manga-translate-auto", "manga-translate-interactive"]

    def do_create_procedure(self, name):
        """Create procedure instances"""
        if name == "manga-translate-auto":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN,
                self.run_auto_translate, None)
            
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_documentation(
                "Auto translate manga page",
                "Automatically detect and translate manga text",
                name)
            procedure.set_menu_label("Auto Translate Page")
            procedure.add_menu_path("<Image>/Filters/Manga")
            
        elif name == "manga-translate-interactive":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN,
                self.run_interactive_translate, None)
                
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_documentation(
                "Interactive manga translation",
                "Translate manga with user control and preview",
                name)
            procedure.set_menu_label("Translate Page...")
            procedure.add_menu_path("<Image>/Filters/Manga")
            
            # Add parameters
            procedure.add_enum_argument("mode", "Translation mode",
                                      "Translation mode", 
                                      TranslationMode, TranslationMode.AUTO,
                                      GObject.ParamFlags.READWRITE)
                                      
        return procedure

    def run_auto_translate(self, procedure, run_mode, image, n_layers, layers, 
                          args, run_data):
        """Execute auto translation"""
        try:
            # Get active layer
            active_layer = layers[0] if layers else image.get_active_layer()
            
            # Start undo group
            image.undo_group_start()
            
            # Initialize translator
            translator = MangaTranslatorCore(image, active_layer)
            
            # Run translation
            result = translator.auto_translate()
            
            # Finish
            image.undo_group_end()
            Gimp.displays_flush()
            
            return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, 
                                             GLib.Error())
                                             
        except Exception as e:
            image.undo_group_end()
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR,
                                             GLib.Error(str(e)))

    def run_interactive_translate(self, procedure, run_mode, image, n_layers, 
                                layers, args, run_data):
        """Execute interactive translation"""
        if run_mode == Gimp.RunMode.INTERACTIVE:
            # Show dialog
            dialog = MangaTranslatorGtkDialog(image, layers[0])
            response = dialog.run()
            
            if response == Gtk.ResponseType.OK:
                settings = dialog.get_settings()
                # Execute translation with settings
                return self.execute_translation(image, layers[0], settings)
            else:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL,
                                                 GLib.Error())
        else:
            # Non-interactive mode
            return self.execute_translation(image, layers[0], default_settings)

Gimp.main(MangaTranslatePlugin.__gtype__, sys.argv)
```

### Layer Management System

**Layer Organization Strategy**:
```python
class LayerManager:
    def __init__(self, gimp_image):
        self.image = gimp_image
        self.layer_groups = {}
        
    def create_translation_layer_structure(self):
        """Create organized layer structure for translation"""
        
        # Create main translation group
        translation_group = pdb.gimp_layer_group_new(self.image)
        pdb.gimp_item_set_name(translation_group, "Manga Translation")
        pdb.gimp_image_insert_layer(self.image, translation_group, None, -1)
        
        # Create subgroups
        subgroups = {
            "original": "Original",
            "cleaned": "Cleaned Background", 
            "translated": "Translated Text",
            "effects": "Text Effects"
        }
        
        for key, name in subgroups.items():
            group = pdb.gimp_layer_group_new(self.image)
            pdb.gimp_item_set_name(group, name)
            pdb.gimp_image_insert_layer(self.image, group, translation_group, -1)
            self.layer_groups[key] = group
        
        return translation_group
    
    def preserve_original_layer(self, original_layer):
        """Create backup of original layer"""
        backup_layer = pdb.gimp_layer_copy(original_layer, False)
        pdb.gimp_item_set_name(backup_layer, f"{original_layer.name} (Original)")
        pdb.gimp_image_insert_layer(
            self.image, 
            backup_layer, 
            self.layer_groups["original"], 
            -1
        )
        return backup_layer
    
    def create_inpainted_layer(self, inpainted_image, source_layer):
        """Create layer with inpainted background"""
        # Create new layer
        inpainted_layer = pdb.gimp_layer_new(
            self.image,
            source_layer.width,
            source_layer.height,
            source_layer.type,
            "Cleaned Background",
            source_layer.opacity,
            source_layer.mode
        )
        
        # Copy inpainted image data
        self.copy_array_to_layer(inpainted_image, inpainted_layer)
        
        # Insert into appropriate group
        pdb.gimp_image_insert_layer(
            self.image,
            inpainted_layer,
            self.layer_groups["cleaned"],
            -1
        )
        
        return inpainted_layer
    
    def create_text_layers(self, text_results):
        """Create individual text layers for each translated text"""
        text_layers = []
        
        for i, result in enumerate(text_results):
            # Create text layer
            text_layer = pdb.gimp_text_layer_new(
                self.image,
                result.text,
                result.font_name,
                result.font_size,
                UNIT_PIXEL
            )
            
            # Set properties
            pdb.gimp_item_set_name(text_layer, f"Text {i+1}")
            pdb.gimp_layer_set_offsets(
                text_layer, 
                result.position_x, 
                result.position_y
            )
            
            # Apply styling
            if result.has_outline:
                outline_layer = self.create_text_outline(text_layer, result)
                text_layers.append(outline_layer)
            
            # Insert into text group
            pdb.gimp_image_insert_layer(
                self.image,
                text_layer,
                self.layer_groups["translated"],
                -1
            )
            
            text_layers.append(text_layer)
        
        return text_layers
```

### Progress Tracking and User Feedback

**Progress Dialog System**:
```python
class ProgressTracker:
    def __init__(self, total_steps=5):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names = [
            "Detecting speech bubbles...",
            "Extracting text with OCR...",
            "Translating text...", 
            "Removing original text...",
            "Rendering translated text..."
        ]
        
        # Create progress dialog
        self.progress_dialog = self.create_progress_dialog()
        
    def create_progress_dialog(self):
        """Create GTK progress dialog"""
        dialog = gtk.Dialog("Manga Translation Progress", None, 
                          gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT)
        
        # Add progress bar
        self.progress_bar = gtk.ProgressBar()
        self.progress_bar.set_text("Initializing...")
        dialog.vbox.pack_start(self.progress_bar, True, True, 10)
        
        # Add status label  
        self.status_label = gtk.Label("Starting translation...")
        dialog.vbox.pack_start(self.status_label, True, True, 5)
        
        # Add cancel button
        cancel_button = dialog.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
        cancel_button.connect("clicked", self.on_cancel_clicked)
        
        # Size and show
        dialog.set_default_size(400, 120)
        dialog.show_all()
        
        return dialog
    
    def update_progress(self, step_name=None, progress_fraction=None):
        """Update progress display"""
        if step_name:
            self.status_label.set_text(step_name)
            self.current_step += 1
        
        if progress_fraction is not None:
            total_progress = (self.current_step - 1 + progress_fraction) / self.total_steps
        else:
            total_progress = self.current_step / self.total_steps
        
        self.progress_bar.set_fraction(total_progress)
        self.progress_bar.set_text(f"{int(total_progress * 100)}%")
        
        # Process pending GUI events
        while gtk.events_pending():
            gtk.main_iteration()
    
    def finish(self):
        """Close progress dialog"""
        self.progress_dialog.destroy()
    
    def on_cancel_clicked(self, button):
        """Handle user cancellation"""
        # Set cancellation flag that main process can check
        self.cancelled = True
```

### Settings and Configuration Management

**Persistent Settings System**:
```python
import json
import os
from pathlib import Path

class SettingsManager:
    def __init__(self):
        self.settings_dir = Path.home() / ".gimp-2.10" / "manga-translator"
        self.settings_file = self.settings_dir / "settings.json"
        self.ensure_settings_dir()
        self.load_settings()
    
    def ensure_settings_dir(self):
        """Create settings directory if it doesn't exist"""
        self.settings_dir.mkdir(parents=True, exist_ok=True)
    
    def load_settings(self):
        """Load settings from file or create defaults"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
            except:
                self.settings = self.get_default_settings()
        else:
            self.settings = self.get_default_settings()
    
    def get_default_settings(self):
        """Return default plugin settings"""
        return {
            "translation": {
                "primary_engine": "openai",
                "fallback_engines": ["deepl", "argos"],
                "source_language": "auto",
                "target_language": "en",
                "api_keys": {}
            },
            "ocr": {
                "confidence_threshold": 0.7,
                "enable_vertical_text": True,
                "preprocessing_level": "balanced"
            },
            "inpainting": {
                "method": "neural",
                "quality_level": "balanced",
                "use_gpu": True
            },
            "typesetting": {
                "auto_font_selection": True,
                "preferred_fonts": {
                    "dialogue": "CC Wild Words",
                    "narration": "Lato",
                    "sfx": "Impact"
                },
                "margin_factor": 0.85
            },
            "ui": {
                "show_progress": True,
                "auto_refresh": True,
                "remember_window_position": True
            }
        }
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            gimp.message(f"Failed to save settings: {str(e)}")
    
    def get(self, category, key, default=None):
        """Get setting value"""
        return self.settings.get(category, {}).get(key, default)
    
    def set(self, category, key, value):
        """Set setting value"""
        if category not in self.settings:
            self.settings[category] = {}
        self.settings[category][key] = value
        self.save_settings()
```

### Error Handling and Recovery

**Robust Error Management**:
```python
class ErrorHandler:
    def __init__(self, image):
        self.image = image
        self.undo_stack = []
        
    def safe_execute(self, operation, *args, **kwargs):
        """Execute operation with automatic error recovery"""
        
        # Record state before operation
        checkpoint = self.create_checkpoint()
        
        try:
            result = operation(*args, **kwargs)
            return result
            
        except Exception as e:
            # Error occurred - restore state
            self.restore_checkpoint(checkpoint)
            
            # Log error
            logging.error(f"Operation failed: {str(e)}")
            
            # Show user-friendly error message
            error_dialog = self.create_error_dialog(e)
            error_dialog.run()
            error_dialog.destroy()
            
            # Re-raise for caller to handle
            raise
    
    def create_checkpoint(self):
        """Create restoration point"""
        return {
            'undo_group_id': pdb.gimp_image_undo_group_start(self.image),
            'layer_count': len(self.image.layers),
            'active_layer': self.image.active_layer
        }
    
    def restore_checkpoint(self, checkpoint):
        """Restore to previous state"""
        try:
            # Undo all operations since checkpoint
            while pdb.gimp_image_undo_group_end(self.image):
                pass
                
            # Refresh display
            gimp.displays_flush()
            
        except Exception as e:
            logging.error(f"Failed to restore checkpoint: {str(e)}")
    
    def create_error_dialog(self, error):
        """Create user-friendly error dialog"""
        dialog = gtk.MessageDialog(
            None,
            gtk.DIALOG_MODAL,
            gtk.MESSAGE_ERROR,
            gtk.BUTTONS_OK,
            f"Translation Error: {str(error)}"
        )
        
        # Add helpful suggestions based on error type
        if isinstance(error, APIKeyError):
            dialog.format_secondary_text(
                "Please check your API keys in the plugin settings."
            )
        elif isinstance(error, ModelLoadError):
            dialog.format_secondary_text(
                "Try restarting GIMP or reinstalling the plugin models."
            )
        else:
            dialog.format_secondary_text(
                "Please check the GIMP console for more details."
            )
        
        return dialog
```

### Performance Optimization

**Memory and Processing Optimization**:
```python
class PerformanceOptimizer:
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self.processing_cache = {}
    
    def optimize_image_processing(self, image):
        """Optimize processing based on image size and system resources"""
        
        image_size = image.width * image.height * len(image.layers)
        available_memory = self.memory_monitor.get_available_memory()
        
        if image_size > available_memory * 0.5:
            # Large image - use tile-based processing
            return self.setup_tile_processing(image)
        else:
            # Normal processing
            return self.setup_normal_processing(image)
    
    def setup_tile_processing(self, image):
        """Configure tile-based processing for large images"""
        tile_size = self.calculate_optimal_tile_size(image)
        
        return {
            'processing_mode': 'tiled',
            'tile_size': tile_size,
            'overlap': tile_size // 8,  # 12.5% overlap
            'parallel_tiles': min(4, os.cpu_count())
        }
    
    def manage_model_memory(self, models_needed):
        """Manage ML model loading/unloading"""
        
        current_memory = self.memory_monitor.get_used_memory()
        
        # Unload unused models if memory pressure
        if current_memory > 0.8:  # 80% memory usage
            self.unload_unused_models()
        
        # Load required models
        for model_name in models_needed:
            if model_name not in self.loaded_models:
                self.load_model(model_name)
    
    def cache_intermediate_results(self, operation_id, result):
        """Cache expensive intermediate results"""
        
        # Only cache if we have memory available
        if self.memory_monitor.get_available_memory() > 1024 * 1024 * 100:  # 100MB
            self.processing_cache[operation_id] = result
    
    def get_cached_result(self, operation_id):
        """Retrieve cached result if available"""
        return self.processing_cache.get(operation_id)
```