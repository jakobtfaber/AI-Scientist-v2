"""Fallback strategies for VLM failures"""

import logging
from typing import List, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class VLMFallbackHandler:
    """Handle VLM failures by trying multiple models and fallback strategies"""
    
    def __init__(self, models: Optional[List[str]] = None):
        """
        Initialize fallback handler
        
        Args:
            models: List of VLM models to try in order. Defaults to:
                    ["gemini-2.5-flash", "gpt-4o", "claude-3.5-sonnet"]
        """
        self.models = models or [
            "gemini-2.5-flash",
            "gpt-4o",
            "claude-3.5-sonnet"
        ]
        self.attempt_log = []
    
    def get_response_robust(
        self,
        image_path: str,
        prompt: str,
        min_response_length: int = 50,
        max_retries: int = 2
    ) -> str:
        """
        Try multiple VLMs until getting a valid response
        
        Args:
            image_path: Path to image file
            prompt: Prompt for VLM
            min_response_length: Minimum acceptable response length
            max_retries: Number of retries per model
            
        Returns:
            Valid VLM response
            
        Raises:
            VLMFallbackError: If all attempts fail
        """
        from ai_scientist.vlm import get_vlm_response
        
        for model in self.models:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting VLM call with {model} (attempt {attempt + 1}/{max_retries})")
                    
                    response = get_vlm_response(
                        image_path=image_path,
                        prompt=prompt,
                        model=model
                    )
                    
                    if self.is_valid_response(response, min_response_length):
                        logger.info(f"✓ Success with {model}")
                        self.attempt_log.append({
                            "model": model,
                            "attempt": attempt + 1,
                            "status": "success",
                            "response_length": len(response)
                        })
                        return response
                    else:
                        logger.warning(
                            f"✗ {model} returned short/empty response: "
                            f"'{response[:100]}...' (len={len(response)})"
                        )
                        self.attempt_log.append({
                            "model": model,
                            "attempt": attempt + 1,
                            "status": "invalid_response",
                            "response_length": len(response) if response else 0
                        })
                
                except Exception as e:
                    logger.error(f"✗ {model} failed with error: {e}")
                    self.attempt_log.append({
                        "model": model,
                        "attempt": attempt + 1,
                        "status": "error",
                        "error": str(e)
                    })
                
                # Brief delay before retry
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        # All VLMs failed - try OCR fallback
        logger.warning("All VLM attempts failed. Trying OCR fallback...")
        try:
            fallback_response = self.ocr_fallback(image_path, prompt)
            if fallback_response:
                logger.info("✓ OCR fallback successful")
                return fallback_response
        except Exception as e:
            logger.error(f"✗ OCR fallback failed: {e}")
        
        # Complete failure
        error_summary = self._format_error_summary()
        raise VLMFallbackError(
            f"All VLM attempts failed after trying {len(self.models)} models.\n"
            f"{error_summary}"
        )
    
    def is_valid_response(self, response: Optional[str], min_length: int) -> bool:
        """
        Check if VLM response is valid
        
        Args:
            response: VLM response text
            min_length: Minimum acceptable length
            
        Returns:
            True if response is valid
        """
        if not response:
            return False
        
        response = response.strip()
        
        # Check minimum length
        if len(response) < min_length:
            return False
        
        # Check for common failure patterns
        failure_patterns = [
            "I cannot",
            "I'm unable to",
            "I don't have access",
            "cannot process",
            "error occurred",
            "[STOP]",
            "[EMPTY]"
        ]
        
        response_lower = response.lower()
        if any(pattern.lower() in response_lower for pattern in failure_patterns):
            return False
        
        return True
    
    def ocr_fallback(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Fallback: Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            prompt: Original prompt (used for context)
            
        Returns:
            Structured description of image or None if OCR fails
        """
        try:
            import pytesseract
            from PIL import Image
            
            logger.info(f"Running OCR on {image_path}")
            
            # Open image
            img = Image.open(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(img)
            
            if not text or len(text.strip()) < 10:
                logger.warning("OCR extracted minimal text")
                return None
                
            # Format as structured response
            response = f"""
[OCR Fallback Response]

Extracted text from image:
{text}

Note: This is an OCR-based extraction as VLM analysis failed. 
Manual review recommended for accuracy.
"""
            return response
            
        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR fallback")
            return None
        except Exception as e:
            logger.error(f"OCR fallback error: {e}")
            return None
    
    def _format_error_summary(self) -> str:
        """Format summary of all attempts for error message"""
        lines = ["Attempt Log:"]
        for i, attempt in enumerate(self.attempt_log, 1):
            model = attempt["model"]
            status = attempt["status"]
            if status == "error":
                lines.append(f"  {i}. {model}: ERROR - {attempt.get('error', 'unknown')}")
            elif status == "invalid_response":
                length = attempt.get("response_length", 0)
                lines.append(f"  {i}. {model}: Invalid response (length={length})")
            else:
                lines.append(f"  {i}. {model}: {status}")
        return "\n".join(lines)
    
    def get_attempt_summary(self) -> Dict[str, Any]:
        """Get summary of all attempts for logging/debugging"""
        return {
            "total_attempts": len(self.attempt_log),
            "models_tried": list(set(a["model"] for a in self.attempt_log)),
            "attempts": self.attempt_log
        }


class VLMFallbackError(Exception):
    """Raised when all VLM fallback attempts fail"""
    pass
