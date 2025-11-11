"""
Melo NLP Engine - Lightweight ML Version
Uses transformers with ONNX Runtime (no torch needed)
"""

import logging

logger = logging.getLogger(__name__)

# Global model instance
emotion_classifier = None


def initialize_nlp():
    """Initialize the NLP model - loads on app startup"""
    global emotion_classifier

    try:
        from transformers import pipeline
        logger.info("Loading emotion detection model...")

        # Use smaller, faster model
        emotion_classifier = pipeline(
            "text-classification",
            model="michellejieli/emotion_text_classification",  # Lighter than BERT
            device=-1  # CPU only
        )
        logger.info("✓ ML model loaded successfully")
        return True

    except Exception as e:
        logger.warning(f"Could not load ML model: {e}")
        logger.info("Falling back to pattern-based detection")
        return False


def analyze_and_respond(user_message):
    """
    Analyze emotion and generate response
    Uses ML model if available, falls back to patterns
    """
    global emotion_classifier

    # Ensure model is initialized
    if emotion_classifier is None:
        initialize_nlp()

    # Crisis detection first
    crisis_keywords = ["suicide", "kill myself", "end it all", "do not want to live",
                       "harm myself", "hurt myself", "self harm"]

    message_lower = user_message.lower()
    needs_escalation = any(kw in message_lower for kw in crisis_keywords)

    if needs_escalation:
        return {
            'emotion': 'Crisis',
            'confidence': 1.0,
            'all_emotions': [],
            'response': "I'm really concerned about your safety. Please reach out immediately: 988 (US) or Crisis Text Line: 741741. Your life matters.",
            'coping_strategy': None,
            'needs_escalation': True
        }

    # Try ML model first
    if emotion_classifier:
        try:
            result = emotion_classifier(user_message[:512])  # Truncate to 512 chars
            primary_emotion = result[0]['label']
            confidence = result[0]['score']
        except Exception as e:
            logger.warning(f"ML model error: {e}, falling back to patterns")
            primary_emotion, confidence = detect_emotion_pattern(message_lower)
    else:
        primary_emotion, confidence = detect_emotion_pattern(message_lower)

    # Response templates
    responses = {
        'joy': "That's wonderful! I'm so glad you're feeling good. Keep spreading that positive energy!",
        'sadness': "I'm sorry you're going through a difficult time. It's okay to feel sad. I'm here to listen.",
        'anger': "I can sense your frustration. That's a valid emotion. Would you like to talk about what's bothering you?",
        'fear': "I understand you're feeling anxious. You're not alone. Let's work through this together.",
        'surprise': "Wow, that's quite unexpected! How are you feeling about it?",
        'neutral': "I'm here to listen. Tell me what's on your mind.",
    }

    emotion_key = primary_emotion.lower()
    response = responses.get(emotion_key, responses['neutral'])

    # Coping strategies
    coping = {
        'sadness': "Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        'anger': "Take 5 deep breaths. In for 4 counts, hold for 4, out for 4. This helps calm your nervous system.",
        'fear': "Remember: You've overcome challenges before. You have the strength to handle this too.",
    }

    coping_strategy = coping.get(emotion_key)

    return {
        'emotion': primary_emotion,
        'confidence': confidence,
        'all_emotions': [],
        'response': response,
        'coping_strategy': coping_strategy,
        'needs_escalation': False
    }


def detect_emotion_pattern(message_lower):
    """Fallback: Pattern-based emotion detection"""
    emotion_patterns = {
        'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'love', 'amazing', 'good'],
        'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'down', 'lonely', 'blue', 'hurt'],
        'anger': ['angry', 'furious', 'mad', 'frustrated', 'annoyed', 'irritated', 'rage'],
        'fear': ['scared', 'anxious', 'nervous', 'worried', 'afraid', 'terrified', 'panic'],
        'neutral': []
    }

    scores = {}
    for emotion, keywords in emotion_patterns.items():
        score = sum(1 for kw in keywords if kw in message_lower)
        scores[emotion] = score

    primary_emotion = max(scores, key=scores.get)
    confidence = 0.7 if scores[primary_emotion] > 0 else 0.5

    return primary_emotion, confidence
