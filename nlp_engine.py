"""
NLP Engine for Melo - The Therapist Chatbox
Implements emotion detection using Hugging Face transformers
Model: boltuix/bert-emotion (13-class emotion detection)
"""

from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Emotion detection engine using BERT-based emotion classification
    Supports 13 emotions: Love, Happiness, Sadness, Anger, Fear, Surprise,
    Neutral, Disgust, Shame, Guilt, Confusion, Desire, Sarcasm
    """

    def __init__(self, model_name="boltuix/bert-emotion"):
        """
        Initialize the emotion detection model

        Args:
            model_name (str): HuggingFace model name for emotion classification
        """
        try:
            logger.info(f"Loading emotion detection model: {model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                top_k=3  # Get top 3 emotions for better understanding
            )
            logger.info("Emotion detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect_emotion(self, text):
        """
        Detect emotion from user text input

        Args:
            text (str): User's message text

        Returns:
            dict: {
                'primary_emotion': str,
                'confidence': float,
                'all_emotions': list of dicts
            }
        """
        try:
            if not text or not text.strip():
                return {
                    'primary_emotion': 'Neutral',
                    'confidence': 1.0,
                    'all_emotions': [{'label': 'Neutral', 'score': 1.0}]
                }

            # Get emotion predictions
            results = self.classifier(text)

            # Results is a list with one element (list of top emotions)
            emotions = results[0] if isinstance(results[0], list) else results

            primary = emotions[0]

            return {
                'primary_emotion': primary['label'],
                'confidence': primary['score'],
                'all_emotions': emotions
            }

        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return {
                'primary_emotion': 'Neutral',
                'confidence': 0.5,
                'all_emotions': [{'label': 'Neutral', 'score': 0.5}]
            }


class ResponseGenerator:
    """
    Generate empathetic responses based on detected emotions
    Uses rule-based system with emotion-specific response templates
    """

    # Emotion-based response templates
    EMOTION_RESPONSES = {
        'Sadness': [
            "I can sense you're feeling down. It's completely okay to feel sad sometimes. Would you like to talk about what's bothering you?",
            "I hear you, and I'm here for you. Sadness is a natural emotion. What's weighing on your heart right now?",
            "I'm sorry you're going through a difficult time. Remember, it's okay to feel sad. I'm here to listen.",
            "Your feelings are valid. Sometimes we need to acknowledge sadness before we can move forward. Want to share more?"
        ],
        'Anger': [
            "I understand you're feeling frustrated. It's okay to feel angry. Let's talk through what's making you feel this way.",
            "Anger is a valid emotion. Would you like to share what's triggering these feelings? I'm here to listen without judgment.",
            "I can tell something is really bothering you. Taking a few deep breaths might help. What's going on?",
            "It sounds like you're dealing with something really frustrating. I'm here to help you work through it."
        ],
        'Fear': [
            "I hear the worry in your message. Fear can feel overwhelming, but you're not alone. What's making you feel anxious?",
            "It's brave of you to acknowledge your fears. Let's explore what's causing this anxiety together.",
            "Feeling afraid is completely natural. Would you like to talk about what's worrying you?",
            "I understand fear can be paralyzing. Remember to breathe. I'm here to support you through this."
        ],
        'Happiness': [
            "It's wonderful to hear such positive energy from you! What's bringing you joy today?",
            "Your happiness is contagious! I'm so glad you're feeling good. Tell me more!",
            "That's fantastic! It's great to see you in such high spirits. What's going well?",
            "I love your positive energy! Celebrating the good moments is so important. What made your day?"
        ],
        'Love': [
            "That's beautiful! Love is such a powerful emotion. It's wonderful that you're experiencing it.",
            "How heartwarming! Love brings so much light into our lives. Tell me more about what you're feeling.",
            "That's lovely to hear! Expressing love and appreciation is so important. Who or what are you feeling grateful for?",
            "What a beautiful sentiment! Love in all its forms enriches our lives."
        ],
        'Surprise': [
            "Oh wow! That sounds unexpected! How are you processing this surprise?",
            "Life certainly keeps us on our toes! How do you feel about this unexpected turn?",
            "That must have caught you off guard! Tell me more about what happened.",
            "Surprises can be quite something! How are you handling this unexpected situation?"
        ],
        'Neutral': [
            "I'm listening. Feel free to share whatever's on your mind.",
            "I'm here for you. What would you like to talk about today?",
            "Tell me more about what's going on. I'm here to listen and support you.",
            "I'm all ears. What brings you here today?"
        ],
        'Disgust': [
            "I can tell something is really bothering you. Your feelings are completely valid.",
            "That must be really unpleasant to deal with. Would you like to talk about it?",
            "I understand this is difficult for you. Let's work through these feelings together.",
            "It's okay to feel repulsed by things. Your reactions are natural and valid."
        ],
        'Shame': [
            "Please know that everyone makes mistakes. You're being very brave by acknowledging these feelings.",
            "Shame can be such a heavy burden. Remember, you deserve compassion and understanding, especially from yourself.",
            "I appreciate you sharing this with me. It takes courage to confront feelings of shame.",
            "You are not defined by your past mistakes. Let's talk about moving forward with self-compassion."
        ],
        'Guilt': [
            "Guilt shows you care about doing the right thing. Let's explore what you're feeling guilty about.",
            "It's important to acknowledge guilt, but also to forgive yourself. What's troubling you?",
            "Feeling guilty can be painful. Would you like to talk through what happened?",
            "Your conscience is speaking to you. Let's work through these feelings together."
        ],
        'Confusion': [
            "It's okay to feel uncertain. Let's try to untangle what's confusing you.",
            "Confusion is often the first step to clarity. What's puzzling you right now?",
            "I'm here to help you make sense of things. Tell me what's unclear.",
            "Feeling lost is completely normal. Let's work through this together, one step at a time."
        ],
        'Desire': [
            "It's great that you know what you want! Tell me more about your aspirations.",
            "Desire and ambition can be powerful motivators. What are you hoping to achieve?",
            "I hear your passion! What steps can you take toward what you want?",
            "Your enthusiasm is wonderful! Let's explore how you can work toward your goals."
        ],
        'Sarcasm': [
            "I detect some sarcasm there! Sometimes humor helps us cope. What's really on your mind?",
            "I appreciate your wit! But let's dig a bit deeper - how are you really feeling?",
            "Humor can be a shield. I'm here when you're ready to talk about what's underneath.",
            "I hear you! But I'd love to understand what you're truly feeling right now."
        ]
    }

    # Coping suggestions for negative emotions
    COPING_STRATEGIES = {
        'Sadness': [
            "💙 Try the 5-4-3-2-1 grounding technique: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste.",
            "💙 Gentle exercise like a short walk can help lift your mood.",
            "💙 Reach out to someone you trust. Connection can be healing.",
            "💙 Allow yourself to feel - sometimes we need to sit with sadness before it passes."
        ],
        'Anger': [
            "🔥 Try box breathing: Breathe in for 4 counts, hold for 4, exhale for 4, hold for 4.",
            "🔥 Physical activity can help release angry energy safely.",
            "🔥 Write down your feelings without censoring yourself.",
            "🔥 Count backwards from 100 to give yourself time to cool down."
        ],
        'Fear': [
            "🌟 Practice progressive muscle relaxation: tense and release each muscle group.",
            "🌟 Challenge anxious thoughts: What evidence supports and contradicts your worry?",
            "🌟 Stay present with deep breathing exercises.",
            "🌟 Remember: You've overcome challenges before, and you can do it again."
        ],
        'Shame': [
            "💚 Practice self-compassion: Speak to yourself as you would a dear friend.",
            "💚 Remember that mistakes are part of being human.",
            "💚 Write yourself a forgiveness letter.",
            "💚 Focus on growth rather than perfection."
        ],
        'Guilt': [
            "💜 Make amends if possible and appropriate.",
            "💜 Practice self-forgiveness meditation.",
            "💜 Learn from the experience and commit to different choices.",
            "💜 Remember: You are more than your mistakes."
        ]
    }

    def __init__(self):
        """Initialize the response generator"""
        import random
        self.random = random

    def generate_response(self, emotion, confidence, user_message=""):
        """
        Generate an empathetic response based on detected emotion

        Args:
            emotion (str): Detected emotion
            confidence (float): Confidence score
            user_message (str): Original user message for context

        Returns:
            dict: {
                'response': str,
                'coping_strategy': str or None,
                'needs_escalation': bool
            }
        """
        # Check for crisis keywords
        crisis_keywords = ['suicide', 'kill myself', 'end it all', 'don't want to live',
                          'harm myself', 'hurt myself', 'die']
        needs_escalation = any(keyword in user_message.lower() for keyword in crisis_keywords)

        if needs_escalation:
            return {
                'response': "I'm really concerned about what you're sharing. Your safety is the top priority. Please reach out to a crisis helpline immediately:

🆘 National Suicide Prevention Lifeline: 988 (US)
🆘 Crisis Text Line: Text HOME to 741741
🆘 International: findahelpline.com

Please talk to a trained professional who can provide immediate help. Your life matters.",
                'coping_strategy': None,
                'needs_escalation': True
            }

        # Get emotion-specific response
        responses = self.EMOTION_RESPONSES.get(emotion, self.EMOTION_RESPONSES['Neutral'])
        response = self.random.choice(responses)

        # Add coping strategy for negative emotions
        coping_strategy = None
        if emotion in self.COPING_STRATEGIES:
            coping_strategy = self.random.choice(self.COPING_STRATEGIES[emotion])

        return {
            'response': response,
            'coping_strategy': coping_strategy,
            'needs_escalation': False
        }


# Initialize global instances
emotion_detector = None
response_generator = ResponseGenerator()

def initialize_nlp():
    """Initialize NLP models - call this when app starts"""
    global emotion_detector
    if emotion_detector is None:
        emotion_detector = EmotionDetector()
    return emotion_detector

def analyze_and_respond(user_message):
    """
    Main function to analyze emotion and generate response

    Args:
        user_message (str): User's input text

    Returns:
        dict: Complete analysis and response
    """
    global emotion_detector

    if emotion_detector is None:
        emotion_detector = initialize_nlp()

    # Detect emotion
    emotion_result = emotion_detector.detect_emotion(user_message)

    # Generate response
    response_data = response_generator.generate_response(
        emotion_result['primary_emotion'],
        emotion_result['confidence'],
        user_message
    )

    return {
        'emotion': emotion_result['primary_emotion'],
        'confidence': emotion_result['confidence'],
        'all_emotions': emotion_result['all_emotions'],
        'response': response_data['response'],
        'coping_strategy': response_data['coping_strategy'],
        'needs_escalation': response_data['needs_escalation']
    }
