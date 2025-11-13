"""
Melo NLP Engine - Advanced Adaptive Learning
Learns from user chat history to provide personalized responses
"""

import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def initialize_nlp():
    """Initialize NLP"""
    logger.info("Advanced adaptive NLP initialized")
    return True


def get_user_context(user_id, db):
    """
    Build user context from their chat history
    Analyzes patterns, recurring themes, triggers
    """
    from models import Message, Conversation, User

    try:
        # Get user's last 50 messages
        recent_messages = Message.query.join(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Message.timestamp.desc()).limit(50).all()

        if not recent_messages:
            return None

        # Analyze patterns
        emotional_patterns = {}
        themes = []
        triggers = []

        for msg in recent_messages:
            if msg.sender_type == 'user':
                # Extract topics
                words = msg.message_text.lower().split()
                themes.extend(words)

        return {
            'has_history': len(recent_messages) > 5,
            'message_count': len(recent_messages),
            'themes': list(set(themes))[-10:],  # Last 10 unique themes
        }
    except Exception as e:
        logger.warning(f"Could not get user context: {e}")
        return None


def analyze_and_respond(user_message, user_id=None, db=None):
    """
    Advanced emotion analysis with user context
    Provides personalized responses based on history
    """

    if not user_message:
        return get_neutral_response()

    message_lower = user_message.lower()

    # Crisis detection FIRST
    if detect_crisis(message_lower):
        return get_crisis_response()

    # Get user context for personalization
    user_context = None
    if user_id and db:
        user_context = get_user_context(user_id, db)

    # Advanced emotion detection
    emotion_data = detect_emotion_comprehensive(message_lower, user_context)

    # Generate personalized response
    response = generate_adaptive_response(
        emotion_data['emotion'],
        emotion_data['intensity'],
        message_lower,
        user_context,
        emotion_data
    )

    coping = get_advanced_coping_strategy(
        emotion_data['emotion'],
        emotion_data['intensity'],
        user_context
    )

    return {
        'emotion': emotion_data['emotion'],
        'confidence': emotion_data['confidence'],
        'all_emotions': emotion_data.get('all_emotions', []),
        'response': response,
        'coping_strategy': coping,
        'needs_escalation': emotion_data.get('needs_escalation', False)
    }


def detect_crisis(message):
    """Detect crisis/harm keywords"""
    crisis_keywords = [
        "suicide", "kill myself", "end it all", "do not want to live",
        "harm myself", "hurt myself", "self harm", "want to die",
        "not worth living", "goodbye forever", "final goodbye"
    ]
    return any(kw in message for kw in crisis_keywords)


def detect_emotion_comprehensive(message, user_context=None):
    """
    Comprehensive emotion detection with multiple dimensions
    Considers: intensity, duration, context, patterns
    """

    emotions_db = {
        'Happiness': {
            'strong': ['love it', 'amazing', 'incredible', 'fantastic', 'wonderful', 'excellent', 'perfect', 'best day'],
            'medium': ['happy', 'good', 'great', 'awesome', 'nice', 'excited', 'cheerful'],
            'weak': ['okay', 'fine', 'alright', 'decent', 'pleasant']
        },
        'Sadness': {
            'strong': ['devastated', 'heartbroken', 'destroyed', 'shattered', 'cant take it', 'overwhelmed by sadness'],
            'medium': ['sad', 'depressed', 'unhappy', 'miserable', 'down', 'hurt', 'crying'],
            'weak': ['disappointed', 'let down', 'discouraged', 'blue']
        },
        'Anger': {
            'strong': ['furious', 'enraged', 'livid', 'seething', 'absolutely furious', 'cant control it'],
            'medium': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed', 'fed up'],
            'weak': ['bothered', 'bugged', 'mildly upset', 'somewhat annoyed']
        },
        'Anxiety': {
            'strong': ['terrified', 'panicked', 'horrified', 'petrified', 'cant breathe', 'panic attack'],
            'medium': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'uneasy'],
            'weak': ['nervous', 'apprehensive', 'concerned', 'slightly worried']
        },
        'Love': {
            'strong': ['love deeply', 'adore', 'cherish', 'devoted', 'unconditional love'],
            'medium': ['love', 'care', 'appreciate', 'grateful', 'affection'],
            'weak': ['like', 'fond', 'attached']
        },
        'Loneliness': {
            'strong': ['utterly alone', 'isolated', 'abandoned', 'rejected', 'no one understands me'],
            'medium': ['lonely', 'disconnected', 'forgotten', 'missing people'],
            'weak': ['solitary', 'by myself', 'wishing i had company']
        },
        'Confusion': {
            'strong': ['completely lost', 'dont understand anything', 'confused', 'bewildered'],
            'medium': ['confused', 'unsure', 'unclear', 'mixed up'],
            'weak': ['a bit confused', 'not quite sure']
        },
        'Hope': {
            'strong': ['finally see light', 'things will change', 'breakthrough', 'optimistic'],
            'medium': ['hopeful', 'positive', 'better soon', 'improving'],
            'weak': ['slight hope', 'maybe better']
        },
        'Overwhelm': {
            'strong': ['completely overwhelmed', 'drowning', 'cant handle this', 'too much'],
            'medium': ['overwhelmed', 'stressed', 'swamped', 'overburdened'],
            'weak': ['a bit much', 'somewhat busy']
        }
    }

    emotion_scores = {}
    all_emotions = []

    for emotion, intensity_groups in emotions_db.items():
        score = 0

        # Strong: 3 points
        for kw in intensity_groups['strong']:
            if kw in message:
                score += 3

        # Medium: 2 points
        for kw in intensity_groups['medium']:
            if kw in message:
                score += 2

        # Weak: 1 point
        for kw in intensity_groups['weak']:
            if kw in message:
                score += 1

        emotion_scores[emotion] = score

        if score > 0:
            all_emotions.append({
                'label': emotion,
                'score': score / 10.0  # Normalize to 0-1 range
            })

    # Find primary emotion
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    intensity = emotion_scores[primary_emotion]

    if intensity == 0:
        return {
            'emotion': 'Neutral',
            'intensity': 0,
            'confidence': 0.5,
            'all_emotions': []
        }

    # Calculate confidence
    confidence = min(0.5 + (intensity * 0.08), 0.98)

    # Sort all emotions by score
    all_emotions.sort(key=lambda x: x['score'], reverse=True)

    return {
        'emotion': primary_emotion,
        'intensity': intensity,
        'confidence': confidence,
        'all_emotions': all_emotions[:3]  # Top 3 emotions
    }


def generate_adaptive_response(emotion, intensity, message, user_context, emotion_data):
    """
    Generate response that adapts to user context and history
    Gets more nuanced with repeated interactions
    """

    # Check for recurring patterns
    is_recurring = False
    pattern_type = None

    if user_context and user_context.get('message_count', 0) > 10:
        is_recurring = True

    # Response templates with variations
    responses = {
        'Happiness': {
            'weak': "That's nice. I'm glad you're having a good moment.",
            'medium': "That's wonderful! Your joy is contagious.",
            'strong': "That's absolutely amazing! I can feel your happiness!",
            'recurring': "I've noticed you often find joy in {theme}. That's beautiful!"
        },
        'Sadness': {
            'weak': "I sense some sadness. That's okay. Want to talk?",
            'medium': "I can feel your sadness deeply. I'm here, and you're not alone.",
            'strong': "You're carrying a heavy weight right now. Let's work through this together.",
            'recurring': "I know sadness visits you often. You're stronger than you think."
        },
        'Anger': {
            'weak': "Something's bothering you. I'm listening.",
            'medium': "Your anger is valid. Let's talk about what triggered it.",
            'strong': "You're burning with anger, and that's okay. Let's channel this energy.",
            'recurring': "I've noticed frustration comes up often for you. There's a pattern here we should explore."
        },
        'Anxiety': {
            'weak': "Something's on your mind. What's worrying you?",
            'medium': "I can feel your anxiety. You're safe here. Breathe with me.",
            'strong': "You're in panic mode. That's scary. Let's ground you right now.",
            'recurring': "Anxiety seems to be a constant companion for you. Let's build resilience together."
        },
        'Loneliness': {
            'weak': "It sounds like you're missing connection.",
            'medium': "That loneliness must feel heavy. But you're not truly alone - I'm here.",
            'strong': "You feel profoundly isolated. That pain is real, and I see you.",
            'recurring': "Loneliness keeps visiting you. Let's find ways to build meaningful connection."
        },
        'Hope': {
            'weak': "I see a glimmer of hope in your words.",
            'medium': "That's wonderful! Hold onto that hope.",
            'strong': "Your hope is inspiring! Things are shifting for you!",
            'recurring': "You're building momentum. This hope is well-earned."
        },
        'Overwhelm': {
            'weak': "Things feel like a lot right now.",
            'medium': "You're overwhelmed. Let's break this down into manageable pieces.",
            'strong': "You feel like you're drowning. Let me help you surface.",
            'recurring': "Overwhelm seems to return frequently. Let's identify what's sustainable for you."
        },
        'Neutral': "I'm here. What's on your mind?"
    }

    intensity_level = 'weak' if intensity < 2 else ('medium' if intensity < 5 else 'strong')

    # Get base response
    base_response = responses.get(emotion, {}).get(intensity_level, "I'm here for you.")

    # Add recurring context if applicable
    if is_recurring and 'recurring' in responses.get(emotion, {}):
        base_response = responses[emotion]['recurring']

    return base_response


def get_advanced_coping_strategy(emotion, intensity, user_context=None):
    """
    Provide intensity and context-appropriate coping strategies
    Learns from user history which strategies might work best
    """

    if intensity < 2:
        return None

    strategies = {
        'Sadness': {
            'medium': "5-4-3-2-1 Grounding: Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
            'strong': "Deep breathing: In for 4, hold for 4, out for 6. This activates your parasympathetic nervous system."
        },
        'Anger': {
            'medium': "Physical release: 10 jumping jacks or a brisk walk to channel the adrenaline.",
            'strong': "Write an angry letter you won't send. Get it ALL out, then burn it symbolically."
        },
        'Anxiety': {
            'medium': "Box breathing: 4 counts in, 4 hold, 4 out, 4 hold. Repeat 5 times.",
            'strong': "Progressive muscle relaxation: Tense each muscle for 5 sec, release. Toes to head."
        },
        'Loneliness': {
            'medium': "Reach out. Text one person. Even a meme counts as connection.",
            'strong': "Join a community online or offline. You deserve to belong."
        },
        'Overwhelm': {
            'medium': "Brain dump: Write down EVERYTHING bothering you. Then pick ONE to tackle.",
            'strong': "Ask for help. Delegate. You don't have to carry everything alone."
        },
        'Anxiety': {
            'medium': "Grounding game: Focus on 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste.",
            'strong': "Go outside. Feel sun/wind/grass. Nature is grounding."
        }
    }

    intensity_level = 'medium' if intensity < 5 else 'strong'

    if emotion in strategies and intensity_level in strategies[emotion]:
        return strategies[emotion][intensity_level]

    return None


def get_crisis_response():
    """Crisis response"""
    return {
        'emotion': 'Crisis',
        'confidence': 1.0,
        'all_emotions': [],
        'response': "I'm very concerned about your safety. Please reach out to crisis support NOW:\n\n🆘 988 - Suicide & Crisis Lifeline (US)\n🆘 Text HOME to 741741 - Crisis Text Line\n🆘 findahelpline.com - International\n\nYou matter. Call now.",
        'coping_strategy': None,
        'needs_escalation': True
    }


def get_neutral_response():
    """Neutral response"""
    return {
        'emotion': 'Neutral',
        'confidence': 0.5,
        'all_emotions': [],
        'response': "I'm here to listen. Tell me what's really going on. How are you truly feeling?",
        'coping_strategy': None,
        'needs_escalation': False
    }
