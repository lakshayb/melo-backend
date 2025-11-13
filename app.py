"""
Melo - The Therapist Chatbox
Flask Backend Application - Railway Deployment
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import os
import logging
import sys

# Import models and NLP engine
from models import db, User, Chatbot, Conversation, Message, EmotionAnalysis, Feedback
from nlp_engine import analyze_and_respond, initialize_nlp

# Configure logging - output to stdout for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration - Railway specific
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    # Railway uses postgres:// which SQLAlchemy 1.4+ doesn't support
    # Convert to postgresql://
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL or 'sqlite:///melo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = os.environ.get('FLASK_ENV') == 'development'
app.config['JSON_SORT_KEYS'] = False

# Security headers
@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# Initialize extensions
db.init_app(app)

# CORS configuration - allow frontend domains
ALLOWED_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, resources={
    r"/api/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

# Create database tables and initialize
with app.app_context():
    db.create_all()
    logger.info("Database tables created/verified")

    # Initialize default chatbot if not exists
    chatbot = Chatbot.query.first()
    if not chatbot:
        chatbot = Chatbot(
            name='Melo',
            model_version='1.0-BERT',
            status='active'
        )
        db.session.add(chatbot)
        db.session.commit()
        logger.info("Default chatbot initialized")

# Initialize NLP model
logger.info("Initializing NLP models...")
try:
    initialize_nlp()
    logger.info("NLP models loaded successfully")
except Exception as e:
    logger.error(f"Error loading NLP models: {e}")
    # Don't crash on model load failure - will load on first request


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'service': 'Melo Chatbot Backend',
        'version': '1.0',
        'environment': os.environ.get('FLASK_ENV', 'production')
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        db.session.close()

        return jsonify({
            'status': 'healthy',
            'service': 'Melo Chatbot',
            'version': '1.0',
            'database': 'connected'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    """
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        user_message = data['message'].strip()

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        if len(user_message) > 1000:
            return jsonify({'error': 'Message too long (max 1000 characters)'}), 400

        # Get or create user
        user_id = data.get('user_id', 1)
        user = User.query.get(user_id)

        if not user:
            user = User(
                username=f'guest_{user_id}',
                email=f'guest_{user_id}@melo.app',
                password_hash=generate_password_hash('guest')
            )
            db.session.add(user)
            db.session.commit()

        # Get default chatbot
        chatbot = Chatbot.query.filter_by(status='active').first()
        if not chatbot:
            return jsonify({'error': 'Chatbot not available'}), 503

        # Get or create conversation
        conversation_id = data.get('conversation_id')
        conversation = None

        if conversation_id:
            conversation = Conversation.query.get(conversation_id)
            if not conversation or conversation.user_id != user.user_id:
                conversation = None

        if not conversation:
            conversation = Conversation(
                user_id=user.user_id,
                chatbot_id=chatbot.chatbot_id,
                status='active'
            )
            db.session.add(conversation)
            db.session.flush()

        # Save user message
        user_msg = Message(
            conversation_id=conversation.conversation_id,
            sender_type='user',
            message_text=user_message,
            message_type='text'
        )
        db.session.add(user_msg)
        db.session.flush()

        # Analyze emotion and generate response
        analysis_result = analyze_and_respond(user_message)

        # Save emotion analysis
        emotion_analysis = EmotionAnalysis(
            message_id=user_msg.message_id,
            detected_emotion=analysis_result['emotion'],
            confidence_score=analysis_result['confidence'],
            secondary_emotions=json.dumps(analysis_result['all_emotions'])
        )
        db.session.add(emotion_analysis)

        # Prepare bot response
        bot_response = analysis_result['response']
        if analysis_result['coping_strategy']:
            bot_response += f"\n\n{analysis_result['coping_strategy']}"

        # Save bot message
        bot_msg = Message(
            conversation_id=conversation.conversation_id,
            sender_type='bot',
            message_text=bot_response,
            message_type='suggestion' if analysis_result['coping_strategy'] else 'text'
        )
        db.session.add(bot_msg)

        # Commit all changes
        db.session.commit()

        return jsonify({
            'reply': bot_response,
            'emotion': analysis_result['emotion'],
            'confidence': round(analysis_result['confidence'], 4),
            'coping_strategy': analysis_result['coping_strategy'],
            'needs_escalation': analysis_result['needs_escalation'],
            'conversation_id': conversation.conversation_id,
            'message_id': bot_msg.message_id
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/conversations/<int:conversation_id>/history', methods=['GET'])
def get_conversation_history(conversation_id):
    """Get conversation history"""
    try:
        conversation = Conversation.query.get(conversation_id)

        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404

        messages = Message.query.filter_by(
            conversation_id=conversation_id
        ).order_by(Message.timestamp).all()

        history = []
        for msg in messages:
            msg_data = {
                'message_id': msg.message_id,
                'sender_type': msg.sender_type,
                'message_text': msg.message_text,
                'timestamp': msg.timestamp.isoformat(),
                'message_type': msg.message_type
            }

            if msg.sender_type == 'user' and msg.emotion_analysis:
                msg_data['emotion'] = {
                    'detected_emotion': msg.emotion_analysis.detected_emotion,
                    'confidence': msg.emotion_analysis.confidence_score
                }

            history.append(msg_data)

        return jsonify({
            'conversation_id': conversation_id,
            'started_at': conversation.started_at.isoformat(),
            'status': conversation.status,
            'message_count': len(history),
            'messages': history
        }), 200

    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/conversations/<int:conversation_id>/end', methods=['POST'])
def end_conversation(conversation_id):
    """End a conversation"""
    try:
        conversation = Conversation.query.get(conversation_id)

        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404

        conversation.status = 'completed'
        conversation.ended_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'message': 'Conversation ended successfully',
            'conversation_id': conversation_id
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error ending conversation: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.get_json()

        if not data or 'user_id' not in data:
            return jsonify({'error': 'User ID is required'}), 400

        feedback = Feedback(
            user_id=data['user_id'],
            conversation_id=data.get('conversation_id'),
            rating=data.get('rating'),
            feedback_text=data.get('feedback_text'),
            helpful=data.get('helpful')
        )

        db.session.add(feedback)
        db.session.commit()

        return jsonify({
            'message': 'Feedback submitted successfully',
            'feedback_id': feedback.feedback_id
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/user/<int:user_id>/conversations', methods=['GET'])
def get_user_conversations(user_id):
    """Get all conversations for a user"""
    try:
        conversations = Conversation.query.filter_by(user_id=user_id).order_by(
            Conversation.started_at.desc()
        ).all()

        result = []
        for conv in conversations:
            msg_count = Message.query.filter_by(conversation_id=conv.conversation_id).count()
            result.append({
                'conversation_id': conv.conversation_id,
                'started_at': conv.started_at.isoformat(),
                'ended_at': conv.ended_at.isoformat() if conv.ended_at else None,
                'status': conv.status,
                'message_count': msg_count
            })

        return jsonify({
            'user_id': user_id,
            'conversation_count': len(result),
            'conversations': result
        }), 200

    except Exception as e:
        logger.error(f"Error getting user conversations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    logger.error(f"Internal error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


# Railway specific: Listen on PORT environment variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Melo on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
