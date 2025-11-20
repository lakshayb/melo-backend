"""
Melo - Advanced with User Context Adaptation
Flask Backend with Context-Aware NLP
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import logging
import sys
from datetime import datetime, timedelta


from models import db, User, Chatbot, Conversation, Message, EmotionAnalysis
from nlp_engine import analyze_and_respond, initialize_nlp

# Logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Config
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL or 'sqlite:///melo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

ALLOWED_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, resources={
    r"/api/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

db.init_app(app)

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return response

# Initialize
with app.app_context():
    db.create_all()
    chatbot = Chatbot.query.first()
    if not chatbot:
        chatbot = Chatbot(name='Melo', model_version='2.0-Advanced', status='active')
        db.session.add(chatbot)
        db.session.commit()

    initialize_nlp()


# ==================== AUTH ====================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User signup"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if len(username) < 3 or len(password) < 6:
            return jsonify({'error': 'Invalid credentials'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username exists'}), 409

        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        logger.info(f"User signed up: {username}")
        return jsonify({'message': 'Success', 'user_id': user.user_id, 'username': user.username}), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Signup error: {e}")
        return jsonify({'error': 'Signup failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify({'message': 'Login successful', 'user_id': user.user_id, 'username': user.username}), 200

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


# ==================== CHAT ====================

@app.route('/api/health', methods=['GET'])
def health():
    try:
        db.session.execute('SELECT 1')
        return jsonify({'status': 'healthy', 'database': 'connected'}), 200
    except:
        return jsonify({'status': 'unhealthy'}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with context learning"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')

        if not user_id or not user_message or len(user_message) > 1000:
            return jsonify({'error': 'Invalid input'}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get/create conversation
        if conversation_id:
            conversation = Conversation.query.get(conversation_id)
            if not conversation or conversation.user_id != user_id:
                conversation = None
        else:
            conversation = None

        if not conversation:
            chatbot = Chatbot.query.filter_by(status='active').first()
            conversation = Conversation(user_id=user_id, chatbot_id=chatbot.chatbot_id, status='active')
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

        # Analyze with user context (PASS USER_ID FOR CONTEXT LEARNING)
        analysis = analyze_and_respond(user_message, user_id=user_id, db=db)

        # Save emotion
        emotion_analysis = EmotionAnalysis(
            message_id=user_msg.message_id,
            detected_emotion=analysis['emotion'],
            confidence_score=analysis['confidence']
        )
        db.session.add(emotion_analysis)

        # Save bot response
        bot_msg = Message(
            conversation_id=conversation.conversation_id,
            sender_type='bot',
            message_text=analysis['response'],
            message_type='suggestion' if analysis['coping_strategy'] else 'text'
        )
        db.session.add(bot_msg)
        db.session.commit()

        return jsonify({
            'reply': analysis['response'],
            'emotion': analysis['emotion'],
            'confidence': round(analysis['confidence'], 4),
            'coping_strategy': analysis['coping_strategy'],
            'needs_escalation': analysis['needs_escalation'],
            'conversation_id': conversation.conversation_id
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Chat failed'}), 500


# ==================== HISTORY ====================

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    try:
        user_id = request.args.get('user_id', type=int)
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400

        conversations = Conversation.query.filter_by(user_id=user_id).order_by(
            Conversation.started_at.desc()
        ).all()

        result = [
            {
                'conversation_id': c.conversation_id,
                'started_at': c.started_at.isoformat(),
                'status': c.status,
                'message_count': Message.query.filter_by(conversation_id=c.conversation_id).count()
            }
            for c in conversations
        ]

        return jsonify({'conversations': result}), 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': 'Failed'}), 500


@app.route('/api/conversations/<int:cid>/messages', methods=['GET'])
def get_messages(cid):
    try:
        messages = Message.query.filter_by(conversation_id=cid).order_by(Message.timestamp).all()

        result = [
            {
                'sender_type': m.sender_type,
                'message_text': m.message_text,
                'timestamp': m.timestamp.isoformat(),
                'emotion': m.emotion_analysis.detected_emotion if m.emotion_analysis else None
            }
            for m in messages
        ]

        return jsonify({'messages': result}), 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': 'Failed'}), 500
@app.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a specific conversation"""
    try:
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Delete all messages in the conversation
        Message.query.filter_by(conversation_id=conversation_id).delete()
        
        # Delete the conversation
        db.session.delete(conversation)
        db.session.commit()
        
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/cleanup', methods=['DELETE'])
def cleanup_old_conversations():
    """Delete conversations older than specified days"""
    try:
        days = request.args.get('days', 7, type=int)
        user_id = request.args.get('user_id', type=int)
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Find old conversations
        old_conversations = Conversation.query.filter(
            Conversation.user_id == user_id,
            Conversation.created_at < cutoff_date
        ).all()
        
        count = 0
        for conv in old_conversations:
            Message.query.filter_by(conversation_id=conv.id).delete()
            db.session.delete(conv)
            count += 1
        
        db.session.commit()
        
        return jsonify({'deleted': count}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
