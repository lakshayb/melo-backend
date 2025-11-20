"""
SQLAlchemy Database Models for Melo - The Therapist Chatbox
Implements the complete ERD schema with all entities and relationships
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class User(db.Model):
    """
    User entity - stores user information
    Attributes as per ERD: user_id, username, password_hash, 
    created_at, last_login, preferences
    """
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    preferences = Column(Text)  # JSON string for user preferences

    # Relationships
    conversations = relationship('Conversation', back_populates='user', cascade='all, delete-orphan')
    feedbacks = relationship('Feedback', back_populates='user', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username}>'


class Chatbot(db.Model):
    """
    Chatbot entity - stores chatbot configuration
    Attributes: chatbot_id, name, model_version, created_at, status
    """
    __tablename__ = 'chatbots'

    chatbot_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, default='Melo')
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default='active')  # active, inactive, maintenance

    # Relationships
    conversations = relationship('Conversation', back_populates='chatbot', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Chatbot {self.name} v{self.model_version}>'


class Conversation(db.Model):
    """
    Conversation entity - stores conversation sessions
    Attributes: conversation_id, user_id, chatbot_id, started_at, 
    ended_at, status, conversation_summary
    Relationships: 1-to-N with User, 1-to-N with Chatbot, 1-to-N with Message
    """
    __tablename__ = 'conversations'

    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    chatbot_id = Column(Integer, ForeignKey('chatbots.chatbot_id'), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime)
    status = Column(String(20), default='active')  # active, completed, abandoned
    conversation_summary = Column(Text)

    # Relationships
    user = relationship('User', back_populates='conversations')
    chatbot = relationship('Chatbot', back_populates='conversations')
    messages = relationship('Message', back_populates='conversation', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Conversation {self.conversation_id} - User {self.user_id}>'


class Message(db.Model):
    """
    Message entity - stores individual messages
    Attributes: message_id, conversation_id, sender_type, message_text,
    timestamp, message_type
    Relationships: N-to-1 with Conversation, 1-to-1 with Emotion_Analysis
    """
    __tablename__ = 'messages'

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey('conversations.conversation_id'), nullable=False)
    sender_type = Column(String(10), nullable=False)  # 'user' or 'bot'
    message_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    message_type = Column(String(50))  # text, suggestion, quote, exercise

    # Relationships
    conversation = relationship('Conversation', back_populates='messages')
    emotion_analysis = relationship('EmotionAnalysis', back_populates='message', 
                                   uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Message {self.message_id} from {self.sender_type}>'


class EmotionAnalysis(db.Model):
    """
    Emotion_Analysis entity - stores emotion detection results
    Attributes: analysis_id, message_id, detected_emotion, confidence_score,
    analysis_timestamp
    Relationships: 1-to-1 with Message
    """
    __tablename__ = 'emotion_analysis'

    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey('messages.message_id'), 
                       nullable=False, unique=True)
    detected_emotion = Column(String(50), nullable=False)
    confidence_score = Column(db.Float, nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    secondary_emotions = Column(Text)  # JSON string for other detected emotions

    # Relationships
    message = relationship('Message', back_populates='emotion_analysis')

    def __repr__(self):
        return f'<EmotionAnalysis {self.detected_emotion} ({self.confidence_score:.2f})>'


class Feedback(db.Model):
    """
    Feedback entity - stores user feedback on chatbot responses
    Attributes: feedback_id, user_id, conversation_id, rating, 
    feedback_text, submitted_at
    Relationships: N-to-1 with User
    """
    __tablename__ = 'feedbacks'

    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    conversation_id = Column(Integer, ForeignKey('conversations.conversation_id'))
    rating = Column(Integer)  # 1-5 rating scale
    feedback_text = Column(Text)
    submitted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    helpful = Column(Boolean)  # Was the chatbot helpful?

    # Relationships
    user = relationship('User', back_populates='feedbacks')

    def __repr__(self):
        return f'<Feedback {self.feedback_id} - Rating {self.rating}>'


class Therapist(db.Model):
    """
    Therapist entity - stores information about human therapists for escalation
    Attributes: therapist_id, name, specialization, phone,
    availability_status, created_at
    Note: This is for future enhancement when users need human intervention
    """
    __tablename__ = 'therapists'

    therapist_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False)
    specialization = Column(String(100))
    phone = Column(String(20))
    availability_status = Column(String(20), default='available')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    license_number = Column(String(50))

    def __repr__(self):
        return f'<Therapist {self.name} - {self.specialization}>'
