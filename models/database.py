import sqlalchemy as sa
from sqlalchemy import create_engine, JSON  # Changed: Import JSON instead of JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

from config.settings import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# def get_db_session():
#     """Get a database session"""
#     session = SessionLocal()
#     try:
#         yield session
#     finally:
#         session.close()

def get_db_session():
    """Get a database session"""
    return SessionLocal()

class DocumentChunk(Base):
    """Model for storing document chunks"""
    __tablename__ = "document_chunks"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    chunk_id = sa.Column(sa.String, unique=True, index=True, nullable=False)
    text = sa.Column(sa.Text, nullable=False)
    source = sa.Column(sa.String, nullable=False)
    file_type = sa.Column(sa.String, nullable=False)
    file_hash = sa.Column(sa.String, nullable=False)
    chunk_hash = sa.Column(sa.String, nullable=False)
    chunk_length = sa.Column(sa.Integer, nullable=False)
    # FIXED: Changed from JSONB to JSON for SQLite compatibility
    meta_data = sa.Column(JSON, nullable=True)
    created_at = sa.Column(sa.DateTime, default=sa.func.now())
    updated_at = sa.Column(sa.DateTime, default=sa.func.now(), onupdate=sa.func.now())

class QueryHistory(Base):
    """Model for storing query history"""
    __tablename__ = "query_history"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    query = sa.Column(sa.Text, nullable=False)
    strategy = sa.Column(sa.String, nullable=False)
    results_count = sa.Column(sa.Integer, nullable=False)
    processing_time = sa.Column(sa.Float, nullable=False)
    user_id = sa.Column(sa.String, nullable=True)
    created_at = sa.Column(sa.DateTime, default=sa.func.now())

class UserFeedback(Base):
    """Model for storing user feedback"""
    __tablename__ = "user_feedback"
    
    id = sa.Column(sa.Integer, primary_key=True, index=True)
    query_id = sa.Column(sa.Integer, sa.ForeignKey('query_history.id'), nullable=False)
    rating = sa.Column(sa.Integer, nullable=False)  # 1-5 scale
    feedback = sa.Column(sa.Text, nullable=True)
    user_id = sa.Column(sa.String, nullable=True)
    created_at = sa.Column(sa.DateTime, default=sa.func.now())

# Create tables
Base.metadata.create_all(bind=engine)