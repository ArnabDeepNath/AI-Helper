import json
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

Base = declarative_base()

class Question(Base):
    __tablename__ = 'questions'
    
    id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    topic = Column(String(100))
    difficulty = Column(String(50))
    embedding = Column(Text)  # Store embedding as JSON string
    
    options = relationship("Option", back_populates="question", cascade="all, delete-orphan")

class Option(Base):
    __tablename__ = 'options'
    
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey('questions.id'))
    text = Column(Text, nullable=False)
    is_correct = Column(Integer, default=0)
    position = Column(Integer)
    explanation = Column(Text, nullable=True)
    
    question = relationship("Question", back_populates="options")

class QuestionDatabaseManager:
    def __init__(self, db_url: str = "sqlite:///questions.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_question(self, question_data: Dict):
        """Add a new question to the database."""
        session = self.Session()
        try:
            # Create a Question instance
            question = Question(
                question_text=question_data['question'],
                topic=question_data.get('topic', ''),
                difficulty="medium",  # Default difficulty
                embedding=json.dumps(self.compute_embedding(question_data['question']))
            )

            # Create Option instances and append them to the question
            for option in question_data['options']:
                option_instance = Option(
                    question=question,
                    text=option['text'],
                    is_correct=int(option['is_correct']),
                    explanation=option.get('explanation', None)
                )
                question.options.append(option_instance)

            # Add the question to the session and commit
            session.add(question)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error adding question: {e}")
        finally:
            session.close()

    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a question text."""
        return self.embedding_model.encode([text]).tolist()

    def get_all_questions(self) -> List[Dict]:
        """Retrieve all questions with their options and embeddings."""
        session = self.Session()
        questions = []
        try:
            for q in session.query(Question).all():
                options = [{"text": opt.text, "is_correct": opt.is_correct, 
                            "explanation": opt.explanation}
                        for opt in q.options]
                
                questions.append({
                    "id": q.id,
                    "question": q.question_text,
                    "topic": q.topic,
                    "difficulty": q.difficulty,
                    "embedding": json.loads(q.embedding),  # Convert embedding back to list
                    "options": options
                })
        finally:
            session.close()
        return questions

    
    def close(self):
        """Close the database session."""
        self.engine.dispose()  # Properly dispose the engine

# Example usage
if __name__ == "__main__":
    # Example usage with a sample question directly
    sample_question = {
        "question": "What is the force of friction?",
        "topic": "Friction",
        "options": [
            {"text": "Force of friction > applied force", "is_correct": False},
            {"text": "Force of friction < applied force", "is_correct": True},
            {"text": "Force of friction = applied force", "is_correct": False},
            {"text": "There is no friction", "is_correct": False}
        ]
    }

    db_manager = QuestionDatabaseManager()
    db_manager.add_question(sample_question)
    
    questions = db_manager.get_all_questions()
    print(json.dumps(questions, indent=2))

    db_manager.close()
