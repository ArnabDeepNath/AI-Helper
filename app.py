import ollama
import numpy as np
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scrapper import QuestionDatabaseManager


class JEEPrepBotDB:
    def __init__(self, db_url: str = "sqlite:///questions.db"):
        """Initialize the bot with database connection and language model."""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_manager = QuestionDatabaseManager(db_url)

        self.refresh_questions()  # Load questions at initialization

    def answer_question(self, query: str) -> str:
        """Generate response for student query using the language model."""
        # First, check if a similar question exists in the database
        similar_question = self.find_similar_question(query)
        
        if similar_question:
            print('Reached Here')
            return self.solve_question_with_llm(similar_question, query)

        # If no similar question found, generate a response from the model
        return self.generate_response(query)

    def find_similar_question(self, query: str):
        """Check for similar questions in the database."""
        similar_problems = self.find_similar_problems(query)
        return similar_problems[0]["problem"] if similar_problems else None

    def solve_question_with_llm(self, question: Dict, query: str) -> str:
        """Use the model to solve the given question."""
        # Show the similar question first
        similar_question_prompt = f"Similar Question: {question['question']}\nOptions: {question['options']}\n"
        
        # Now, ask the model to solve the new question based on the similar one
        prompt = (
            f"{similar_question_prompt} - This is a similar question, and it provides a reliable solution explanation. "
            f"Use only this explanation to solve the question below, strictly following each step without deviation. "
            f"Do not infer or make assumptions beyond the steps provided in the similar question. "
            f"Question to Solve: {query}. Please provide a one line,  solution that mirrors the explanation above, "
            f"and dont give the final additional explanation."
        )


        return self.generate_response(prompt)   

    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Ollama API and estimate the time taken."""
        start_time = time.time()  # Start the timer
        
        response = ollama.chat(model="qwen2.5:1.5b", messages=[{"role": "user", "content": prompt}])
        
        end_time = time.time()  # End the timer
        response_time = end_time - start_time  # Calculate the time taken
        
        # Prepare the response and include the time estimate
        response_message = response['message']['content']
        time_estimate = f"\nResponse time: {response_time:.2f} seconds."
        
        return response_message + time_estimate

    def refresh_questions(self):
        """Refresh questions from the database."""
        self.questions = self.db_manager.get_all_questions()
        
        # Ensure we have embeddings to process
        if not self.questions:
            self.embeddings = np.array([])  # Handle case where no questions are available
            return
        
        self.embeddings = np.array([q['embedding'] for q in self.questions])

        # Ensure embeddings are 2D
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        elif self.embeddings.ndim > 2:
            self.embeddings = self.embeddings.reshape(self.embeddings.shape[0], -1)

    def find_similar_problems(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar problems based on the query."""
        # Check if there are any embeddings to compare against
        if self.embeddings.size == 0:
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, self.embeddings)

        if similarities.ndim > 1:
            similarities = similarities.flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_problems = []

        for idx in top_indices:
            question = self.questions[idx]
            similar_problems.append({
                "similarity": similarities[idx],
                "problem": {
                    "question": question['question'],
                    "topic": question['topic'],
                    "options": question['options'],
                    "solution_steps": self._generate_solution_steps(question)
                }
            })

        return similar_problems

    def add_sample_question(self):
        """Add a specific question to the database."""
        question_data = {
            "question": "The upper half of an inclined plane with inclination Φ is perfectly smooth, while the lower half is rough. A body starting from rest at the top will again come to rest at the bottom if the coefficient of friction for the lower half is given by?",
            "topic": "Inclined Planes and Friction",
            "correct_answer": "2 tanΦ",
            "options": [
                {"text": "2 sinΦ", "is_correct": False, "explanation": None},
                {"text": "2 cosΦ", "is_correct": False, "explanation": None},
                {"text": "2 tanΦ", "is_correct": True, "explanation": "The coefficient of friction is calculated by equating the work done by friction to the decrease in potential energy as the block slides down."},
                {"text": "tanΦ", "is_correct": False, "explanation": None}
            ]
        }
        
        # Pass the data to the database manager for processing
        self.db_manager.add_question(question_data)


    def _generate_solution_steps(self, question: Dict) -> List[str]:
        """Generate solution steps from question data."""
        correct_option = next((opt for opt in question['options'] if opt['is_correct']), None)
        
        steps = [
            f"Topic: {question['topic']}",
            f"Question: {question['question']}",
            f"Correct Answer: {correct_option['text'] if correct_option else 'Not available'}",
        ]
        
        if correct_option and correct_option['explanation']:
            steps.append(f"Explanation: {correct_option['explanation']}")
        
        return steps

    def close(self):
        """Close database connection."""
        self.db_manager.close()


# Example usage
if __name__ == "__main__":
    bot = JEEPrepBotDB()
    # bot.add_sample_question()
    query = "The upper half of an inclined plane with inclination Φ is perfectly smooth, while the lower half is rough. A body starting from rest at the top will again come to rest at the bottom if the coefficient of friction for the lower half is given by?"
    response = bot.answer_question(query)
    print(response)
    bot.close()
