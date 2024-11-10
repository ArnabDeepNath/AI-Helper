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
        f"Here is a similar question with a solution method: {similar_question_prompt}. "
        f"This solution provides a reliable method for answering questions of this type. "
        f"To solve the new question, carefully analyze both the similarities and differences between the similar question and the new question. "
        f"Pay close attention to the specific numbers, coefficients, and conditions in the new question, as these may require an adjustment in the solution method. "
        f"Do not simply replicate the same method; instead, adapt it based on any new variables, conditions, or parameters present in the new question. "
        f"Take into account all the changes and ensure your answer reflects those correctly. "
        f"Question to Solve: {query}. "
        f"Please provide a one-line solution that follows the same logic as the similar question but has been modified to fit the new details accurately."
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
        question_data = {"question": "20 mL of 0.1 M NH 4 OH is mixed with 40 mL of 0.05 M HCl. The pH of the mixture is nearest to ?", "topic": "Chemical Reactions and Stoichiometry", "correct_answer": "6.2", "options": [{"text": "4.2", "is_correct": False, "explanation": None}, {"text": "5.2", "is_correct": True, "explanation": " \\(\x08egin{array}{l}\x08egin{matrix}NH_4OH & + & HCl & \\longrightarrow & NH_4Cl+H_2O \\20\\ mL, 0.1\\ M & & 40\\ mL, 0.05\\ M & & \\2\\ mmoles & & 2\\ mmoles & & 2\\ mmoles \\\\end{matrix}\\end{array} \\) , \\(\x08egin{array}{l} =\x0crac{1}{2}\\left[14-5-\\left(-1.48\right)\right] = 5.24\\end{array} \\)"}, {"text": "6.2", "is_correct": False, "explanation": None}]}
        
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
    query = "The interhalogen compound formed from the reaction of bromine with excess of fluorine is a :"


    response = bot.answer_question(query)
    print(response)
    bot.close()
