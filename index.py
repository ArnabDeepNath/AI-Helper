import ollama

class OllamaTextGenerator:
    def __init__(self, model_name="qwen2.5:1.5b"):
        # Initialize the Ollama model
        self.model_name = model_name

    def generate_text(self, prompt: str, max_tokens: int = 100):
        # Generate text using the Ollama API
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        
        # Extract the generated text from the response

        # print(response)
        generated_answer = response['message']['content']
        return generated_answer

# Example usage
if __name__ == "__main__":
    # Create an instance of OllamaTextGenerator
    ollama_bot = OllamaTextGenerator()

    # Take user input as a prompt
    prompt = input("Enter a prompt for text generation: ")

    # Generate text
    response = ollama_bot.generate_text(prompt, max_tokens=100)
    
    # Display the generated text
    print("\nGenerated Text:")
    print(response)
