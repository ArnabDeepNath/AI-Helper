import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'Solve this',
        'images': ['chem_question.png']
    }]
)

print(response)