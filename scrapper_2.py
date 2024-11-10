from bs4 import BeautifulSoup
import re

def parse_html_to_question_data(html):
    """Parse HTML input to create a question data dictionary."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract the question text
    question_text = soup.find("p").get_text(separator=" ")
    equation_div = soup.find("div", class_="mathjax-scroll")
    if equation_div:
        question_text += f" {equation_div.get_text()}"

    # Extract the answer explanation if available
    explanation = None
    solution_div = soup.find_all("div", class_="mathjax-scroll")[-1]
    if solution_div:
        explanation = solution_div.get_text()

    # Extract the correct answer
    correct_answer_match = re.search(r"Answer\s\((\w)\)", html)
    correct_answer_letter = correct_answer_match.group(1) if correct_answer_match else None
    
    # Extract the options
    options_text = [p.get_text() for p in soup.find_all("p")[3:7]]
    options = []
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    for i, opt in enumerate(options_text):
        option_letter = chr(65 + i)
        option_text = opt[3:].strip()
        options.append({
            "text": option_text,
            "is_correct": (option_letter == correct_answer_letter),
            "explanation": explanation if (option_letter == correct_answer_letter) else None
        })

    # Build the question data dictionary
    question_data = {
        "question": question_text,
        "topic": "Chemical Reactions and Stoichiometry",  # Topic can be customized as needed
        "correct_answer": options[answer_map[correct_answer_letter]]["text"],
        "options": options
    }

    return question_data

# Sample HTML input
html_input = """
<p><strong>1. SO<sub>2</sub>Cl<sub>2</sub> on reaction with excess of water results into acidic mixture</strong></p>
<p><strong> <div class="mathjax-scroll">\(\begin{array}{l} SO_2Cl_2 + 2H_2O \rightarrow H_2SO_4 + 2HCl\end{array} \)</div></strong></p>
<p><strong> 16 moles of NaOH is required for the complete neutralisation of the resultant acidic mixture. The number of moles of SO<sub>2</sub>Cl<sub>2</sub> used is</strong></p>
<p>(A) 16</p>
<p>(B) 8</p>
<p>(C) 4</p>
<p>(D) 2</p>
<p><strong>Answer (C)</strong></p>
<p><strong>Sol. </strong><div class="mathjax-scroll">\(\begin{array}{l} SO_2Cl_2 + 2H_2O \rightarrow H_2SO_4 + 2HCl\end{array} \)</div>
<p>Moles of NaOH required for complete neutralisation of resultant acidic mixture = 16 moles</p>
<p>And 1 mole of SO<sub>2</sub>Cl<sub>2</sub> produced 4 moles of H<sup>+</sup>.</p>
<div class="mathjax-scroll">\(\begin{array}{l}\therefore\text{Moles of}~SO_2Cl_2~ \text{used will be} = \frac{16}{4}= 4~ \text{moles}\end{array} \)</div>
"""

# Generate the question data dictionary
question_data = parse_html_to_question_data(html_input)

# # Example of adding the question to the database
# def add_sample_question(self, question_data):
#     """Add a parsed question to the database."""
#     # Pass the data to the database manager for processing
#     self.db_manager.add_question(question_data)

# Print to verify the parsed data
print(question_data)
