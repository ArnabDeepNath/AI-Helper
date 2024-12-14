import { useState } from "react";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [progress, setProgress] = useState(0); // Track the user's progress

  // Predefined responses for the bot
  const botResponses = [
    "Hello! How can I assist you today?",
    "I'm here to help! Let me know what you need.",
    "Sure, I'd love to assist you! Can you tell me more?",
    "That's interesting! I'm curious to know more.",
    "Feel free to ask me anything. I'm listening.",
    "You're doing great! Keep going.",
    "Have you checked the latest updates in the field?",
    "Let me know if you'd like any tips or advice!",
  ];

  const syllabus = [
    "1. Introduction to Programming",
    "2. Data Structures",
    "3. Algorithms",
    "4. Web Development",
    "5. Advanced Topics",
  ];

  const references = [
    "Book: 'Introduction to JavaScript' by John Doe",
    "Website: https://www.example.com",
    "Article: 'The Future of Web Development' at https://www.dev.com",
  ];

  // Generate a random bot response
  const getBotResponse = () => {
    const randomIndex = Math.floor(Math.random() * botResponses.length);
    return botResponses[randomIndex];
  };

  // Handle user message submission
  const handleSendMessage = (e) => {
    e.preventDefault();
    if (input.trim() === "") return;

    const userMessage = { text: input, sender: "user" };
    let botMessage = {};

    if (input.toLowerCase().includes("syllabus")) {
      botMessage = { text: `Here’s the syllabus: \n${syllabus.join("\n")}`, sender: "bot" };
    } else if (input.toLowerCase().includes("references")) {
      botMessage = { text: `Here are some references: \n${references.join("\n")}`, sender: "bot" };
    } else if (input.toLowerCase().includes("progress")) {
      botMessage = { text: `Your current progress is: ${progress}%`, sender: "bot" };
    } else {
      botMessage = { text: getBotResponse(), sender: "bot" };
    }

    setMessages((prevMessages) => [...prevMessages, userMessage, botMessage]);
    setInput("");
    updateProgress();
  };

  // Update the progress of the user interaction
  const updateProgress = () => {
    setProgress((prevProgress) => Math.min(prevProgress + 10, 100)); // Increase progress by 10%
  };

  return (
    <div className="flex  min-h-screen bg-gray-200">
      <div className="relative flex flex-col w-full max-w-8xl h-screen bg-white rounded-lg shadow-lg">

        {/* Chat Body */}
        <div className="flex-1 p-4 overflow-y-auto bg-grey-600">
          {/* Welcome Message */}
          {messages.length === 0 && (
            <div className="text-center text-gray-600 text-lg italic">
              "Welcome to the Chatbot! How can I assist you today?"
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`${
                  message.sender === "user"
                    ? "bg-[#63adf2] text-white text-2xl"
                    : "text-black font-bold text-2xl"
                } rounded-lg p-3 max-w-xs`}
              >
                {message.text}
              </div>
            </div>
          ))}
        </div>

        

        {/* Input Section */}
        <form onSubmit={handleSendMessage} className="p-4 border-t bg-gray-100 flex fixed bottom-0 left-0 w-full bg-[#304d6d]">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500  text-black input-blue-500 "
          />
          <button
            type="submit"
            className="ml-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default Chatbot;
