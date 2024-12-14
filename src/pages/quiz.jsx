import React, { useState } from "react";

const Quiz = () => {
  // Define subjects and chapters
  const subjects = [
    { name: "Physics", chapters: ["Mathematics in Physics", "Mechanics", "Thermodynamics"] },
    { name: "Chemistry", chapters: ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry"] },
    { name: "Biology", chapters: ["Botany", "Zoology", "Genetics"] },
  ];

  // Default data
  const [selectedSubject, setSelectedSubject] = useState("Physics");
  const [selectedChapter, setSelectedChapter] = useState("Mathematics in Physics");
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showSolution, setShowSolution] = useState(false);
  const [quizStarted, setQuizStarted] = useState(false);

  // Questions array (can be expanded as needed)
  const questions = [
    {
      question:
        "If the length of the pendulum in pendulum clock increases by 0.1%, then the error in time per day is:",
      options: ["43.2 s", "86.4 s", "8.64 s", "4.32 s"],
      answer: "43.2 s",
      explanation:
        "T=2π√l/g \nT ∝ l^(1/2)\nΔT/T = 1/2Δl/l\nΔT/24×3600 = 1/2 × 0.1/100\nΔT = 1/2 × 0.1/100 × 24 × 3600\nΔT = 43.2 s",
    },
  ];

  const handleAnswerClick = (option) => {
    setSelectedAnswer(option);
  };

  const handleViewSolution = () => {
    setShowSolution(true);
  };

  const handleNextQuestion = () => {
    setShowSolution(false);
    setSelectedAnswer(null);
    setCurrentQuestion(currentQuestion + 1);
  };

  const handleSubjectChange = (subject) => {
    setSelectedSubject(subject);
    setSelectedChapter(subjects.find(s => s.name === subject).chapters[0]); // Default to first chapter of selected subject
  };

  const handleChapterChange = (chapter) => {
    setSelectedChapter(chapter);
  };

  const handleStartQuiz = () => {
    setQuizStarted(true);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {!quizStarted ? (
        // Subject and Chapter Selection Page
        <div className="flex items-center justify-center min-h-screen bg-gray-200">
          <div className="bg-white p-8 rounded-lg shadow-md w-96">
            <h2 className="text-2xl font-semibold mb-4">Select Subject and Chapter</h2>

            {/* Subject Selection */}
            <div className="mb-4">
              <label className="block text-lg font-medium mb-2">Select Subject</label>
              <select
                className="w-full p-2 border rounded-lg"
                value={selectedSubject}
                onChange={(e) => handleSubjectChange(e.target.value)}
              >
                {subjects.map((subject, index) => (
                  <option key={index} value={subject.name}>
                    {subject.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Chapter Selection */}
            <div className="mb-4">
              <label className="block text-lg font-medium mb-2">Select Chapter</label>
              <select
                className="w-full p-2 border rounded-lg"
                value={selectedChapter}
                onChange={(e) => handleChapterChange(e.target.value)}
              >
                {subjects
                  .find((subject) => subject.name === selectedSubject)
                  .chapters.map((chapter, index) => (
                    <option key={index} value={chapter}>
                      {chapter}
                    </option>
                  ))}
              </select>
            </div>

            {/* Start Quiz Button */}
            <button
              className="bg-blue-600 text-white px-4 py-2 rounded-lg"
              onClick={handleStartQuiz}
            >
              Start Quiz
            </button>
          </div>
        </div>
      ) : (
        // Quiz Page
        <div className="flex min-h-screen bg-gray-100">
          {/* Top Navigation (Breadcrumb-like Structure) */}
          <div className="bg-white p-4 shadow-md w-full">
            <div className="flex items-center space-x-4">
              <span
                className="text-lg font-semibold cursor-pointer text-blue-600"
                onClick={() => handleSubjectChange("Physics")}
              >
                {selectedSubject}
              </span>
              <span className="text-lg">→</span>
              <span
                className="text-lg font-semibold cursor-pointer text-blue-600"
                onClick={() => handleChapterChange(selectedChapter)}
              >
                {selectedChapter}
              </span>
              
            </div>
            <div className="flex min-h-screen bg-gray-100">
            {/* Left Section - Question and Options */}
            <div className="w-3/4 p-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h2 className="text-2xl font-semibold mb-4">Question</h2>
                <p className="text-lg mb-6">{questions[currentQuestion].question}</p>
                <div className="space-y-4 mb-6">
                  {questions[currentQuestion].options.map((option, index) => (
                    <button
                      key={index}
                      onClick={() => handleAnswerClick(option)}
                      className={`w-full py-3 text-lg font-semibold text-white rounded-lg transition-colors duration-200 ${selectedAnswer === option ? "bg-blue-600" : "bg-gray-600 hover:bg-gray-500"}`}
                    >
                      {option}
                    </button>
                  ))}
                </div>
                <div className="flex justify-between mt-6">
                  <button className="bg-blue-600 text-white px-4 py-2 rounded-lg" onClick={handleViewSolution}>
                    View Solution
                  </button>
                  <button className="bg-gray-300 text-gray-700 px-4 py-2 rounded-lg" onClick={handleNextQuestion}>
                    Next
                  </button>
                </div>
              </div>
            </div>

            {/* Right Section - Solution Explanation */}
            <div className="w-1/4 bg-white p-6 rounded-lg shadow-md ml-4 flex flex-col">
              {showSolution && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Answer</h3>
                    <p className="text-lg text-blue-600 font-semibold">{questions[currentQuestion].answer}</p>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-medium">Solution Explanation</h4>
                    <pre className="text-sm whitespace-pre-wrap font-mono bg-gray-100 p-4 rounded-lg">
                      {questions[currentQuestion].explanation}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
          </div>

          
        </div>
      )}
    </div>
  );
};

export default Quiz;
