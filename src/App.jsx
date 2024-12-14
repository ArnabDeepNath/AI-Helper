import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import AdminPage from "./pages/Admin";
import Quiz from "./pages/Quiz";
import Chatbot from "./pages/Chatbot";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {faHouse , faComment , faNewspaper} from '@fortawesome/free-solid-svg-icons';
import Header from "./components/Header";

function App() {
  return (
    
    <Router>
        <Header />

      <div className="flex">
        {/* Side Panel (Navigation) */}
        <div className="w-32 bg-white text-black p-4">
          <nav className="space-y-4">
            {/* FontAwesome Icon */}
            
            <a href="/dashboard" className="block px-4 py-2 hover:text-blue-400 rounded-md"><FontAwesomeIcon icon={faHouse} size="2x"  />Home</a>
            <a href="/quiz" className="block px-4 py-2 hover:text-blue-400 rounded-md"><FontAwesomeIcon icon={faComment} size="2x" />Quiz</a>
            <a href="/chatbot" className="block px-4 py-2 hover:text-blue-400 rounded-md"><FontAwesomeIcon icon={faNewspaper} size="2x" />Chatbot</a>
          </nav>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/quiz" element={<Quiz />} />
            <Route path="/chatbot" element={<Chatbot />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
