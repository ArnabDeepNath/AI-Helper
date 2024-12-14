import React from "react";
import image_01 from "../assets/image_01.jpg";
import image_02 from "../assets/image_02.jpg";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Navbar */}
      <nav className="bg-white shadow-md px-6 py-4 flex flex-col justify-between">
      {/* First Row: Image and Title */}
      <div className="flex justify-between items-center">
        <img src={image_01} className="w-40 h-auto" alt="Logo" />
        <h1 className="text-xl font-bold text-gray-800">Welcome Back , Arnab</h1>
      </div>

      {/* Second Row: Buttons */}
      <div className="flex justify-end mt-4 space-x-4">
        <button className="px-4 py-2 bg-blue-500 text-white rounded-md">
          Chat with Bot
        </button>
        <button className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md">
          Logout
        </button>
      </div>
    </nav>

      {/* Main Content */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Metrics */}
        <div className="col-span-1 lg:col-span-2 bg-white shadow-md rounded-lg p-6 flex-col items-center jusitfy-center">
          <img src={image_02} className="h-40 w-40" />
          <h2 className="text-lg font-bold mb-4 text-gray-800 ">Performance Overview</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <h3 className="text-xl font-semibold text-blue-600">85%</h3>
              <p className="text-gray-600">Overall Score</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg text-center">
              <h3 className="text-xl font-semibold text-green-600">95%</h3>
              <p className="text-gray-600">Physics</p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg text-center">
              <h3 className="text-xl font-semibold text-yellow-600">80%</h3>
              <p className="text-gray-600">Mathematics</p>
            </div>
          </div>
        </div>

        {/* Upcoming Tests */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Upcoming Tests</h2>
          <ul className="space-y-3">
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <h3 className="text-blue-600 font-semibold">Mock Test 1</h3>
              <p className="text-gray-500 text-sm">Physics - 12 Dec 2024, 10:00 AM</p>
            </li>
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <h3 className="text-blue-600 font-semibold">Mock Test 2</h3>
              <p className="text-gray-500 text-sm">Mathematics - 14 Dec 2024, 2:00 PM</p>
            </li>
          </ul>
        </div>

        {/* Notifications */}
        <div className="col-span-1 lg:col-span-3 bg-white shadow-md rounded-lg p-6">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Notifications</h2>
          <ul className="space-y-3">
            <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
              <div className="text-blue-600 mr-3">
                <i className="fas fa-info-circle"></i>
              </div>
              <p className="text-gray-600">
                Your Physics test result is available. Check now!
              </p>
            </li>
            <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
              <div className="text-green-600 mr-3">
                <i className="fas fa-check-circle"></i>
              </div>
              <p className="text-gray-600">
                Your account has been successfully updated.
              </p>
            </li>
          </ul>
        </div>

        {/* Test History */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Test History</h2>
          <ul className="space-y-3">
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <h3 className="text-blue-600 font-semibold">Physics Mock Test 1</h3>
              <p className="text-gray-500 text-sm">Score: 75% - 10 Nov 2024</p>
            </li>
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <h3 className="text-blue-600 font-semibold">Mathematics Mock Test 1</h3>
              <p className="text-gray-500 text-sm">Score: 80% - 15 Nov 2024</p>
            </li>
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <h3 className="text-blue-600 font-semibold">Chemistry Mock Test 1</h3>
              <p className="text-gray-500 text-sm">Score: 90% - 20 Nov 2024</p>
            </li>
          </ul>
        </div>

        {/* Study Resources */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Study Resources</h2>
          <ul className="space-y-3">
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <a href="#" className="text-blue-600 font-semibold hover:underline">
                Physics Study Material
              </a>
            </li>
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <a href="#" className="text-blue-600 font-semibold hover:underline">
                Mathematics Formulae
              </a>
            </li>
            <li className="bg-gray-50 p-3 rounded-lg shadow-sm">
              <a href="#" className="text-blue-600 font-semibold hover:underline">
                Chemistry Revision Notes
              </a>
            </li>
          </ul>
        </div>

        {/* Motivational Quote */}
        <div className="col-span-1 lg:col-span-3 bg-white shadow-md rounded-lg p-6">
          <h2 className="text-lg font-bold mb-4 text-gray-800">Motivational Quote</h2>
          <div className="text-center text-xl font-semibold italic text-gray-600">
            "Success is the sum of small efforts, repeated day in and day out."
            <br />
            - Robert Collier
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;