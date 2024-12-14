import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUserCircle, faFileAlt, faInfoCircle } from "@fortawesome/free-solid-svg-icons";

const Header = () => {
  return (
    <header className="bg-white py-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        {/* Logo Section */}
        <h1 className="text-3xl font-bold text-black p-4">AI Helper</h1>

        {/* Icons Section (Right Side) */}
        <div className="flex space-x-6 px-8">
          {/* Avatar Icon */}
          <FontAwesomeIcon icon={faUserCircle} size="2x" className="text-gray-700 cursor-pointer" />

          {/* Report Icon */}
          <FontAwesomeIcon icon={faFileAlt} size="2x" className="text-gray-700 cursor-pointer" />

          {/* About Icon */}
          <FontAwesomeIcon icon={faInfoCircle} size="2x" className="text-gray-700 cursor-pointer" />
        </div>
      </div>
    </header>
  );
};

export default Header;
