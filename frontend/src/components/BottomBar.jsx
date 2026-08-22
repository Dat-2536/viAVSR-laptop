import { useState } from 'react'
import {
  AudioLines,
  Upload,
  Square,
  Circle,
  RotateCcw,
  Download,
  ChevronDown
} from 'lucide-react';



const VideoInput = () => {
  return (
    <div className="border-2 border-dashed border-gray-400 p-4 text-center text-gray-500">
      Placeholder
    </div>
  );
};





const RecordButton = () => {
  const [isRecording, setIsRecording] = useState(false);

  const handleToggleRecord = () => {
    setIsRecording(!isRecording);
    // TODO: Implement start/stop recording logic
  };

  return (
    <button
      onClick={handleToggleRecord}
      className={`flex items-center gap-2 px-5 py-3 rounded-[5px] text-white font-medium transition-colors ${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
        }`}
    >
      {isRecording ? (
        <>
          <Square className="w-5 h-5 fill-current" />
          <span>Stop Recording</span>
        </>
      ) : (
        <>
          <Circle className="w-5 h-5 fill-current" />
          <span>Start Recording</span>
        </>
      )}
    </button>
  );
};

const ControlAndStatus = () => {
  const [selectedFormat, setSelectedFormat] = useState("");
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const exportResult = (format) => {
    console.log("Export format:", format);
  };

  const handleSelectFormat = (format) => {
    setSelectedFormat(format);
    setIsMenuOpen(false);
  };

  return (
    <div className="flex w-full items-center h-full p-[10px]">
      {/* Control Side (flex: 1) */}
      <div className="flex-1 flex justify-around items-center">
        {/* Upload Button */}
        <button className="flex items-center gap-1 bg-[#0096FF] text-white px-4 py-2 rounded-[5px]">
          <Upload className="w-5 h-5" />
          <span>Upload</span>
        </button>

        {/* Record Button */}
        <RecordButton />

        {/* Process Button */}
        <button className="flex items-center gap-1 bg-[#5CE65C] text-white px-4 py-2 rounded-[5px]">
          <span>Process</span>
        </button>

        {/* Reset Button */}
        <button className="flex items-center gap-1 bg-white text-black border border-gray-300 px-4 py-2 rounded-[5px]">
          <RotateCcw className="w-5 h-5" />
          <span>Reset</span>
        </button>
      </div>

      {/* Vertical Divider */}
      <div className="w-[1px] bg-gray-400 my-1 self-stretch" />

      {/* Status Side (flex: 2) */}
      <div className="flex-[2] flex justify-around items-center">
        <span>Status</span>

        {/* Export Split Button */}
        <div className="relative inline-flex items-center bg-white border border-gray-300 rounded-[10px]">
          <button
            onClick={() => exportResult(selectedFormat)}
            className="flex items-center gap-2 px-4 py-3 rounded-l-[10px] hover:bg-gray-50 font-semibold"
          >
            <Download className="w-[18px] h-[18px]" />
            <span>Export Result</span>
          </button>

          <div className="w-[1px] h-6 bg-gray-300" />

          {/* Menu Dropdown Toggle */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="p-2 hover:bg-gray-50 rounded-r-[10px]"
          >
            <ChevronDown className="w-5 h-5" />
          </button>

          {/* Dropdown Menu Items */}
          {isMenuOpen && (
            <div className="absolute right-0 top-full mt-1 w-32 bg-white border border-gray-200 rounded-md shadow-lg z-10 py-1">
              {['csv', 'xlsx', 'json', 'pdf'].map((fmt) => (
                <button
                  key={fmt}
                  onClick={() => handleSelectFormat(fmt)}
                  className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  {fmt === 'csv' ? 'CSV' : fmt === 'xlsx' ? 'Excel' : fmt.toUpperCase()}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};




export default ControlAndStatus;
