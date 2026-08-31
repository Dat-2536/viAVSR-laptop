import React from 'react';
import ReactDOM from 'react-dom/client';
import Webcam from 'react-webcam';
//
// const webcam = () => (
//   <Webcam />
// );
//

const VideoInput = () => {
  return (
    <div className='m-10 border border-black'>
      <Webcam
        audio={true}
      />
    </div>
  )
}
export default VideoInput;
