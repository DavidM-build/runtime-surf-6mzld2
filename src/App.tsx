import React, { useRef, useEffect, useState } from "react";
import * as faceapi from "face-api.js";
import html2canvas from "html2canvas";
import numeric from "numeric";
import "./styles.css";

function meanLandmarks(landmarks) {
  const sum = landmarks.reduce(
    (acc, pt) => ({ x: acc.x + pt.x, y: acc.y + pt.y }),
    { x: 0, y: 0 }
  );
  return {
    x: sum.x / landmarks.length,
    y: sum.y / landmarks.length,
  };
}

function normalizeLandmarks(landmarks) {
  const mean = meanLandmarks(landmarks);
  let maxDist = 0;
  landmarks.forEach((pt) => {
    const dist = Math.hypot(pt.x - mean.x, pt.y - mean.y);
    maxDist = Math.max(maxDist, dist);
  });

  return landmarks.map((pt) => ({
    x: (pt.x - mean.x) / maxDist,
    y: (pt.y - mean.y) / maxDist,
  }));
}

function procrustesAnalysis(landmarks1, landmarks2) {
  const X = landmarks1.map((pt) => [pt.x, pt.y]);
  const Y = landmarks2.map((pt) => [pt.x, pt.y]);

  const meanX = meanLandmarks(landmarks1);
  const meanY = meanLandmarks(landmarks2);

  const centeredX = X.map((pt) => [pt[0] - meanX.x, pt[1] - meanX.y]);
  const centeredY = Y.map((pt) => [pt[0] - meanY.x, pt[1] - meanY.y]);

  const covMatrix = numeric.dot(numeric.transpose(centeredX), centeredY);
  const svd = numeric.svd(covMatrix);
  const R = numeric.dot(svd.V, numeric.transpose(svd.U));

  const rotatedX = centeredX.map((pt) => numeric.dot(R, pt));

  const weights = Array(68).fill(1);
  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47].forEach(
    (i) => (weights[i] = 2)
  ); // Eyes (reduced from 3 to 2)
  [27, 28, 29, 30].forEach((i) => (weights[i] = 1.75)); // Nose bridge (reduced from 2.5 to 1.75)
  [31, 32, 33, 34, 35].forEach((i) => (weights[i] = 1.5)); // Nose (reduced from 2 to 1.5)
  [48, 49, 50, 51, 57, 58, 59, 60].forEach((i) => (weights[i] = 1.5)); // Mouth (reduced from 2 to 1.5)

  let totalDistance = 0;
  let totalWeight = 0;
  const smoothingFactor = 0.0;

  for (let i = 0; i < rotatedX.length; i++) {
    const dx = rotatedX[i][0] - centeredY[i][0];
    const dy = rotatedX[i][1] - centeredY[i][1];
    totalDistance += weights[i] * Math.hypot(dx, dy);
    totalWeight += weights[i];
  }
  //return totalDistance / totalWeight;
  return totalDistance / totalWeight + smoothingFactor;
}

function App() {
  const imageRef1 = useRef();
  const imageRef2 = useRef();
  const canvasRef1 = useRef();
  const canvasRef2 = useRef();
  const captureRef = useRef();
  const [initialized, setInitialized] = useState(false);
  const [selectedImage1, setSelectedImage1] = useState(null);
  const [selectedImage2, setSelectedImage2] = useState(null);
  const [detection1, setDetection1] = useState(null);
  const [detection2, setDetection2] = useState(null);
  const [similarity, setSimilarity] = useState(null);
  const [threshold, setThreshold] = useState(0.5);

  const loadModels = async () => {
    const MODEL_URL =
      "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights";
    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);
      setInitialized(true);
    } catch (error) {
      console.error("Error loading models:", error);
      alert("Failed to load models. Please refresh the page and try again.");
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleImageChange = (event, setImage) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setImage(file);
      setSimilarity(null);
      setDetection1(null);
      setDetection2(null);
    }
  };

  const handleImageLoad = async (imageRef, canvasRef) => {
    if (imageRef.current) {
      const detections = await faceapi
        .detectAllFaces(imageRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (detections.length === 0) {
        alert("No faces detected. Please select an image with a clear face.");
        return null;
      }

      const displaySize = {
        width: imageRef.current.width,
        height: imageRef.current.height,
      };

      if (canvasRef.current) {
        faceapi.matchDimensions(canvasRef.current, displaySize);
        const resizedDetections = faceapi.resizeResults(
          detections,
          displaySize
        );

        const ctx = canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, displaySize.width, displaySize.height);

        faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);
      }

      return detections[0];
    }
    return null;
  };

  const compareFaces = () => {
    if (detection1 && detection2) {
      // Get confidence scores
      const confidence1 = detection1.detection.score;
      const confidence2 = detection2.detection.score;

      // Compute Euclidean distance between face descriptors
      const descriptorDistance = faceapi.euclideanDistance(
        detection1.descriptor,
        detection2.descriptor
      );

      // Calculate descriptor similarity score
      const descriptorSimilarity = Math.max(0, (1 - descriptorDistance) * 100);

      // Extract and normalize landmark positions
      const landmarks1 = detection1.landmarks.positions;
      const landmarks2 = detection2.landmarks.positions;
      const normalizedLandmarks1 = normalizeLandmarks(landmarks1);
      const normalizedLandmarks2 = normalizeLandmarks(landmarks2);

      // Compute landmark distance
      const landmarkDistance = procrustesAnalysis(
        normalizedLandmarks1,
        normalizedLandmarks2
      );

      // Calculate landmark similarity using exponential decay
      const landmarksSimilarity = Math.max(
        0,
        100 * Math.exp(-5 * landmarkDistance)
      );

      // Weight the two similarity scores
      const DESCRIPTOR_WEIGHT = 0.7;
      const LANDMARK_WEIGHT = 0.3;

      const overallSimilarity = (
        DESCRIPTOR_WEIGHT * descriptorSimilarity +
        LANDMARK_WEIGHT * landmarksSimilarity
      ).toFixed(2);

      // Compare with threshold
      const isMatch = overallSimilarity >= threshold * 100;

      // Check for possible doppelgänger
      const isPossibleDoppelganger =
        landmarksSimilarity >= 50 && descriptorSimilarity <= 49;

      setSimilarity({
        score: overallSimilarity,
        descriptorScore: descriptorSimilarity.toFixed(2),
        landmarkScore: landmarksSimilarity.toFixed(2),
        margin: ((confidence1 + confidence2) * 50).toFixed(2),
        isMatch: isMatch,
        threshold: (threshold * 100).toFixed(0),
        confidence1: (confidence1 * 100).toFixed(2),
        confidence2: (confidence2 * 100).toFixed(2),
        landmarks1: landmarks1.length,
        landmarks2: landmarks2.length,
        isPossibleDoppelganger: isPossibleDoppelganger,
      });
    } else {
      alert("Both images must have a detectable face.");
    }
  };

  const handleScreenshot = async () => {
    if (captureRef.current) {
      try {
        const canvas = await html2canvas(captureRef.current, {
          useCORS: true,
          allowTaint: true,
        });
        const dataUrl = canvas.toDataURL("image/png");
        const link = document.createElement("a");
        link.href = dataUrl;
        link.download = "comparison_result.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (error) {
        console.error("Error capturing screenshot:", error);
      }
    } else {
      alert("No result to capture.");
    }
  };

  return (
    <div className="App">
      <h1>Face Detection and Comparison</h1>
      {!initialized && <p>Loading models...</p>}
      {initialized && (
        <div>
          <div ref={captureRef}>
            <div className="image-container">
              <div className="image-box">
                <h3>First Image</h3>
                <div className="upload-button">
                  <label htmlFor="file-input-1">Choose File</label>
                  <input
                    id="file-input-1"
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleImageChange(e, setSelectedImage1)}
                  />
                </div>
                {selectedImage1 && (
                  <div className="image-wrapper">
                    <img
                      ref={imageRef1}
                      src={URL.createObjectURL(selectedImage1)}
                      alt="First"
                      onLoad={async () => {
                        const detection = await handleImageLoad(
                          imageRef1,
                          canvasRef1
                        );
                        setDetection1(detection);
                      }}
                    />
                    <canvas ref={canvasRef1} />
                  </div>
                )}
              </div>

              <div className="image-box">
                <h3>Second Image</h3>
                <div className="upload-button">
                  <label htmlFor="file-input-2">Choose File</label>
                  <input
                    id="file-input-2"
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleImageChange(e, setSelectedImage2)}
                  />
                </div>
                {selectedImage2 && (
                  <div className="image-wrapper">
                    <img
                      ref={imageRef2}
                      src={URL.createObjectURL(selectedImage2)}
                      alt="Second"
                      onLoad={async () => {
                        const detection = await handleImageLoad(
                          imageRef2,
                          canvasRef2
                        );
                        setDetection2(detection);
                      }}
                    />
                    <canvas ref={canvasRef2} />
                  </div>
                )}
              </div>
            </div>

            {similarity && (
              <div className="result">
                <h3>Overall Similarity Score: {similarity.score}%</h3>
                <p>Descriptor Similarity: {similarity.descriptorScore}%</p>
                <p>Landmark Similarity: {similarity.landmarkScore}%</p>
                <p className={similarity.isMatch ? "match" : "no-match"}>
                  {similarity.isMatch ? "Faces match!" : "Faces do not match"}
                </p>
                <p>Current threshold: {similarity.threshold}%</p>

                {similarity.isPossibleDoppelganger && (
                  <p className="doppelganger-warning">
                    High landmark similarity but low descriptor similarity
                    detected.
                    <br />
                    This may indicate a doppelgänger!
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="threshold-container">
            <label>
              Match Threshold: {threshold * 100}%
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={threshold * 100}
                onChange={(e) => setThreshold(parseFloat(e.target.value) / 100)}
                style={{ width: "200px", margin: "0 10px" }}
              />
            </label>
          </div>

          <button
            onClick={compareFaces}
            disabled={!detection1 || !detection2}
            className="compare-button"
          >
            Compare Faces
          </button>

          {similarity && (
            <button onClick={handleScreenshot} className="screenshot-button">
              Download Screenshot
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
