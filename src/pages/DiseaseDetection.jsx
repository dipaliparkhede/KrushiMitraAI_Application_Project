import React, { useEffect, useRef, useState } from "react";
import "./css/DiseaseDetection.css";

const API_URL = "http://127.0.0.1:8000";

// ============================================================
// DISEASE / PEST SUGGESTIONS
// ============================================================

const diseaseSuggestions = {
  "Cashew anthracnose": {
    type: "Disease",
    description:
      "The image is classified as cashew anthracnose.",
    treatment:
      "Remove severely affected leaves and plant parts, maintain good field sanitation, and follow locally recommended fungicide practices when necessary.",
  },

  "Cashew gumosis": {
    type: "Disease",
    description:
      "The image is classified as cashew gumosis.",
    treatment:
      "Remove damaged portions where practical, maintain good drainage and field sanitation, and consult local agricultural guidance for suitable treatment.",
  },

  "Cashew healthy": {
    type: "Healthy",
    description:
      "The image is classified as a healthy cashew plant.",
    treatment:
      "Continue regular monitoring, balanced irrigation and nutrition, and inspect new leaves and shoots for early symptoms.",
  },

  "Cashew leaf miner": {
    type: "Pest",
    description:
      "The image is classified as cashew leaf miner damage.",
    treatment:
      "Monitor new leaves regularly, remove severely affected leaves where practical, and use approved pest-control measures only when necessary.",
  },

  "Cashew red rust": {
    type: "Disease",
    description:
      "The image is classified as cashew red rust.",
    treatment:
      "Maintain field sanitation, remove severely affected plant material where practical, and follow locally recommended disease-management practices.",
  },

  "Cassava bacterial blight": {
    type: "Disease",
    description:
      "The image is classified as cassava bacterial blight.",
    treatment:
      "Use healthy planting material, remove severely affected plant material where recommended, and avoid moving infected material between fields.",
  },

  "Cassava brown spot": {
    type: "Disease",
    description:
      "The image is classified as cassava brown spot.",
    treatment:
      "Remove heavily affected leaves where practical, maintain field sanitation, improve airflow around plants, and follow locally recommended disease-control practices.",
  },

  "Cassava green mite": {
    type: "Pest",
    description:
      "The image is classified as cassava green mite damage.",
    treatment:
      "Inspect young leaves regularly, reduce heavily affected foliage where practical, and use approved mite-control measures when necessary.",
  },

  "Cassava healthy": {
    type: "Healthy",
    description:
      "The image is classified as healthy cassava.",
    treatment:
      "Continue regular crop monitoring and maintain proper irrigation, nutrition and field sanitation.",
  },

  "Cassava mosaic": {
    type: "Disease",
    description:
      "The image is classified as cassava mosaic.",
    treatment:
      "Use healthy planting material, monitor nearby plants, and follow locally recommended practices for controlling disease spread and insect vectors.",
  },

  "Maize fall armyworm": {
    type: "Pest",
    description:
      "The image is classified as maize fall armyworm damage.",
    treatment:
      "Inspect the whorl and young leaves regularly, remove visible larvae where practical, and use approved insect-control measures according to local recommendations.",
  },

  "Maize grasshoper": {
    type: "Pest",
    description:
      "The image is classified as maize grasshopper damage.",
    treatment:
      "Monitor the crop regularly, control weeds around the field, and use approved pest-management measures when infestation becomes significant.",
  },

  "Maize healthy": {
    type: "Healthy",
    description:
      "The image is classified as healthy maize.",
    treatment:
      "Continue regular monitoring and maintain balanced irrigation, nutrition and crop management.",
  },

  "Maize leaf beetle": {
    type: "Pest",
    description:
      "The image is classified as maize leaf beetle damage.",
    treatment:
      "Inspect leaves regularly, monitor pest levels, maintain field sanitation, and use approved insect-control measures when necessary.",
  },

  "Maize leaf blight": {
    type: "Disease",
    description:
      "The image is classified as maize leaf blight.",
    treatment:
      "Remove or properly manage infected crop residues, improve air circulation, avoid unnecessary leaf wetness, and follow local disease-management recommendations.",
  },

  "Maize leaf spot": {
    type: "Disease",
    description:
      "The image is classified as maize leaf spot.",
    treatment:
      "Monitor the spread of lesions, maintain field sanitation, improve crop airflow, and follow locally recommended disease-control practices when needed.",
  },

  "Maize streak virus": {
    type: "Disease",
    description:
      "The image is classified as maize streak virus.",
    treatment:
      "Use healthy planting material, monitor insect vectors, remove severely affected plants where locally recommended, and consider resistant varieties where available.",
  },

  "Tomato healthy": {
    type: "Healthy",
    description:
      "The image is classified as a healthy tomato plant.",
    treatment:
      "Continue regular monitoring, maintain proper irrigation and nutrition, and inspect leaves regularly for early pest or disease symptoms.",
  },

  "Tomato leaf blight": {
    type: "Disease",
    description:
      "The image is classified as tomato leaf blight.",
    treatment:
      "Remove severely affected leaves, improve airflow, avoid prolonged leaf wetness, and follow locally recommended fungicide practices when required.",
  },

  "Tomato leaf curl": {
    type: "Disease",
    description:
      "The image is classified as tomato leaf curl.",
    treatment:
      "Monitor for insect vectors such as whiteflies, remove severely affected plants where locally recommended, and use healthy planting material.",
  },

  "Tomato septoria leaf spot": {
    type: "Disease",
    description:
      "The image is classified as tomato septoria leaf spot.",
    treatment:
      "Remove affected leaves where practical, maintain good field sanitation, keep foliage dry where possible, and follow locally recommended disease-control practices.",
  },

  "Tomato verticulium wilt": {
    type: "Disease",
    description:
      "The image is classified as tomato verticillium wilt.",
    treatment:
      "Remove severely affected plants where recommended, avoid moving contaminated soil, maintain field sanitation, and use resistant varieties where available.",
  },
};

// ============================================================
// FALLBACK SUGGESTION
// ============================================================

const fallbackSuggestion = {
  type: "Unknown",
  description:
    "The model returned a prediction, but detailed information is not available for this class.",
  treatment:
    "Retake the image in good lighting with the affected leaf clearly visible. If symptoms continue, consult a local agricultural expert.",
};

// ============================================================
// COMPONENT
// ============================================================

function DiseaseDetection() {
  const fileInputRef = useRef(null);

  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // ----------------------------------------------------------
  // Cleanup preview URL when component unmounts
  // ----------------------------------------------------------

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  // ----------------------------------------------------------
  // HANDLE FILE
  // ----------------------------------------------------------

  const handleFile = (file) => {
    if (!file) return;

    setError("");
    setResult(null);

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Image size should be less than 10 MB.");
      return;
    }

    // Revoke previous preview URL
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    const imageUrl = URL.createObjectURL(file);

    setSelectedImage(file);
    setPreview(imageUrl);
  };

  // ----------------------------------------------------------
  // FILE INPUT
  // ----------------------------------------------------------

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];

    if (file) {
      handleFile(file);
    }
  };

  // ----------------------------------------------------------
  // DRAG & DROP
  // ----------------------------------------------------------

  const handleDrop = (event) => {
    event.preventDefault();

    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      handleFile(file);
    }
  };

  // ----------------------------------------------------------
  // REMOVE IMAGE
  // ----------------------------------------------------------

  const removeImage = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setSelectedImage(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // ----------------------------------------------------------
  // DETECT DISEASE
  // ----------------------------------------------------------

  const detectDisease = async () => {
    if (!selectedImage) {
      setError("Please upload a crop leaf image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      // Create multipart form data
      const formData = new FormData();

      formData.append("file", selectedImage);

      console.log("Sending image to backend:", selectedImage.name);

      // Call FastAPI
      const response = await fetch(
        `${API_URL}/disease/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      // Try to read JSON response
      const data = await response.json();

      console.log("Backend response:", data);

      if (!response.ok) {
        throw new Error(
          data.detail || "Disease prediction failed."
        );
      }

      // ------------------------------------------------------
      // Backend gives:
      //
      // data.prediction
      // data.confidence
      // data.top_predictions
      // ------------------------------------------------------

      const prediction = data.prediction;

      const suggestion =
        diseaseSuggestions[prediction] ||
        fallbackSuggestion;

      // Convert backend response into format
      // already used by your existing UI
      setResult({
        disease: prediction,
        confidence: data.confidence,
        treatment: suggestion.treatment,
        type: suggestion.type,
        description: suggestion.description,
        topPredictions: data.top_predictions || [],
      });
    } catch (err) {
      console.error("Disease prediction error:", err);

      setError(
        err.message ||
          "Unable to connect to the disease detection server."
      );
    } finally {
      setLoading(false);
    }
  };

  // ==========================================================
  // JSX
  // ==========================================================

  return (
    <main className="disease-page">

      {/* ================= HEADER ================= */}

      <section className="disease-hero">

        <div className="disease-hero-content">

          <div className="disease-badge">
            <span>🌿</span>
            AI POWERED CROP HEALTH
          </div>

          <h1>
            Detect Crop Diseases
            <span> Early with AI</span>
          </h1>

          <p>
            Upload a clear photo of your crop leaf and let
            KrishiMitra AI identify possible diseases and
            provide helpful guidance.
          </p>

        </div>

      </section>


      {/* ================= MAIN ================= */}

      <section className="disease-content">

        <div className="disease-card">

          {/* ================= CARD HEADER ================= */}

          <div className="card-header">

            <div className="header-icon">
              🔬
            </div>

            <div>
              <h2>
                Leaf Disease Detection
              </h2>

              <p>
                Upload a clear image of the affected crop leaf.
              </p>
            </div>

          </div>


          {/* ================= UPLOAD ================= */}

          {!preview ? (

            <div
              className={`upload-area ${
                isDragging ? "dragging" : ""
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => {
                setIsDragging(false);
              }}
              onDrop={handleDrop}
              onClick={() =>
                fileInputRef.current?.click()
              }
            >

              <div className="upload-icon">
                📷
              </div>

              <h3>
                Upload your leaf image
              </h3>

              <p>
                Drag & drop your image here or
                <span> browse files</span>
              </p>

              <small>
                JPG, JPEG or PNG • Maximum 10 MB
              </small>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp"
                onChange={handleImageChange}
                hidden
              />

            </div>

          ) : (

            /* ================= IMAGE PREVIEW ================= */

            <div className="preview-section">

              <div className="preview-header">

                <div>
                  <h3>
                    Selected Leaf Image
                  </h3>

                  <p>
                    {selectedImage?.name}
                  </p>
                </div>

                <button
                  className="remove-button"
                  onClick={removeImage}
                  type="button"
                  disabled={loading}
                >
                  ✕ Remove
                </button>

              </div>

              <div className="preview-image-wrapper">

                <img
                  src={preview}
                  alt="Selected crop leaf"
                  className="preview-image"
                />

              </div>

            </div>

          )}


          {/* ================= ERROR ================= */}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}


          {/* ================= DETECT BUTTON ================= */}

          {preview && !loading && !result && (

            <button
              className="detect-button"
              onClick={detectDisease}
              type="button"
            >
              <span>🔍</span>
              Detect Disease
            </button>

          )}


          {/* ================= LOADING ================= */}

          {loading && (

            <div className="scanning-box">

              <div className="scanner">

                <div className="scanner-line"></div>

                <img
                  src={preview}
                  alt="Scanning leaf"
                />

              </div>

              <div className="scanning-text">

                <h3>
                  🔄 Analyzing your crop...
                </h3>

                <p>
                  AI is examining the leaf image for
                  possible diseases.
                </p>

                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>

              </div>

            </div>

          )}


          {/* ================= RESULT ================= */}

          {result && !loading && (

            <div className="result-section">

              <div className="result-header">

                <div className="success-icon">
                  ✓
                </div>

                <div>

                  <span>
                    AI ANALYSIS COMPLETE
                  </span>

                  <h2>
                    Detection Result
                  </h2>

                </div>

              </div>


              {/* ================= DISEASE ================= */}

              <div className="disease-result">

                <div>

                  <p className="result-label">
                    DETECTED {result.type?.toUpperCase()}
                  </p>

                  <h2>
                    {result.disease}
                  </h2>

                </div>

                <div className="confidence">

                  <p>
                    Confidence
                  </p>

                  <strong>
                    {result.confidence}%
                  </strong>

                </div>

              </div>


              {/* ================= CONFIDENCE ================= */}

              <div className="confidence-bar">

                <div
                  className="confidence-fill"
                  style={{
                    width: `${Math.min(
                      result.confidence,
                      100
                    )}%`,
                  }}
                ></div>

              </div>


              {/* ================= DESCRIPTION ================= */}

              <div className="treatment-card">

                <div className="treatment-icon">
                  🌱
                </div>

                <div>

                  <h3>
                    About this Result
                  </h3>

                  <p>
                    {result.description}
                  </p>

                </div>

              </div>


              {/* ================= RECOMMENDED ACTION ================= */}

              <div className="treatment-card">

                <div className="treatment-icon">
                  💡
                </div>

                <div>

                  <h3>
                    Recommended Action
                  </h3>

                  <p>
                    {result.treatment}
                  </p>

                </div>

              </div>


              {/* ================= OTHER PREDICTIONS ================= */}

              {result.topPredictions?.length > 1 && (

                <div className="treatment-card">

                  <div className="treatment-icon">
                    📊
                  </div>

                  <div>

                    <h3>
                      Other Possible Results
                    </h3>

                    {result.topPredictions
                      .slice(1)
                      .map((prediction) => (

                        <p
                          key={prediction.class_index}
                          style={{
                            marginBottom: "6px",
                          }}
                        >
                          {prediction.display_name}
                          {" — "}
                          {prediction.confidence}%
                        </p>

                      ))}

                  </div>

                </div>

              )}


              {/* ================= NEW SCAN ================= */}

              <button
                className="new-scan-button"
                onClick={removeImage}
                type="button"
              >
                🔄 Scan Another Leaf
              </button>

            </div>

          )}

        </div>


        {/* ================= INFORMATION ================= */}

        <div className="disease-info">

          <div className="info-item">

            <span>📸</span>

            <div>
              <h4>
                Clear Image
              </h4>

              <p>
                Take a clear photo of the affected leaf.
              </p>
            </div>

          </div>


          <div className="info-item">

            <span>☀️</span>

            <div>
              <h4>
                Good Lighting
              </h4>

              <p>
                Avoid dark or blurry photos.
              </p>
            </div>

          </div>


          <div className="info-item">

            <span>🍃</span>

            <div>
              <h4>
                Focus on Leaf
              </h4>

              <p>
                Keep the affected leaf clearly visible.
              </p>
            </div>

          </div>

        </div>

      </section>

    </main>
  );
}

export default DiseaseDetection;