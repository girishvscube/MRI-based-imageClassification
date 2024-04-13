import React, { useState } from "react";

const Predict = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [imageFile, setImageFile] = useState();
  const [predictionResults, setPredictionResults] = useState();
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      // Assuming imageFile is a state variable that holds the selected image file
      const formData = new FormData();
      formData.append("image", imageFile); // Replace imageFile with your file variable

      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          body: formData,
          // Add headers if needed, e.g., content-type
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch");

        setLoading(false);
      }

      const data = await response.json();
      // Handle the response data as needed
      console.log("Prediction result:", data);
      setPredictionResults(data);

      setLoading(false);
      // You can update state or perform other actions with the received data
    } catch (error) {
      console.error("Error:", error);

      setLoading(false);
      // Handle errors, display error messages, etc.
    }
  };

  return (
    <div className="h-screen w-full   bgImage">
      <div className="text-black text-4xl text-center pt-10 capitalize font-extrabold">
        MRI BASED BRAIN TUMOR DETECTION SYSTEM
      </div>
      <div className="grid grid-cols-2 gap-6">
        <div className="text-blue-600">
          <div className="w-full flex flex-col items-center justify-center h-screen my-auto p-4  rounded-lg text-center">
            <input type="file" onChange={handleFileChange} />
            {imagePreview && (
              <img
                src={imagePreview}
                alt="Preview"
                className="mt-4 mx-auto max-w-full max-h-[40vh]"
              />
            )}

            <button
              className="mt-4 w-32 py-2 bg-black rounded-md text-white"
              onClick={handlePredict}
            >
              {loading ? "Loading...." : "Predict"}
            </button>
          </div>
        </div>

        <div className="text-black w-full mt-[20vh] ">
          <p className="text-center text-3xl pt-6">Prediction Results</p>

          {loading ? (
            <p className="text-3xl text-center">Loading</p>
          ) : (
            <>
              <div className="w-[90%] flex flex-col gap-4 h-[70vh]  p-4 m-auto mt-10 rounded-md  text-blue-600 shadow-2xl">
                <div className="flex gap-2">
                  <p className="font-[600]">Results:</p>
                  <p className="font-[800]">{predictionResults?.name}</p>
                </div>
               
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Predict;