const API_URL = "http://127.0.0.1:8000";

export const predictStockSentiment = async (formData) => {
  try {
    console.log("Sending data to API:", formData) // Add this line for debugging
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || "An error occurred")
    }

    return await response.json();
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
};
