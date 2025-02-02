import { useState } from "react"

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    ticker: "",
    start_date: "",
    end_date: "",
    model: "",
    sentiment_files: [],
  })

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files)
    const filePaths = files.map((file) => file.name)
    setFormData((prev) => ({
      ...prev,
      sentiment_files: filePaths,
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    console.log("Submitting form data:", formData)
    onSubmit(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6 bg-white rounded-lg shadow-md p-6 sticky top-6">
      <div>
        <label className="block text-sm font-medium text-gray-700">Ticker Symbol</label>
        <select
          name="ticker"
          value={formData.ticker}
          onChange={handleInputChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          required
        >
          <option value="">Select a ticker</option>
          <option value="TSLA">TSLA</option>
          <option value="O">O</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Start Date</label>
        <input
          type="date"
          name="start_date"
          value={formData.start_date}
          onChange={handleInputChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">End Date</label>
        <input
          type="date"
          name="end_date"
          value={formData.end_date}
          onChange={handleInputChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Model</label>
        <select
          name="model"
          value={formData.model}
          onChange={handleInputChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
        >
          <option value="">Select a model</option>
          <option value="random_forest">Random Forest</option>
          <option value="xgboost">XGBoost</option>
          <option value="lightgbm">LGBoost</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Sentiment Files</label>
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          className="mt-1 block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
          required
        />
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
      >
        {isLoading ? "Processing..." : "Predict"}
      </button>
    </form>
  )
}

export default PredictionForm
