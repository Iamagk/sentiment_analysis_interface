import { useState } from "react"
import PredictionForm from "./components/PredictionForm"
import ResultsDisplay from "./components/ResultsDisplay"
import { predictStockSentiment } from "./services/api"

function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (formData) => {
    try {
      setIsLoading(true)
      setError(null)
      const response = await predictStockSentiment(formData)
      setResults(response)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-center mb-8">Stock Sentiment Analysis</h1>

        {error && <div className="mb-4 p-4 text-red-700 bg-red-100 rounded-md">{error}</div>}

        <div className="flex flex-col lg:flex-row gap-8">
          <div className="lg:w-2/5">
            <PredictionForm onSubmit={handleSubmit} isLoading={isLoading} />
          </div>
          <div className="lg:w-3/5">{results && <ResultsDisplay results={results} />}</div>
        </div>
      </div>
    </div>
  )
}

export default App
