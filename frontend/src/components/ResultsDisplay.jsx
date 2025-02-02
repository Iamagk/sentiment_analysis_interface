const PLOT_BASE_URL = "" // Adjust this to match your actual base URL

const ResultsDisplay = ({ results }) => {
  if (!results) return null

  const { metrics, feature_importance, plot_path, predictions, actual_values, dates } = results

  return (
    <div className="space-y-8">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Model Metrics</h2>
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="border p-3 rounded">
              <div className="text-sm text-gray-500">{key}</div>
              <div className="text-lg font-medium">{value.toFixed(4)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Feature Importance</h2>
        <div className="space-y-2">
          {feature_importance.map(({ feature, importance }) => (
            <div key={feature} className="flex items-center">
              <div className="w-32 text-sm">{feature}</div>
              <div className="flex-1 bg-gray-200 rounded-full h-4">
                <div className="bg-blue-600 rounded-full h-4" style={{ width: `${importance * 100}%` }} />
              </div>
              <div className="w-16 text-right text-sm">{(importance * 100).toFixed(2)}%</div>
            </div>
          ))}
        </div>
      </div>

      {plot_path && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Prediction Plot</h2>
          <img src={plot_path || "/placeholder.svg"} alt="Prediction Plot" className="w-full" />
        </div>
      )}

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Predictions vs Actual Values</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead>
              <tr>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Predicted
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actual
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {dates.map((date, index) => (
                <tr key={date}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{date}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{predictions[index].toFixed(2)}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {actual_values[index].toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default ResultsDisplay
