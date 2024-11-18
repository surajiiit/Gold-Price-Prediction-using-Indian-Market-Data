document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const errorMessage = document.getElementById('error-message');
    const predictionResult = document.getElementById('prediction-result');
    const predictionValue = document.getElementById('prediction-value');
    const predictionTimestamp = document.getElementById('prediction-timestamp');

    // Function to update market data
    async function updateMarketData() {
        try {
            const response = await fetch('/api/market-data');
            const data = await response.json();
            // Update market cards with new data
            // Implementation depends on your market data structure
        } catch (error) {
            console.error('Error updating market data:', error);
        }
    }

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Hide previous results/errors
        errorMessage.classList.add('hidden');
        predictionResult.classList.add('hidden');

        // Get form data
        const formData = new FormData(form);
        const data = {
            sensex: formData.get('sensex'),
            nifty50: formData.get('nifty50'),
            crude_oil_usd: formData.get('crude_oil_usd'),
            eur_usd: formData.get('eur_usd')
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            // Ensure the prediction value is parsed as float and formatted in INR
            const prediction = parseFloat(result.prediction); // Parse to float
            const formattedValue = `₹${prediction.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

            // Format timestamp as dd/mm/yyyy
            const formattedDate = new Date().toLocaleDateString('en-IN');  // dd/mm/yyyy format

            // Update the prediction display
            predictionValue.textContent = formattedValue;
            predictionTimestamp.textContent = `Date: ${formattedDate}`;
            predictionResult.classList.remove('hidden');
        } catch (error) {
            errorMessage.querySelector('p').textContent = error.message || 'Failed to predict gold price';
            errorMessage.classList.remove('hidden');
        }
    });

    // Update market data periodically
    setInterval(updateMarketData, 60000); // Update every minute
});
