async function generatePrediction() {

    const predictionBox = document.getElementById("prediction");

    predictionBox.innerHTML = "Processing EMG Signals...";

    try {

        const response = await fetch("/predict", {
            method: "POST"
        });

        const data = await response.json();

        predictionBox.innerHTML = data.prediction;

    } catch (error) {

        predictionBox.innerHTML = "Prediction Failed";
    }
}