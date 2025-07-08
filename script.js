async function predictStock() {
    const stock = document.getElementById('stock').value;
    const open = parseFloat(document.getElementById('open').value);
    const high = parseFloat(document.getElementById('high').value);
    const low = parseFloat(document.getElementById('low').value);

    // Validate inputs
    if (!stock || isNaN(open) || isNaN(high) || isNaN(low)) {
        document.getElementById('predicted_close').innerText = 'Please fill all fields with valid numbers';
        document.getElementById('predicted_close').style.color = 'red';
        document.getElementById('recommendation').innerText = '';
        document.getElementById('model_used').innerText = '';
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stock, open, high, low })
        });
        const data = await response.json();

        if (data.error) {
            document.getElementById('predicted_close').innerText = `Error: ${data.error}`;
            document.getElementById('predicted_close').style.color = 'red';
            document.getElementById('recommendation').innerText = '';
            document.getElementById('model_used').innerText = '';
        } else {
            document.getElementById('predicted_close').innerText = `₹${data.predicted_close}`;
            document.getElementById('predicted_close').style.color = 'black';
            document.getElementById('recommendation').innerText = data.recommendation;
            document.getElementById('recommendation').style.color = 
                data.recommendation === 'Buy' ? 'green' : 
                data.recommendation === 'Sell' ? 'red' : 'black';
        }
    } catch (error) {
        document.getElementById('predicted_close').innerText = `Error: ${error.message}`;
        document.getElementById('predicted_close').style.color = 'red';
        document.getElementById('recommendation').innerText = '';
    }
}