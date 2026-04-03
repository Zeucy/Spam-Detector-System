document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('detect-form');
    const messageInput = document.getElementById('message-input');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.getElementById('spinner');
    
    const resultContainer = document.getElementById('result-container');
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const resultDesc = document.getElementById('result-desc');
    const confidenceBadge = document.getElementById('confidence-badge');
    const progressBarFill = document.getElementById('progress-bar-fill');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        // Set Loading State
        setLoading(true);
        resultContainer.classList.add('hidden'); // Hide previous result

        try {
            // Make API Call
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to process request");
            }

            // Artificial delay to show off the cool spinner animation and build tension (300ms)
            setTimeout(() => {
                showResult(data);
                setLoading(false);
            }, 300);

        } catch (error) {
            console.error("Error:", error);
            alert("Error: " + error.message);
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'block';
        } else {
            submitBtn.disabled = false;
            btnText.style.display = 'block';
            spinner.style.display = 'none';
        }
    }

    function showResult(data) {
        // Reset classes
        resultCard.classList.remove('spam', 'ham');
        progressBarFill.style.width = '0%';
        
        let probability = data.probability;
        if (!data.is_spam) {
            // If it's ham, confidence is 1 - spam probability
            probability = 1 - probability;
        }

        const percentage = (probability * 100).toFixed(1) + '%';
        
        if (data.is_spam) {
            resultCard.classList.add('spam');
            resultTitle.textContent = 'Spam Detected';
            resultDesc.textContent = 'Warning! This message shows signs of being malicious or unsolicited spam. Avoid clicking links or replying.';
        } else {
            resultCard.classList.add('ham');
            resultTitle.textContent = 'Safe Message';
            resultDesc.textContent = 'This message appears to be safe and legitimate (Ham).';
        }

        confidenceBadge.textContent = 'Confidence: ' + percentage;

        // Display the block
        resultContainer.style.display = 'block';
        
        // Small delay to allow display block to apply before animating opacity/transform
        setTimeout(() => {
            resultContainer.classList.remove('hidden');
            resultContainer.style.opacity = '1';
            resultContainer.style.transform = 'translateY(0)';
            
            // Animate progress bar
            progressBarFill.style.width = percentage;
        }, 50);
    }
});
