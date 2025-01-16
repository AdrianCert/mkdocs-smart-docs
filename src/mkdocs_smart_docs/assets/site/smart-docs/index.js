async function askQuestion(modelName, question) {
    const url = 'http://127.0.0.1:8000/answer';

    const requestData = {
        model_name: modelName,
        question: question
    };

    try {
        const response = await fetch(url, {
            method: 'POST', // Use POST method
            headers: {
                'Content-Type': 'application/json', // Specify that we're sending JSON
            },
            body: JSON.stringify(requestData) // Convert the request data to JSON format
        });

        // Check if the response is OK (status code 200-299)
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the JSON response
        const data = await response.json();
        console.log('Answer:', data.answer); // Output the answer from the API

        return data.answer; // Return the answer to the caller
    } catch (error) {
        console.error('Error:', error);
    }
}


document.addEventListener('DOMContentLoaded', function () {
    // Create the button
    var btn = document.createElement('button');
    btn.innerText = 'Chat with AI';
    btn.classList.add('chat-button'); // Use CSS class for styles

    // Append the button to the body
    document.body.appendChild(btn);

    // Create the modal structure
    var modal = document.createElement('div');
    modal.classList.add('chat-modal'); // Use CSS class for styles
    modal.setAttribute('data-annotation', 'smart-doc'); // Add data annotation

    // Modal content container
    var modalContent = document.createElement('div');
    modalContent.classList.add('chat-modal-content'); // Use CSS class for styles
    modalContent.style.backgroundColor = getComputedStyle(document.body).backgroundColor; // Inherit background color

    // Chat area
    var chatArea = document.createElement('div');
    chatArea.classList.add('chat-area'); // Use CSS class for styles

    // Input area
    var inputContainer = document.createElement('div');
    inputContainer.classList.add('chat-input-container'); // Use CSS class for styles

    var inputField = document.createElement('input');
    inputField.type = 'text';
    inputField.placeholder = 'Type your question...';
    inputField.classList.add('chat-input'); // Use CSS class for styles

    var sendButton = document.createElement('button');
    sendButton.innerText = 'Send';
    sendButton.classList.add('chat-send-button'); // Use CSS class for styles

    // Append input field and button to input container
    inputContainer.appendChild(inputField);
    inputContainer.appendChild(sendButton);

    // Close button
    var closeButton = document.createElement('button');
    closeButton.innerText = 'Close';
    closeButton.classList.add('chat-close-button'); // Use CSS class for styles
    closeButton.addEventListener('click', function () {
        modal.classList.remove('active'); // Use CSS class for styles
    });

    // Append everything to modal content
    modalContent.appendChild(chatArea);
    modalContent.appendChild(inputContainer);
    modalContent.appendChild(closeButton);

    // Append modal content to modal
    modal.appendChild(modalContent);

    // Append modal to the body
    document.body.appendChild(modal);

    // Show modal on button click
    btn.addEventListener('click', function () {
        modal.classList.add('active'); // Use CSS class for styles
    });

    var modelSelected = 'qa-model-roberta-based';

    // Handle send button click
    sendButton.addEventListener('click', function () {
        var userMessage = inputField.value.trim();
        if (userMessage === '') return;

        if (userMessage.toLowerCase().startsWith('model:')) {
            modelSelected = userMessage.substring(6).trim();
            console.log('Model selected:', modelSelected);
            inputField.value = '';
            return;
        }

        // Display user's message
        var userMessageDiv = document.createElement('div');
        userMessageDiv.innerText = 'You: ' + userMessage;
        chatArea.appendChild(userMessageDiv);

        // Simulate AI response
        var aiMessageDiv = document.createElement('div');
        aiMessageDiv.innerText = 'AI: Thinking...';
        chatArea.appendChild(aiMessageDiv);

        askQuestion(modelSelected, userMessage).then((answer) => {
            aiMessageDiv.innerText = modelSelected + ': ' + answer;
        });
        // Simulate AI delay
        // setTimeout(() => {
        //     aiMessageDiv.innerText = 'AI: This is an example response for: "' + userMessage + '"';
        // }, 1000);

        // Clear input field
        inputField.value = '';
        chatArea.scrollTop = chatArea.scrollHeight; // Scroll to the bottom
    });
});
