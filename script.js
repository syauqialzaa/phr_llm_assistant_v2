const API_CONFIG = {
    REST_URL: null,
    WS_URL: null,
    DEFAULT_URL: window.localStorage.getItem('NGROK_URL') || 'http://localhost:8080'
};

// WebSocket connection
let ws = null;

// Initialize API configuration
async function initializeApiConfig() {
    const savedConfig = localStorage.getItem('API_CONFIG');
    if (savedConfig) {
        const config = JSON.parse(savedConfig);
        API_CONFIG.REST_URL = config.REST_URL;
        API_CONFIG.WS_URL = config.WS_URL;
    }
    
    if (!API_CONFIG.REST_URL) {
        let apiUrl = prompt("Please enter the API URL (from ngrok):", "");
        if (apiUrl) {
            apiUrl = apiUrl.replace(/\/$/, "");
            const wsUrl = apiUrl.replace(/^http/, 'ws') + '/ws';
            
            API_CONFIG.REST_URL = apiUrl;
            API_CONFIG.WS_URL = wsUrl;
            localStorage.setItem('API_CONFIG', JSON.stringify(API_CONFIG));
        }
    }
    
    if (API_CONFIG.WS_URL) {
        initializeWebSocket();
    }
}

// Initialize WebSocket connection
function initializeWebSocket() {
    if (!API_CONFIG.WS_URL) return;

    ws = new WebSocket(API_CONFIG.WS_URL);
    
    ws.onopen = () => {
        updateConnectionStatus(true);
        addSystemMessage('Connected to chat server');
    };
    
    ws.onclose = () => {
        updateConnectionStatus(false);
        setTimeout(initializeWebSocket, 5000); // Retry connection after 5 seconds
    };
    
    ws.onerror = () => {
        updateConnectionStatus(false);
        addSystemMessage('Error connecting to chat server');
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleResponse(data);
        } catch (error) {
            addSystemMessage('Error processing server response');
        }
    };
}

// Update connection status display
function updateConnectionStatus(connected) {
    const statusDiv = document.getElementById('connection-status');
    statusDiv.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
    statusDiv.textContent = connected ? 'Connected' : 'Disconnected';
}

// Add system message to chat
function addSystemMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    const systemDiv = document.createElement('div');
    systemDiv.className = 'message system-message';
    systemDiv.textContent = message;
    chatMessages.appendChild(systemDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Create message element
function createMessageElement(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    messageDiv.textContent = content;
    return messageDiv;
}

// Handle sending questions
async function sendQuestion() {
    const questionInput = document.getElementById('question-input');
    const chatMessages = document.getElementById('chat-messages');
    const question = questionInput.value.trim();
    
    if (!question) return;
    
    chatMessages.appendChild(createMessageElement(question, true));
    questionInput.value = '';
    
    const messageData = {
        question: question,
        timestamp: new Date().toISOString()
    };
    
    try {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(messageData));
        } else {
            const response = await fetch(`${API_CONFIG.REST_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(messageData),
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            handleResponse(data);
        }
    } catch (error) {
        addSystemMessage('Sorry, there was an error processing your request.');
    }
}

// Handle response from server
function handleResponse(data) {
    const chatMessages = document.getElementById('chat-messages');
    
    if (data.type === 'error') {
        addSystemMessage(`Error: ${data.message}`);
        return;
    }
    
    const assistantContainer = document.createElement('div');
    assistantContainer.className = 'message assistant-message';
    
    if (data.explanation) {
        const explanationDiv = document.createElement('div');
        explanationDiv.textContent = data.explanation;
        assistantContainer.appendChild(explanationDiv);
    }
    
    if (data.sql) {
        const sqlDiv = document.createElement('div');
        sqlDiv.className = 'sql-code';
        sqlDiv.textContent = data.sql;
        assistantContainer.appendChild(sqlDiv);
    }
    
    if (data.visualization) {
        const img = document.createElement('img');
        img.className = 'visualization';
        img.src = `data:image/png;base64,${data.visualization}`;
        assistantContainer.appendChild(img);

        if (data.visualization_explanation) {
          const vizExp = document.createElement('div');
          vizExp.className = 'visualization-explanation';
          vizExp.textContent = data.visualization_explanation;
          assistantContainer.appendChild(vizExp);
      }
    }
    
    if (data.data && data.data.length > 0) {
        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'data-table-wrapper';
        const table = document.createElement('table');
        table.className = 'data-table';
        
        // Add headers
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        Object.keys(data.data[0]).forEach(key => {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Add data rows
        const tbody = document.createElement('tbody');
        data.data.forEach(row => {
            const tr = document.createElement('tr');
            Object.values(row).forEach(value => {
                const td = document.createElement('td');
                td.textContent = value !== null ? value : '';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        
        tableWrapper.appendChild(table);
        assistantContainer.appendChild(tableWrapper);
    }

    if (data.app_url) {
        const appLink = document.createElement('div');
        appLink.className = 'external-app-link';
        
        // Determine app type for more specific message
        let appType = '';
        if (data.app_url.includes('dca')) {
            appType = 'DCA analysis';
        } else if (data.app_url.includes('wellbore')) {
            appType = 'wellbore visualization';
        } else {
            appType = 'detailed visualization';
        }
        
        appLink.innerHTML = `
            <p>For more detailed ${appType}, open the app: <a href="${data.app_url}" target="_blank" rel="noopener noreferrer">${data.app_url}</a></p>
        `;
        assistantContainer.appendChild(appLink);
    }
    
    chatMessages.appendChild(assistantContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeApiConfig();
    
    // Add event listener for config button
    const configBtn = document.getElementById('config-btn');
    if (configBtn) {
        configBtn.addEventListener('click', () => {
            localStorage.removeItem('API_CONFIG');
            if (ws) ws.close();
            initializeApiConfig();
        });
    }
    
    // Add event listener for Enter key
    document.getElementById('question-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuestion();
        }
    });
});
