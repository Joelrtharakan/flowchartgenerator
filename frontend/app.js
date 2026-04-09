import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    fontFamily: "'Outfit', sans-serif"
});

document.addEventListener('DOMContentLoaded', () => {
    const API_URL = '/generate';
    
    // DOM Elements
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const copyBtn = document.getElementById('copy-btn');
    
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const errorState = document.getElementById('error-state');
    const successState = document.getElementById('success-state');
    const errorMessage = document.getElementById('error-message');
    
    const chartContainer = document.getElementById('chart-container');
    const mermaidCode = document.getElementById('mermaid-code');

    // Utility: Manage UI States
    function showState(stateElement) {
        [emptyState, loadingState, errorState, successState].forEach(el => {
            if (el === stateElement) {
                el.classList.remove('hidden');
            } else {
                el.classList.add('hidden');
            }
        });
    }

    // Process Generation
    async function handleGenerate() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // Set UI to loading mode
        generateBtn.disabled = true;
        promptInput.disabled = true;
        showState(loadingState);

        try {
            // Fetch from Mermaid API
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt }),
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const flowchartStr = data.flowchart;

            if (!flowchartStr) {
                throw new Error("API returned an empty flowchart string.");
            }

            // Render output
            await renderMermaid(flowchartStr);
            
            // Show textual code
            mermaidCode.textContent = flowchartStr;
            
            // UI Update
            showState(successState);
        } catch (error) {
            console.error(error);
            errorMessage.textContent = error.message.includes('Failed to fetch') 
                ? 'Network error: Make sure the API server is running.' 
                : error.message;
            showState(errorState);
        } finally {
            // Restore UI inputs
            generateBtn.disabled = false;
            promptInput.disabled = false;
            promptInput.focus();
        }
    }

    // Render logic for Mermaid.js
    async function renderMermaid(code) {
        try {
            // Render the code to an SVG string using mermaid API
            const { svg } = await mermaid.render('dynamic-mermaid-svg', code);
            // Inject SVG
            chartContainer.innerHTML = svg;
        } catch (err) {
            console.error("Mermaid Render Error:", err);
            // If mermaid fails (syntax error), we still want to show the code provided by the model.
            chartContainer.innerHTML = `
                <div style="color: #dc2626; text-align: center; font-size: 0.9rem; padding: 1rem; background: #fee2e2; border-radius: 8px;">
                    Model generated invalid Mermaid syntax. See syntax below.
                </div>
            `;
        }
    }

    // Copy to clipboard
    async function handleCopy() {
        const code = mermaidCode.textContent;
        if (!code) return;
        
        try {
            await navigator.clipboard.writeText(code);
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.style.color = '#10b981';
            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.style.color = '';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy', err);
        }
    }

    // Event Listeners
    generateBtn.addEventListener('click', handleGenerate);

    // Allow Control+Enter or Cmd+Enter to submit
    promptInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            handleGenerate();
        }
    });

    copyBtn.addEventListener('click', handleCopy);
});
