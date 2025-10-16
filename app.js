document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('output-canvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('start-button');
    const promptUpload = document.getElementById('prompt-upload');
    const statusDiv = document.getElementById('status');

    let session;
    let visualPromptTensor = null;

    // 1. Load the ONNX model
    async function loadModel() {
        try {
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.0/dist/';
            
            // Using the model name from your export script
            session = await ort.InferenceSession.create('./yoloe-11s-seg.onnx', { executionProviders: ['wasm'] });
            
            statusDiv.textContent = 'Model loaded successfully!';
            startButton.disabled = false;
        } catch (error) {
            console.error('Failed to load the model:', error);
            statusDiv.textContent = 'Error: Could not load model. Check console.';
        }
    }

    loadModel();

    // 2. Handle visual prompt upload
    promptUpload.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            const img = new Image();
            img.onload = async () => {
                visualPromptTensor = await preprocessImage(img); 
                statusDiv.textContent = `Prompt image '${file.name}' loaded.`;
                console.log("Visual prompt processed and stored.");
            };
            img.src = e.target.result;
        };
        reader.readDataURL(file);
    });

    // 3. Start camera and run detection
    startButton.addEventListener('click', async () => {
        if (!session) {
            alert('Model is not loaded yet.');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    facingMode: 'environment' // Prefer the rear camera
                }
            });
            video.srcObject = stream;
            
            // --- FIX: Manually trigger play() to start the video feed ---
            video.play(); 
            
            video.onloadedmetadata = () => {
                // --- IMPROVEMENT: Adjust container to match video aspect ratio ---
                const videoAspectRatio = video.videoWidth / video.videoHeight;
                const container = document.getElementById('video-container');
                container.style.paddingTop = `${(1 / videoAspectRatio) * 100}%`;

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                statusDiv.textContent = 'Camera started. Detecting...';
                
                detectFrame();
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
            statusDiv.textContent = 'Error: Could not access camera.';
        }
    });

    // 4. Preprocess image data to a tensor
    async function preprocessImage(image) {
        const modelWidth = 640;
        const modelHeight = 640;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = modelWidth;
        tempCanvas.height = modelHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(image, 0, 0, modelWidth, modelHeight);
        
        const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
        const { data } = imageData;
        
        const float32Data = new Float32Array(1 * 3 * modelWidth * modelHeight);
        
        for (let i = 0; i < modelWidth * modelHeight; i++) {
            const r = data[i * 4 + 0] / 255.0;
            const g = data[i * 4 + 1] / 255.0;
            const b = data[i * 4 + 2] / 255.0;
            
            float32Data[i] = r;
            float32Data[i + modelWidth * modelHeight] = g;
            float32Data[i + 2 * modelWidth * modelHeight] = b;
        }

        return new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
    }

    // 5. The main detection loop
    async function detectFrame() {
        if (video.paused || video.ended) return;

        const inputTensor = await preprocessImage(video);
        const feeds = { images: inputTensor };
        if (visualPromptTensor) {
            feeds.prompt = visualPromptTensor;
        }

        try {
            const results = await session.run(feeds);
            const outputTensor = results.output0; 
            processAndDraw(outputTensor.data, video.videoWidth, video.videoHeight);
        } catch (error) {
            console.error('Inference error:', error);
        }

        requestAnimationFrame(detectFrame);
    }

    // 6. Process model output and draw detections
    function processAndDraw(outputData, originalWidth, originalHeight) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const stride = 6; 
        for (let i = 0; i < outputData.length; i += stride) {
            const score = outputData[i + 4];
            
            if (score > 0.5) {
                const x1 = outputData[i] * originalWidth;
                const y1 = outputData[i + 1] * originalHeight;
                const x2 = outputData[i + 2] * originalWidth;
                const y2 = outputData[i + 3] * originalHeight;
                const label = `Object ${outputData[i+5]}`;

                ctx.strokeStyle = '#1e88e5';
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillStyle = '#1e88e5';
                ctx.font = '18px Arial';
                ctx.fillText(`${label} (${score.toFixed(2)})`, x1, y1 > 20 ? y1 - 5 : y1 + 20);
            }
        }
    }
});

