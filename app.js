document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('output-canvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('start-button');
    const promptUpload = document.getElementById('prompt-upload');
    const statusDiv = document.getElementById('status');

    let session;
    let visualPromptTensor = null; // To store the processed prompt image

    // 1. Load the ONNX model
    async function loadModel() {
        try {
            // Use 'wasm' as the backend for broad compatibility, especially on mobile
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.0/dist/';
            
            // NOTE: Replace 'yoloe.onnx' with the actual name of your converted model file.
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
                // You'll need a function to preprocess the image into a tensor
                // This is a placeholder for a more complex preprocessing step
                visualPromptTensor = await preprocessImage(img); 
                statusDiv.textContent = `Prompt image '${file.name}' loaded.`;
                console.log("Visual prompt processed and stored.");
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
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
            video.onloadedmetadata = () => {
                // Set canvas dimensions once video is ready
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                statusDiv.textContent = 'Camera started. Detecting...';
                
                // Start the detection loop
                detectFrame();
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
            statusDiv.textContent = 'Error: Could not access camera.';
        }
    });

    // 4. Preprocess image data to a tensor
    // This is a CRITICAL and model-specific step. You MUST adjust this
    // function based on your YOLOE model's expected input format 
    // (e.g., size 640x640, normalization, etc.).
    async function preprocessImage(image) {
        const modelWidth = 640; // Example dimension
        const modelHeight = 640; // Example dimension

        // Use a temporary canvas to resize the image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = modelWidth;
        tempCanvas.height = modelHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(image, 0, 0, modelWidth, modelHeight);
        
        const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
        const { data } = imageData;
        
        // Convert pixel data to float32 tensor [batch_size, channels, height, width]
        const float32Data = new Float32Array(1 * 3 * modelWidth * modelHeight);
        
        for (let i = 0; i < modelWidth * modelHeight; i++) {
            // Normalize pixels to range [0, 1]
            const r = data[i * 4 + 0] / 255.0;
            const g = data[i * 4 + 1] / 255.0;
            const b = data[i * 4 + 2] / 255.0;
            
            // Store in NCHW format
            float32Data[i] = r;
            float32Data[i + modelWidth * modelHeight] = g;
            float32Data[i + 2 * modelWidth * modelHeight] = b;
        }

        return new ort.Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
    }

    // 5. The main detection loop
    async function detectFrame() {
        if (video.paused || video.ended) return;

        // Preprocess the current video frame
        const inputTensor = await preprocessImage(video);

        // Prepare the model inputs. The names ('images', 'prompt') must match
        // the names your ONNX model expects.
        const feeds = { images: inputTensor };
        if (visualPromptTensor) {
            feeds.prompt = visualPromptTensor; // Add prompt if it exists
        }

        try {
            // Run inference
            const results = await session.run(feeds);
            
            // The output name 'output0' is a placeholder. 
            // You MUST check your model's actual output name.
            const outputTensor = results.output0; 

            // Process results and draw on canvas
            processAndDraw(outputTensor.data, video.videoWidth, video.videoHeight);

        } catch (error) {
            console.error('Inference error:', error);
        }

        // Loop
        requestAnimationFrame(detectFrame);
    }

    // 6. Process model output and draw detections
    function processAndDraw(outputData, originalWidth, originalHeight) {
        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // This function is HIGHLY model-specific.
        // You need to parse the flat 'outputData' array to get bounding boxes,
        // scores, and labels based on your model's output structure.
        
        // Example placeholder logic:
        // Assume output is [x1, y1, x2, y2, score, class_id] for each detection
        const stride = 6; 
        for (let i = 0; i < outputData.length; i += stride) {
            const score = outputData[i + 4];
            
            if (score > 0.5) { // Confidence threshold
                const x1 = outputData[i] * originalWidth;
                const y1 = outputData[i + 1] * originalHeight;
                const x2 = outputData[i + 2] * originalWidth;
                const y2 = outputData[i + 3] * originalHeight;
                const label = `Object ${outputData[i+5]}`; // Get class label

                // Draw bounding box
                ctx.strokeStyle = '#1e88e5';
                ctx.lineWidth = 4;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Draw label
                ctx.fillStyle = '#1e88e5';
                ctx.font = '18px Arial';
                ctx.fillText(`${label} (${score.toFixed(2)})`, x1, y1 > 20 ? y1 - 5 : y1 + 20);
            }
        }
    }
});
