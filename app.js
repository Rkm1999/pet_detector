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
            
            video.play(); 
            
            video.onloadedmetadata = () => {
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
            // NOTE: The input name for the prompt must match your ONNX model.
            // It could be 'prompt', 'visual_prompt', etc. Inspect your model to be sure.
            feeds.prompt = visualPromptTensor;
        }

        try {
            const results = await session.run(feeds);
            
            // NOTE: The output name must match your model. 'output0' is common.
            const outputTensor = results.output0; 

            // The new, corrected processing and drawing function
            processAndDraw(outputTensor, video.videoWidth, video.videoHeight);
        } catch (error) {
            console.error('Inference error:', error);
        }

        requestAnimationFrame(detectFrame);
    }
    
    // --- NEW AND IMPROVED POST-PROCESSING LOGIC ---

    // 6. Process model output and draw detections
    function processAndDraw(outputTensor, originalWidth, originalHeight) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const modelWidth = 640;
        const modelHeight = 640;
        
        // Transpose the output from [batch, features, num_boxes] to [batch, num_boxes, features]
        const transposedData = transpose(outputTensor.data, outputTensor.dims[1], outputTensor.dims[2]);
        const boxes = [];

        // Loop through all potential detections
        for (let i = 0; i < transposedData.length; i++) {
            const detection = transposedData[i];
            const [x_center, y_center, width, height, ...classProbs] = detection;
            
            // Find the class with the highest probability
            let maxProb = 0;
            let classId = -1;
            for(let j = 0; j < classProbs.length; j++) {
                if (classProbs[j] > maxProb) {
                    maxProb = classProbs[j];
                    classId = j;
                }
            }

            // Apply confidence threshold
            if (maxProb > 0.5) {
                const scaleX = originalWidth / modelWidth;
                const scaleY = originalHeight / modelHeight;

                const x1 = (x_center - width / 2) * scaleX;
                const y1 = (y_center - height / 2) * scaleY;
                const x2 = (x_center + width / 2) * scaleX;
                const y2 = (y_center + height / 2) * scaleY;

                boxes.push({
                    x1: x1, y1: y1, x2: x2, y2: y2,
                    score: maxProb,
                    classId: classId
                });
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        const finalBoxes = nms(boxes, 0.45);

        // Draw the final, filtered boxes
        finalBoxes.forEach(box => {
            const label = `Object ${box.classId}`;
            const score = box.score.toFixed(2);
            
            // Drawing the box
            ctx.strokeStyle = '#1e88e5';
            ctx.lineWidth = 4;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

            // Drawing the label background
            ctx.fillStyle = '#1e88e5';
            const textWidth = ctx.measureText(`${label} (${score})`).width;
            ctx.fillRect(box.x1 - 2, box.y1 > 20 ? box.y1 - 22 : box.y1, textWidth + 4, 22);
            
            // Drawing the label text
            ctx.fillStyle = '#ffffff';
            ctx.font = '16px Arial';
            ctx.fillText(`${label} (${score})`, box.x1, box.y1 > 20 ? box.y1 - 5 : box.y1 + 16);
        });
    }

    // 7. Helper function to transpose the model output
    function transpose(data, C, N) {
        const transposed = new Array(N);
        for (let i = 0; i < N; i++) {
            transposed[i] = new Float32Array(C);
            for (let j = 0; j < C; j++) {
                transposed[i][j] = data[j * N + i];
            }
        }
        return transposed;
    }

    // 8. Helper function for Intersection over Union (IoU)
    function iou(box1, box2) {
        const x1 = Math.max(box1.x1, box2.x1);
        const y1 = Math.max(box1.y1, box2.y1);
        const x2 = Math.min(box1.x2, box2.x2);
        const y2 = Math.min(box1.y2, box2.y2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
        const union = area1 + area2 - intersection;

        return intersection / union;
    }

    // 9. Helper function for Non-Maximum Suppression (NMS)
    function nms(boxes, iouThreshold) {
        if (boxes.length === 0) return [];
        
        boxes.sort((a, b) => b.score - a.score);
        const selectedBoxes = [];
        
        while (boxes.length > 0) {
            selectedBoxes.push(boxes[0]);
            boxes = boxes.slice(1).filter(box => iou(boxes[0], box) < iouThreshold);
        }
        
        return selectedBoxes;
    }
});

