// --- Get references to our HTML elements ---
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const addPromptBtn = document.getElementById('add-prompt-btn');
const promptsContainer = document.getElementById('prompts-container');

// Global variable to hold the AI model session
let session;
let promptCount = 0;

/**
 * Sets up the smartphone camera and gets the video stream.
 */
async function setupCamera() {
    // Request access to the user's media devices
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { 
            facingMode: 'environment', // Use the rear camera
            width: { ideal: 1280 }, // Request a higher resolution if available
            height: { ideal: 720 }
        }
    });
    video.srcObject = stream;

    // Return a promise that resolves once the video metadata is loaded
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            // Match canvas dimensions to the actual video dimensions
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve(video);
        };
    });
}

/**
 * Handles the "Add Another Pet" button clicks to dynamically create UI.
 */
function handlePromptCreation() {
    promptCount++;
    const promptDiv = document.createElement('div');
    promptDiv.innerHTML = `
        <input type="text" placeholder="Pet Name #${promptCount}" id="name-${promptCount}">
        <input type="file" accept="image/*" id="file-${promptCount}">
    `;
    promptsContainer.appendChild(promptDiv);
}

/**
 * Loads the ONNX model using the ONNX Runtime for Web.
 */
async function loadModel() {
    console.log("Loading model...");
    try {
        // Create an inference session with the WebGL backend for GPU acceleration
        session = await ort.InferenceSession.create('./yoloe-11s-seg.onnx', {
            executionProviders: ['webgl','wasm'],
        });
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Failed to load the model:", error);
        alert("Error: Could not load the AI model. Check the console for details.");
    }
}

/**
 * Main detection loop that runs for every frame of the video.
 */
async function runDetection() {
    if (!session) {
        // Don't run if the model isn't loaded yet
        requestAnimationFrame(runDetection);
        return;
    }

    // 1. Pre-process the current video frame for the model
    // This is a placeholder! You MUST implement this based on your model's needs.
    const modelInput = preprocessFrame(video);

    // 2. Pre-process any visual prompts the user has uploaded
    // This is also a placeholder for your custom logic.
    const promptInputs = await preprocessPrompts();

    // 3. Create the input feed object for the model
    // The keys ('images', 'prompts') must exactly match the input names of your ONNX model.
    const feeds = {
        images: modelInput,
        prompts: promptInputs 
    };

    // 4. Run inference
    const results = await session.run(feeds);

    // 5. Post-process the model's output to get readable data
    // This is the final placeholder for you to implement.
    const detections = postprocessResults(results);

    // 6. Draw the results on the canvas
    drawDetections(detections);

    // 7. Request the next frame to continue the loop
    requestAnimationFrame(runDetection);
}

/**
 * Draws the bounding boxes and labels onto the canvas.
 * @param {Array} detections - An array of detection objects.
 */
function drawDetections(detections) {
    // Clear the previous frame's drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const [x, y, width, height] = det.box;
        const label = `${det.label}: ${Math.round(det.score * 100)}%`;
        
        // Draw the bounding box
        ctx.strokeStyle = '#00FF00'; // Green color for the box
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);
        
        // Draw the label background
        ctx.fillStyle = '#00FF00';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 20, textWidth + 10, 20);
        
        // Draw the label text
        ctx.fillStyle = '#000000'; // Black text
        ctx.font = '16px sans-serif';
        ctx.fillText(label, x + 5, y - 5);
    });
}

// --- Placeholder Functions You Must Implement ---

function preprocessFrame(videoElement) {
    const modelWidth = 640;
    const modelHeight = 640;
    
    // 1. Create a temporary canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = modelWidth;
    tempCanvas.height = modelHeight;
    const tempCtx = tempCanvas.getContext('2d');

    // 2. Draw and resize the video frame
    tempCtx.drawImage(videoElement, 0, 0, modelWidth, modelHeight);
    
    // 3. Get pixel data
    const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData; // data is a Uint8ClampedArray [R,G,B,A, R,G,B,A, ...]

    // 4. Create the Float32Array
    const inputData = new Float32Array(1 * 3 * modelWidth * modelHeight);
    
    // 5. Loop and arrange data in NCHW format
    for (let i = 0; i < modelWidth * modelHeight; i++) {
        // Normalize pixel values to be between 0 and 1
        const r = data[i * 4 + 0] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;
        
        // inputData is [R channel, G channel, B channel]
        inputData[i] = r;                                 // Red channel
        inputData[i + (modelWidth * modelHeight)] = g;    // Green channel
        inputData[i + (modelWidth * modelHeight) * 2] = b; // Blue channel
    }

    // Create the ONNX Tensor
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, modelHeight, modelWidth]);
    return inputTensor;
}
async function preprocessPrompts() {
    const promptPromises = [];
    const promptInputs = document.querySelectorAll('#prompts-container input[type="file"]');
    
    promptInputs.forEach(input => {
        if (input.files && input.files[0]) {
            promptPromises.push(processSinglePrompt(input.files[0]));
        }
    });

    // Wait for all images to be loaded and processed
    const promptTensors = await Promise.all(promptPromises);

    if (promptTensors.length === 0) {
        // Return an empty tensor if no prompts are provided
        // The model must be able to handle this case!
        return new ort.Tensor('float32', new Float32Array(0), [0, 3, 640, 640]);
    }
    
    // Combine individual tensors into a single batch tensor
    // This assumes you are concatenating along the 'batch' dimension
    const combinedData = new Float32Array(promptTensors.length * 3 * 640 * 640);
    promptTensors.forEach((tensor, i) => {
        combinedData.set(tensor.data, i * 3 * 640 * 640);
    });

    return new ort.Tensor('float32', combinedData, [promptTensors.length, 3, 640, 640]);
}

/**
 * Helper function to load and process one image file.
 * This re-uses the same logic as preprocessFrame.
 */
function processSinglePrompt(file) {
    return new Promise((resolve, reject) => {
        const modelWidth = 640;
        const modelHeight = 640;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = modelWidth;
        tempCanvas.height = modelHeight;
        const tempCtx = tempCanvas.getContext('2d');
        
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            tempCtx.drawImage(img, 0, 0, modelWidth, modelHeight);
            const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
            const { data } = imageData;
            
            const inputData = new Float32Array(1 * 3 * modelWidth * modelHeight);
            for (let i = 0; i < modelWidth * modelHeight; i++) {
                inputData[i] = data[i * 4 + 0] / 255.0;
                inputData[i + (modelWidth * modelHeight)] = data[i * 4 + 1] / 255.0;
                inputData[i + (modelWidth * modelHeight) * 2] = data[i * 4 + 2] / 255.0;
            }
            
            // Note: We create a tensor here but will use its .data property later
            const tensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
            resolve(tensor);
        };
        img.onerror = reject;
    });
}
function postprocessResults(results) {
    // NOTE: The output name 'output0' is a placeholder. 
    // You MUST find the correct name from your model's documentation or a tool like Netron.
    const outputTensor = results.output0; 
    const data = outputTensor.data; // This is a flat Float32Array
    const detections = [];

    // The shape and structure of 'data' is highly model-dependent.
    // Let's assume the output is [batch, num_detections, 6] where 6 is [x, y, w, h, score, class_id]
    // You must verify this structure!
    const numDetections = outputTensor.dims[1]; // Example: getting the number of detections
    const numColumns = outputTensor.dims[2]; // Example: getting the number of values per detection

    for (let i = 0; i < numDetections; i++) {
        const offset = i * numColumns;
        const score = data[offset + 4];
        
        // 4. Filter by confidence threshold
        if (score < 0.5) { // 50% threshold
            continue;
        }

        // 5. Scale coordinates
        const x = data[offset + 0] * (canvas.width / 640); // Scale from model width to canvas width
        const y = data[offset + 1] * (canvas.height / 640); // Scale from model height to canvas height
        const w = data[offset + 2] * (canvas.width / 640);
        const h = data[offset + 3] * (canvas.height / 640);
        
        const classId = data[offset + 5];
        
        // Get the pet's name from the input field based on the classId
        const petNameInput = document.getElementById(`name-${classId + 1}`);
        const label = petNameInput ? petNameInput.value : `Pet #${classId}`;
        
        detections.push({
            box: [x - w / 2, y - h / 2, w, h], // Convert center x,y to top-left x,y
            label: label,
            score: score
        });
    }

    // 6. You would apply Non-Max Suppression (NMS) here to the 'detections' array.
    // For simplicity, we'll skip it in this example, but it's important for accuracy.

    // 7. Return the final formatted list
    return detections;
}
/**
 * The main function that starts the application.
 */
async function main() {
    addPromptBtn.addEventListener('click', handlePromptCreation);
    handlePromptCreation(); // Add the first prompt input on page load

    await setupCamera();
    video.play();
    await loadModel();
    runDetection();
}

// Start the app!
main();