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
            executionProviders: ['webgl', 'wasm'], // <-- UPDATED LINE
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
    // TODO: Implement this!
    // This function needs to take the video frame, resize it to your model's expected
    // input size (e.g., 640x640), convert it to a Float32Array, normalize the pixel
    // values (e.g., divide by 255), and arrange it into the correct tensor shape 
    // (e.g., [1, 3, 640, 640] for Batch, Channels, Height, Width).
    // You will need to draw the video to a temporary canvas to get pixel data.
    
    // Placeholder returns an empty tensor.
    const placeholderTensor = new ort.Tensor('float32', new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
    return placeholderTensor;
}

async function preprocessPrompts() {
    // TODO: Implement this!
    // This function needs to loop through the file inputs, read the uploaded images,
    // and process them just like the video frame (resize, normalize, etc.).
    // You will likely need to combine them into a single tensor for the model.
    
    // Placeholder returns an empty tensor.
    const placeholderTensor = new ort.Tensor('float32', new Float32Array(0));
    return placeholderTensor;
}

function postprocessResults(results) {
    // TODO: Implement this!
    // This function needs to parse the output tensors from the model.
    // The raw output is usually a large array of numbers. You'll need to decode this
    // to find the coordinates of bounding boxes, the confidence scores, and the
    // class labels for each detected object. You might also need to apply
    // Non-Max Suppression (NMS) to filter out duplicate boxes.
    
    // Placeholder returns an empty array.
    return []; 
    
    // Example of what the final output should look like:
    /*
    return [
        { box: [x1, y1, width, height], label: 'Fluffy', score: 0.95 },
        { box: [x2, y2, width, height], label: 'Whiskers', score: 0.89 }
    ];
    */
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