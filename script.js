// --- Get references to our HTML elements ---
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const addPromptBtn = document.getElementById('add-prompt-btn');
const promptsContainer = document.getElementById('prompts-container');

// --- Configuration ---
const modelWidth = 640;
const modelHeight = 640;
const confidenceThreshold = 0.50; // Filter out detections below this confidence

// --- Global State ---
let session;
let prompts = []; // Will store { id, name, file } for each prompt
let promptCount = 0;
let tempCanvas = document.createElement('canvas'); // For preprocessing frames
tempCanvas.width = modelWidth;
tempCanvas.height = modelHeight;
let tempCtx = tempCanvas.getContext('2d');

/**
 * Sets up the smartphone camera and gets the video stream.
 */
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve(video);
        };
    });
}

/**
 * Handles adding new UI elements for prompts and storing their info.
 */
function handlePromptCreation() {
    promptCount++;
    const id = promptCount - 1;

    const promptDiv = document.createElement('div');
    promptDiv.innerHTML = `
        <input type="text" placeholder="Pet Name #${promptCount}" id="name-${id}">
        <input type="file" accept="image/*" id="file-${id}">
    `;
    promptsContainer.appendChild(promptDiv);

    prompts.push({ id: id, name: `Pet #${promptCount}`, file: null });

    document.getElementById(`name-${id}`).addEventListener('input', (e) => {
        prompts[id].name = e.target.value || `Pet #${promptCount}`;
    });

    document.getElementById(`file-${id}`).addEventListener('change', (e) => {
        prompts[id].file = e.target.files[0];
    });
}

/**
 * Loads the ONNX model using the ONNX Runtime for Web.
 */
async function loadModel() {
    console.log("Loading model...");
    try {
        session = await ort.InferenceSession.create('./yoloe-11s-seg.onnx', {
            executionProviders: ['webgl', 'wasm'],
        });
        console.log("Model loaded successfully using:", session.executionProvider);
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
        requestAnimationFrame(runDetection);
        return;
    }

    const modelInput = preprocessFrame(video);
    const promptInputs = await preprocessPrompts();
    
    // IMPORTANT: The input names ('images', 'prompts') must exactly match your model.
    // Use a tool like Netron to inspect your .onnx file to find the correct names.
    const feeds = {
        images: modelInput,
        visual_prompts: promptInputs // This is a common name, but might be different
    };

    try {
        const results = await session.run(feeds);
        // IMPORTANT: The output name 'output0' is a placeholder. Inspect your model!
        const detections = postprocessResults(results.output0);
        drawDetections(detections);
    } catch (error) {
        console.error("Error during model inference:", error);
    }

    requestAnimationFrame(runDetection);
}

/**
 * Preprocesses a single image (from video or file) into a tensor.
 * @param {HTMLVideoElement | HTMLImageElement} imageSource - The image to process.
 * @returns {ort.Tensor} - The processed tensor for the model.
 */
function preprocessImage(imageSource) {
    // 1. Draw the image to a temporary canvas to resize it
    tempCtx.drawImage(imageSource, 0, 0, modelWidth, modelHeight);

    // 2. Get the pixel data
    const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData;

    // 3. Normalize and transpose the data from [H, W, C] to [C, H, W]
    const red = [], green = [], blue = [];
    for (let i = 0; i < data.length; i += 4) {
        red.push(data[i] / 255);
        green.push(data[i + 1] / 255);
        blue.push(data[i + 2] / 255);
    }
    const transposedData = [...red, ...green, ...blue];

    // 4. Create a tensor
    return new ort.Tensor('float32', new Float32Array(transposedData), [1, 3, modelWidth, modelHeight]);
}

/**
 * Preprocesses the current video frame.
 */
function preprocessFrame(videoElement) {
    return preprocessImage(videoElement);
}

/**
 * Preprocesses all uploaded prompt images.
 */
async function preprocessPrompts() {
    const promptTensors = [];
    for (const prompt of prompts) {
        if (prompt.file) {
            const image = new Image();
            image.src = URL.createObjectURL(prompt.file);
            await image.decode(); // Wait for image to load
            const tensor = preprocessImage(image);
            promptTensors.push(tensor.data);
        }
    }

    if (promptTensors.length === 0) {
        // Return a tensor with a shape the model expects for no prompts, e.g., [0, 3, W, H]
        return new ort.Tensor('float32', [], [0, 3, modelWidth, modelHeight]);
    }

    // Concatenate all prompt tensors into a single batch
    const combinedData = new Float32Array(promptTensors.map(d => [...d]).flat());
    return new ort.Tensor('float32', combinedData, [promptTensors.length, 3, modelWidth, modelHeight]);
}

/**
 * Decodes the raw model output into a clean list of detections.
 * @param {ort.Tensor} outputTensor - The output from the model.
 * @returns {Array} - A list of detected objects.
 */
function postprocessResults(outputTensor) {
    const data = outputTensor.data;
    const boxes = [];
    
    // The output shape and format is highly model-specific.
    // This is a common format for YOLO-style models: [batch, num_detections, 4_coords + 1_confidence + num_classes]
    const outputDimensions = outputTensor.dims;
    const numDetections = outputDimensions[1];
    const detectionSize = outputDimensions[2];

    for (let i = 0; i < numDetections; i++) {
        const detection = data.slice(i * detectionSize, (i + 1) * detectionSize);
        const confidence = detection[4];

        if (confidence > confidenceThreshold) {
            const [x_center, y_center, width, height] = detection.slice(0, 4);
            const classScores = detection.slice(5);
            
            let bestClassIndex = -1;
            let maxScore = 0;
            classScores.forEach((score, index) => {
                if (score > maxScore) {
                    maxScore = score;
                    bestClassIndex = index;
                }
            });

            if (bestClassIndex !== -1) {
                boxes.push({
                    box: [
                        (x_center - width / 2) * (canvas.width / modelWidth),
                        (y_center - height / 2) * (canvas.height / modelHeight),
                        width * (canvas.width / modelWidth),
                        height * (canvas.height / modelHeight)
                    ],
                    label: prompts[bestClassIndex]?.name || `Object ${bestClassIndex}`,
                    score: confidence
                });
            }
        }
    }

    return nonMaxSuppression(boxes, 0.5); // Apply NMS with an IoU threshold of 0.5
}

/**
 * Performs Non-Maximum Suppression to filter out overlapping boxes.
 */
function nonMaxSuppression(boxes, iouThreshold) {
    const sortedBoxes = boxes.sort((a, b) => b.score - a.score);
    const selectedBoxes = [];
    
    while (sortedBoxes.length > 0) {
        const currentBox = sortedBoxes.shift();
        selectedBoxes.push(currentBox);
        
        for (let i = sortedBoxes.length - 1; i >= 0; i--) {
            const iou = intersectionOverUnion(currentBox, sortedBoxes[i]);
            if (iou > iouThreshold) {
                sortedBoxes.splice(i, 1);
            }
        }
    }
    
    return selectedBoxes;
}

/**
 * Calculates the Intersection over Union (IoU) of two boxes.
 */
function intersectionOverUnion(box1, box2) {
    const [x1, y1, w1, h1] = box1.box;
    const [x2, y2, w2, h2] = box2.box;

    const interX1 = Math.max(x1, x2);
    const interY1 = Math.max(y1, y2);
    const interX2 = Math.min(x1 + w1, x2 + w2);
    const interY2 = Math.min(y1 + h1, y2 + h2);

    const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    
    const unionArea = box1Area + box2Area - interArea;
    return interArea / unionArea;
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
