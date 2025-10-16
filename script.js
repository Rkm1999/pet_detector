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
let prompts = []; // Will store { id, name, file, tensor } for each prompt
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

    const newPrompt = { id: id, name: `Pet #${promptCount}`, file: null, tensor: null };
    prompts.push(newPrompt);

    document.getElementById(`name-${id}`).addEventListener('input', (e) => {
        prompts[id].name = e.target.value || `Pet #${promptCount}`;
    });

    // When a user uploads a file, process it into a tensor immediately.
    document.getElementById(`file-${id}`).addEventListener('change', async (e) => {
        if (e.target.files[0]) {
            prompts[id].file = e.target.files[0];
            const image = new Image();
            image.src = URL.createObjectURL(prompts[id].file);
            await image.decode();
            // NOTE: This assumes the prompt image is processed the same way as the main image.
            // A true promptable model might require a different kind of preprocessing for prompts.
            prompts[id].tensor = preprocessImage(image);
            console.log(`Prompt for ${prompts[id].name} has been processed.`);
        }
    });
}

/**
 * Loads the ONNX model using the ONNX Runtime for Web.
 */
async function loadModel() {
    console.log("Loading model...");
    try {
        // IMPORTANT: Replace './yoloe-11s-seg.onnx' with the filename of your NEW, promptable model.
        session = await ort.InferenceSession.create('./yoloe-11s-seg-promptable.onnx', {
            executionProviders: ['webgl', 'wasm'],
        });
        console.log("Model loaded successfully using:", session.executionProvider);
    } catch (error) {
        console.error("Failed to load the model:", error);
        alert("Error: Could not load the AI model. Make sure you have the correct, two-input .onnx file.");
    }
}

/**
 * Main detection loop that runs for every frame of the video.
 */
async function runDetection() {
    if (!session || !video.srcObject) {
        requestAnimationFrame(runDetection);
        return;
    }

    const modelInput = preprocessFrame(video);
    const promptTensors = prompts.filter(p => p.tensor).map(p => p.tensor.data);

    if (promptTensors.length === 0) {
        // If there are no prompts, we can't run a promptable model.
        // We could add logic here to run in a "prompt-free" mode if the model supports it.
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
        requestAnimationFrame(runDetection);
        return;
    }

    // Combine all individual prompt tensors into a single "batch" tensor.
    const combinedPromptData = new Float32Array(promptTensors.map(d => [...d]).flat());
    const promptInputs = new ort.Tensor('float32', combinedPromptData, [promptTensors.length, 3, modelWidth, modelHeight]);

    // !! CRITICAL !!
    // The input names 'images' and 'prompt_embeddings' are educated guesses.
    // You MUST verify these names using Netron with your NEW promptable .onnx model.
    const feeds = {
        images: modelInput,
        prompt_embeddings: promptInputs 
    };

    try {
        const results = await session.run(feeds);
        // Also verify the output name(s) with Netron. 'output0' is a common default.
        const detections = postprocessResults(results.output0);
        drawDetections(detections);
    } catch (error) {
        console.error("Error during model inference:", error);
    }

    requestAnimationFrame(runDetection);
}

/**
 * Preprocesses a single image (from video or file) into a tensor.
 */
function preprocessImage(imageSource) {
    tempCtx.drawImage(imageSource, 0, 0, modelWidth, modelHeight);
    const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
    const { data } = imageData;

    const red = [], green = [], blue = [];
    for (let i = 0; i < data.length; i += 4) {
        red.push(data[i] / 255);
        green.push(data[i + 1] / 255);
        blue.push(data[i + 2] / 255);
    }
    const transposedData = [...red, ...green, ...blue];
    return new ort.Tensor('float32', new Float32Array(transposedData), [1, 3, modelWidth, modelHeight]);
}

function preprocessFrame(imageSource) {
    return preprocessImage(imageSource);
}


/**
 * Decodes the raw model output into a clean list of detections.
 */
function postprocessResults(outputTensor) {
    const originalData = outputTensor.data;
    const outputShape = outputTensor.dims; 
    const numDetections = outputShape[2]; // e.g., 8400
    const detectionSize = outputShape[1]; // e.g., 116
    const numClasses = prompts.length; // The number of classes is now the number of prompts we provided

    const transposedData = [];
    for (let i = 0; i < numDetections; i++) {
        const detection = [];
        for (let j = 0; j < detectionSize; j++) {
            detection.push(originalData[j * numDetections + i]);
        }
        transposedData.push(detection);
    }
    
    const boxes = [];
    for (let i = 0; i < numDetections; i++) {
        const detection = transposedData[i];
        // For a promptable model, the class scores directly correspond to our prompts.
        const classScores = detection.slice(4, 4 + numClasses); 
        
        let bestClassIndex = -1;
        let maxScore = 0;
        classScores.forEach((score, index) => {
            if (score > maxScore) {
                maxScore = score;
                bestClassIndex = index;
            }
        });

        if (maxScore > confidenceThreshold) {
            const [x_center, y_center, width, height] = detection.slice(0, 4);
            boxes.push({
                box: [
                    (x_center - width / 2) * (canvas.width / modelWidth),
                    (y_center - height / 2) * (canvas.height / modelHeight),
                    width * (canvas.width / modelWidth),
                    height * (canvas.height / modelHeight)
                ],
                label: prompts[bestClassIndex]?.name || `Object ${bestClassIndex}`,
                score: maxScore
            });
        }
    }

    return nonMaxSuppression(boxes, 0.5);
}

/**
 * Draws the bounding boxes and labels onto the canvas.
 */
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    detections.forEach(det => {
        const [x, y, width, height] = det.box;
        const label = `${det.label}: ${Math.round(det.score * 100)}%`;
        
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);
        
        ctx.fillStyle = '#00FF00';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y > 20 ? y - 20 : y, textWidth + 10, 20);
        
        ctx.fillStyle = '#000000';
        ctx.font = '16px sans-serif';
        ctx.fillText(label, x + 5, y > 20 ? y - 5 : y + 15);
    });
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
            if (sortedBoxes[i].label !== currentBox.label) continue;
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
    handlePromptCreation();

    await setupCamera();
    video.play();
    await loadModel();
    runDetection();
}

// Start the app!
main();

