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

// --- COCO Class Names ---
// This model was likely trained on the COCO dataset, which has these 80 classes.
const classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
];

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
    if (!session || !video.srcObject) {
        requestAnimationFrame(runDetection);
        return;
    }

    const modelInput = preprocessFrame(video);
    
    // UPDATED: The model only takes one input: 'images'.
    // We no longer send the visual prompts.
    const feeds = { images: modelInput };

    try {
        const results = await session.run(feeds);
        // The detection data is in 'output0' as confirmed by Netron.
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
function preprocessFrame(imageSource) {
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


/**
 * Decodes the raw model output into a clean list of detections.
 * @param {ort.Tensor} outputTensor - The output from the model.
 * @returns {Array} - A list of detected objects.
 */
function postprocessResults(outputTensor) {
    // The output shape is [1, 116, 8400]. We need to transpose it to [1, 8400, 116]
    // to easily iterate through all 8400 potential detections.
    const originalData = outputTensor.data;
    const outputShape = outputTensor.dims; // [1, 116, 8400]
    const numDetections = outputShape[2]; // 8400
    const detectionSize = outputShape[1]; // 116
    const numClasses = classNames.length; // 80

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
        const classScores = detection.slice(4, 4 + numClasses); // Scores for 80 classes
        
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
                label: classNames[bestClassIndex],
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
            if (sortedBoxes[i].label !== currentBox.label) continue; // Only compare boxes of the same class
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
    // This UI is now disconnected from the model but we can leave it.
    addPromptBtn.addEventListener('click', handlePromptCreation);
    handlePromptCreation();

    await setupCamera();
    video.play();
    await loadModel();
    runDetection();
}

// Start the app!
main();

