let useWebcam = false;
let lastPrediction = "";

const uploadTab = document.getElementById("uploadTab");
const liveTab = document.getElementById("liveTab");
const uploadSection = document.getElementById("uploadSection");
const liveSection = document.getElementById("liveSection");
const dropZone = document.getElementById("dropZone");
const mediaUpload = document.getElementById("mediaUpload"); // Updated input
const preview = document.getElementById("preview");
const webcam = document.getElementById("webcam");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("result");

uploadTab.addEventListener("click", () => {
  useWebcam = false;
  uploadSection.classList.remove("hidden");
  liveSection.classList.add("hidden");
  uploadTab.classList.add("active");
  liveTab.classList.remove("active");
});

liveTab.addEventListener("click", () => {
  useWebcam = true;
  uploadSection.classList.add("hidden");
  liveSection.classList.remove("hidden");
  liveTab.classList.add("active");
  uploadTab.classList.remove("active");
  startWebcam();
});

dropZone.addEventListener("click", () => mediaUpload.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.background = "#e0f2fe";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.background = "#f9fafb";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.background = "#f9fafb";
  handleMedia(e.dataTransfer.files[0]);
});

mediaUpload.addEventListener("change", function () {
  handleMedia(this.files[0]);
});

function handleMedia(file) {
  if (!file) return;

  preview.innerHTML = "";

  const url = URL.createObjectURL(file);

  if (file.type.startsWith("image/")) {
    const img = document.createElement("img");
    img.src = url;
    img.style.maxWidth = "100%";
    preview.appendChild(img);

  } else if (file.type.startsWith("video/")) {
    const video = document.createElement("video");
    video.src = url;
    video.controls = true;
    preview.appendChild(video);
  }
}


async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
  } catch (err) {
    alert("Webcam access denied!");
  }
}

predictBtn.addEventListener("click", async () => {
  loader.classList.remove("hidden");
  lastPrediction = "";

  try {
    const formData = new FormData();

    if (useWebcam) {
      loader.classList.add("hidden");
      startLivePrediction();
      return;
    }

    const file = mediaUpload.files[0];
    if (!file) {
      alert("Please upload a file first!");
      loader.classList.add("hidden");
      return;
    }

    formData.append("file", file);

    let endpoint = file.type.startsWith("image/") ? "/predict_image" : "/predict_video";

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    typeText(data.prediction);

  } catch (err) {
    console.error(err);
    alert("Error contacting server!");
  }

  loader.classList.add("hidden");
});

function startLivePrediction() {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const interval = setInterval(async () => {
    if (!useWebcam) return clearInterval(interval);

    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg")
    );

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const response = await fetch("/predict_image", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      typeText(result.prediction);
    } catch (err) {
      console.error(err);
    }
  }, 500); // Predict every 0.5 seconds
}

function typeText(text) {
  if (text === lastPrediction) return; // do not update if same
  lastPrediction = text;
  resultBox.innerText = text;
}