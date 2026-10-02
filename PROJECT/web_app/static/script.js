(function () {
  "use strict";

  const form = document.getElementById("predict-form");
  const fileInput = document.getElementById("leaf-image");
  const dropZone = document.getElementById("drop-zone");
  const dropPrompt = document.getElementById("drop-prompt");
  const previewWrap = document.getElementById("preview-wrap");
  const previewImage = document.getElementById("preview-image");
  const changeBtn = document.getElementById("change-image");
  const removeBtn = document.getElementById("remove-image");
  const predictBtn = document.getElementById("predict-btn");
  const formHint = document.getElementById("form-hint");
  const loading = document.getElementById("loading");
  const errorCard = document.getElementById("error-card");
  const resultCard = document.getElementById("result-card");
  const resultTitle = document.getElementById("result-title");
  const resultStatus = document.getElementById("result-status");
  const resultConfidence = document.getElementById("result-confidence");
  const resultNote = document.getElementById("result-note");
  const topList = document.getElementById("top-list");
  const navToggle = document.querySelector(".nav-toggle");
  const siteNav = document.getElementById("site-nav");

  const ALLOWED_TYPES = ["image/jpeg", "image/png"];
  const ALLOWED_EXT = [".jpg", ".jpeg", ".png"];

  let selectedFile = null;
  let previewUrl = null;

  function isAllowedFile(file) {
    if (!file) return false;
    const name = (file.name || "").toLowerCase();
    const hasExt = ALLOWED_EXT.some(function (ext) {
      return name.endsWith(ext);
    });
    const typeOk = !file.type || ALLOWED_TYPES.includes(file.type);
    return hasExt && typeOk;
  }

  function hide(el) {
    el.classList.add("hidden");
  }

  function show(el) {
    el.classList.remove("hidden");
  }

  function clearPreview() {
    selectedFile = null;
    fileInput.value = "";
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    previewImage.removeAttribute("src");
    hide(previewWrap);
    show(dropPrompt);
    predictBtn.disabled = true;
    formHint.textContent = "Select a leaf image to enable prediction.";
  }

  function setFile(file) {
    if (!isAllowedFile(file)) {
      showError("Please choose a JPG, JPEG, or PNG leaf image.");
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    hide(dropPrompt);
    show(previewWrap);
    predictBtn.disabled = false;
    formHint.textContent = "Image ready. Click Predict Disease to run the trained model.";
    hide(errorCard);
    hide(resultCard);
  }

  function showError(message) {
    errorCard.textContent = message;
    show(errorCard);
    hide(resultCard);
  }

  function renderResult(data) {
    resultTitle.textContent = data.display_name || data.predicted_class;
    resultStatus.textContent = data.status;
    resultStatus.className = "status-pill " + (data.status === "Healthy" ? "healthy" : "diseased");
    resultConfidence.textContent = Number(data.confidence).toFixed(2) + "%";
    resultNote.textContent = data.confidence_note || "";
    topList.innerHTML = "";
    (data.top_predictions || []).forEach(function (item) {
      const li = document.createElement("li");
      li.textContent = item.display_name + " — " + Number(item.confidence).toFixed(2) + "%";
      topList.appendChild(li);
    });
    show(resultCard);
    hide(errorCard);
  }

  dropZone.addEventListener("click", function (event) {
    if (event.target.closest("button") || event.target.closest("label")) return;
    fileInput.click();
  });

  dropZone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach(function (type) {
    dropZone.addEventListener(type, function (event) {
      event.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (type) {
    dropZone.addEventListener(type, function (event) {
      event.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", function (event) {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setFile(file);
  });

  fileInput.addEventListener("change", function () {
    const file = fileInput.files && fileInput.files[0];
    if (file) setFile(file);
  });

  changeBtn.addEventListener("click", function (event) {
    event.stopPropagation();
    fileInput.click();
  });

  removeBtn.addEventListener("click", function (event) {
    event.stopPropagation();
    clearPreview();
  });

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    if (!selectedFile) {
      showError("Please upload a leaf image before running prediction.");
      return;
    }

    hide(resultCard);
    hide(errorCard);
    show(loading);
    predictBtn.disabled = true;

    const body = new FormData();
    body.append("image", selectedFile, selectedFile.name);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: body,
      });

      let data;
      try {
        data = await response.json();
      } catch (parseError) {
        throw new Error("The backend did not return valid JSON. Confirm the Flask server is running.");
      }

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      renderResult(data);
    } catch (error) {
      const message =
        error && error.message
          ? error.message
          : "Could not reach the prediction API. Start the Flask backend and try again.";
      showError(
        message +
          " Predictions are never simulated. If the model file is missing, place plant_disease_mobilenetv2.keras in web_app/models/."
      );
    } finally {
      hide(loading);
      predictBtn.disabled = !selectedFile;
    }
  });

  if (navToggle && siteNav) {
    navToggle.addEventListener("click", function () {
      const open = siteNav.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }
})();
