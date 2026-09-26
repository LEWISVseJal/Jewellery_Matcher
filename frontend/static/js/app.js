/* ============================================================
   JEWELMATCH AI
   JEWELLERY VISUAL SEARCH
   MAIN APPLICATION
============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  console.log("JewelMatch AI app.js loaded");

  /* ========================================================
       ELEMENTS
    ======================================================== */

  const uploadArea = document.getElementById("uploadArea");

  const uploadContent = document.getElementById("uploadContent");

  const imageInput = document.getElementById("imageInput");

  const browseButton = document.getElementById("browseButton");

  const previewContainer = document.getElementById("previewContainer");

  const previewImage = document.getElementById("previewImage");

  const previewWrapper = document.querySelector(".preview-image-wrapper");

  const fileName = document.getElementById("fileName");

  const fileSize = document.getElementById("fileSize");

  const changeImageButton = document.getElementById("changeImageButton");

  const matchButton = document.getElementById("matchButton");

  const matchButtonText = document.getElementById("matchButtonText");

  const matchSpinner = document.getElementById("matchSpinner");

  const errorBox = document.getElementById("error");

  const loadingBox = document.getElementById("loading");

  const resultsSection = document.getElementById("resultsSection");

  const bestMatchBadge = document.getElementById("bestMatchBadge");

  const resultsContainer = document.getElementById("results");

  /* ========================================================
       STATE
    ======================================================== */

  let selectedFile = null;

  let imageZoom = 1;

  let imagePositionX = 0;

  let imagePositionY = 0;

  let isDragging = false;

  let dragStartX = 0;

  let dragStartY = 0;

  let startPositionX = 0;

  let startPositionY = 0;

  /* ========================================================
       CHECK REQUIRED ELEMENTS
    ======================================================== */

  console.log("imageInput:", imageInput);

  console.log("browseButton:", browseButton);

  console.log("uploadArea:", uploadArea);

  if (!imageInput) {
    console.error("ERROR: #imageInput not found");

    return;
  }

  if (!browseButton) {
    console.error("ERROR: #browseButton not found");

    return;
  }

  if (!uploadArea) {
    console.error("ERROR: #uploadArea not found");

    return;
  }

  /* ========================================================
       BROWSE BUTTON
    ======================================================== */

  browseButton.addEventListener("click", function (event) {
    event.preventDefault();

    event.stopPropagation();

    console.log("Browse Image clicked");

    imageInput.click();
  });

  /* ========================================================
       UPLOAD AREA CLICK
    ======================================================== */

  uploadArea.addEventListener("click", function (event) {
    if (event.target === browseButton || browseButton.contains(event.target)) {
      return;
    }

    if (
      changeImageButton &&
      (event.target === changeImageButton ||
        changeImageButton.contains(event.target))
    ) {
      return;
    }

    /*
                When preview is visible,
                clicking the upload area should
                NOT open the file picker.
            */

    if (previewContainer && !previewContainer.classList.contains("hidden")) {
      return;
    }

    console.log("Upload area clicked");

    imageInput.click();
  });

  /* ========================================================
       FILE INPUT
    ======================================================== */

  imageInput.addEventListener("change", function (event) {
    console.log("File input changed");

    const files = event.target.files;

    if (!files || files.length === 0) {
      console.log("No file selected");

      return;
    }

    handleFile(files[0]);
  });

  /* ========================================================
       HANDLE FILE
    ======================================================== */

  function handleFile(file) {
    console.log("Selected file:", file);

    hideError();

    if (!file) {
      return;
    }

    /* ----------------------------------------------------
           FILE TYPE
        ---------------------------------------------------- */

    const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

    if (!allowedTypes.includes(file.type)) {
      showError("Please select a JPG, JPEG, PNG, WEBP, or BMP image.");

      resetSelectedFile();

      return;
    }

    /* ----------------------------------------------------
           FILE SIZE
        ---------------------------------------------------- */

    const maxSize = 20 * 1024 * 1024;

    if (file.size > maxSize) {
      showError("Image size must be less than 20 MB.");

      resetSelectedFile();

      return;
    }

    /* ----------------------------------------------------
           SAVE FILE
        ---------------------------------------------------- */

    selectedFile = file;

    /* ----------------------------------------------------
           RESET IMAGE VIEW
        ---------------------------------------------------- */

    resetImageView();

    /* ----------------------------------------------------
           CREATE PREVIEW
        ---------------------------------------------------- */

    const reader = new FileReader();

    reader.onload = function (event) {
      previewImage.src = event.target.result;

      fileName.textContent = file.name;

      fileSize.textContent = formatFileSize(file.size);

      uploadContent.classList.add("hidden");

      previewContainer.classList.remove("hidden");

      matchButton.disabled = false;

      createZoomControls();

      console.log("Preview displayed");
    };

    reader.onerror = function () {
      showError("Unable to read the selected image.");
    };

    reader.readAsDataURL(file);
  }

  /* ========================================================
       CHANGE IMAGE
    ======================================================== */

  if (changeImageButton) {
    changeImageButton.addEventListener("click", function (event) {
      event.preventDefault();

      event.stopPropagation();

      console.log("Change image clicked");

      imageInput.click();
    });
  }

  /* ========================================================
       DRAG & DROP
    ======================================================== */

  uploadArea.addEventListener("dragover", function (event) {
    event.preventDefault();

    event.stopPropagation();

    uploadArea.classList.add("drag-over");
  });

  uploadArea.addEventListener("dragleave", function (event) {
    event.preventDefault();

    event.stopPropagation();

    uploadArea.classList.remove("drag-over");
  });

  uploadArea.addEventListener("drop", function (event) {
    event.preventDefault();

    event.stopPropagation();

    uploadArea.classList.remove("drag-over");

    const files = event.dataTransfer.files;

    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  /* ========================================================
       MATCH BUTTON
    ======================================================== */

  matchButton.addEventListener("click", async function (event) {
    event.preventDefault();

    if (!selectedFile) {
      showError("Please select an image first.");

      return;
    }

    await findMatches();
  });

  /* ========================================================
       FIND MATCHES
    ======================================================== */

  async function findMatches() {
    hideError();

    matchButton.disabled = true;

    if (matchButtonText) {
      matchButtonText.textContent = "Finding Matches...";
    }

    if (matchSpinner) {
      matchSpinner.classList.remove("hidden");
    }

    if (loadingBox) {
      loadingBox.classList.remove("hidden");
    }

    resultsSection.classList.add("hidden");

    resultsContainer.innerHTML = "";

    try {
      const formData = new FormData();

      formData.append("image", selectedFile);

      console.log("Sending image to /api/match");

      const response = await fetch("/api/match", {
        method: "POST",
        body: formData,
      });

      console.log("Response status:", response.status);

      let data;

      try {
        data = await response.json();
      } catch (jsonError) {
        throw new Error("The server returned an invalid response.");
      }

      console.log("Backend response:", data);

      if (!response.ok) {
        throw new Error(
          data.error || data.message || "Matching request failed.",
        );
      }

      displayResults(data);
    } catch (error) {
      console.error("Matching error:", error);

      showError(error.message || "Something went wrong while finding matches.");
    } finally {
      matchButton.disabled = false;

      if (matchButtonText) {
        matchButtonText.textContent = "Find Similar Jewellery";
      }

      if (matchSpinner) {
        matchSpinner.classList.add("hidden");
      }

      if (loadingBox) {
        loadingBox.classList.add("hidden");
      }
    }
  }

  /* ========================================================
       DISPLAY RESULTS
    ======================================================== */

  function displayResults(data) {
    resultsContainer.innerHTML = "";

    const results = data.results || [];

    console.log("Results received:", results.length);

    if (results.length === 0) {
      resultsSection.classList.remove("hidden");

      bestMatchBadge.textContent = "No matches found";

      resultsContainer.innerHTML = `
                <div class="no-results">

                    <h3>
                        No similar jewellery found
                    </h3>

                    <p>
                        Try another jewellery image
                        with a clear view of the design.
                    </p>

                </div>
            `;

      return;
    }

    resultsSection.classList.remove("hidden");

    bestMatchBadge.textContent = `${results.length} Match${results.length > 1 ? "es" : ""}`;

    /* ====================================================
           RESULT CARDS
        ==================================================== */

    results.forEach((item, index) => {
      const card = document.createElement("div");

      card.className = "result-card";

      /* ------------------------------------------------
                   COLLECTION
                ------------------------------------------------ */

      const collection = item.collection || item.source_collection || "";

      /* ------------------------------------------------
                   IMAGE FILENAME
                ------------------------------------------------ */

      const filename = item.filename || item.image || item.image_name || "";

      /* ------------------------------------------------
                   IMAGE URL
                ------------------------------------------------ */

      let imageUrl = item.image_url;

      if (!imageUrl && collection && filename) {
        imageUrl = `/catalogue/${encodeURIComponent(collection)}/${encodeURIComponent(filename)}`;
      }

      /* ------------------------------------------------
                   SCORE
                ------------------------------------------------ */

      const score = item.similarity ?? item.score ?? item.hybrid_score ?? 0;

      const numericScore = Number(score);

      let percentage;

      if (numericScore <= 1) {
        percentage = (numericScore * 100).toFixed(1);
      } else {
        percentage = numericScore.toFixed(1);
      }

      /* =================================================
                   IMPORTANT DISPLAY NAME LOGIC
                   =================================================

                   Priority:

                   1. display_name
                   2. name
                   3. jewellery_name
                   4. title
                   5. id
                   6. filename
                */

      const displayName =
        item.display_name ||
        item.name ||
        item.jewellery_name ||
        item.title ||
        item.id ||
        filename ||
        "Jewellery";

      /* ------------------------------------------------
                   INTERNAL ID
                   ------------------------------------------------ */

      const jewelleryId = item.jewellery_id || item.id || "";

      console.log(`Result ${index + 1}:`, {
        name: displayName,
        id: jewelleryId,
        image: filename,
        similarity: percentage,
      });

      /* ------------------------------------------------
                   RESULT CARD HTML
                ------------------------------------------------ */

      card.innerHTML = `

                    <div class="result-image-wrapper">

                        <img
                            src="${escapeHtml(imageUrl)}"
                            alt="${escapeHtml(displayName)}"
                            class="result-image"
                        >

                    </div>


                    <div class="result-info">

                        <h3
                            title="${escapeHtml(displayName)}"
                        >
                            ${escapeHtml(displayName)}
                        </h3>


                        <p>
                            Visual similarity:
                            <strong>
                                ${percentage}%
                            </strong>
                        </p>


                        ${
                          collection
                            ? `
                                <span class="collection-badge">
                                    ${escapeHtml(collection)}
                                </span>
                            `
                            : ""
                        }

                    </div>

                `;

      /* ------------------------------------------------
                   IMAGE ERROR
                ------------------------------------------------ */

      const imageElement = card.querySelector(".result-image");

      if (imageElement) {
        imageElement.addEventListener("error", function () {
          console.error("Unable to load result image:", imageUrl);

          this.style.display = "none";
        });
      }

      resultsContainer.appendChild(card);
    });

    /* ========================================================
           SCROLL TO RESULTS
        ======================================================== */

    setTimeout(() => {
      resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 100);
  }

  /* ========================================================
       ZOOM CONTROLS
    ======================================================== */

  function createZoomControls() {
    if (!previewWrapper) {
      return;
    }

    /* ----------------------------------------------------
           REMOVE OLD CONTROLS
        ---------------------------------------------------- */

    const oldControls = previewWrapper.querySelector(".zoom-controls");

    if (oldControls) {
      oldControls.remove();
    }

    const oldHint = previewContainer.querySelector(".zoom-hint");

    if (oldHint) {
      oldHint.remove();
    }

    /* ----------------------------------------------------
           CREATE CONTROL PANEL
        ---------------------------------------------------- */

    const controls = document.createElement("div");

    controls.className = "zoom-controls";

    controls.innerHTML = `

            <button
                type="button"
                class="zoom-button"
                data-zoom="out"
                title="Zoom out"
            >
                −
            </button>


            <button
                type="button"
                class="zoom-button"
                data-zoom="in"
                title="Zoom in"
            >
                +
            </button>


            <button
                type="button"
                class="zoom-button reset"
                data-zoom="reset"
                title="Reset zoom"
            >
                Reset
            </button>

        `;

    previewWrapper.appendChild(controls);

    /* ----------------------------------------------------
           BUTTON EVENTS
        ---------------------------------------------------- */

    const zoomOut = controls.querySelector('[data-zoom="out"]');

    const zoomIn = controls.querySelector('[data-zoom="in"]');

    const resetZoom = controls.querySelector('[data-zoom="reset"]');

    zoomOut.addEventListener("click", function (event) {
      event.preventDefault();

      event.stopPropagation();

      changeZoom(-0.2);
    });

    zoomIn.addEventListener("click", function (event) {
      event.preventDefault();

      event.stopPropagation();

      changeZoom(0.2);
    });

    resetZoom.addEventListener("click", function (event) {
      event.preventDefault();

      event.stopPropagation();

      resetImageView();
    });

    /* ----------------------------------------------------
           ZOOM HINT
        ---------------------------------------------------- */

    const hint = document.createElement("div");

    hint.className = "zoom-hint";

    hint.textContent = "Scroll to zoom • Drag to move • Reset to fit";

    previewContainer.appendChild(hint);

    setupImagePanAndZoom();
  }

  /* ========================================================
       CHANGE ZOOM
    ======================================================== */

  function changeZoom(amount) {
    imageZoom += amount;

    if (imageZoom < 1) {
      imageZoom = 1;
    }

    if (imageZoom > 4) {
      imageZoom = 4;
    }

    applyImageTransform();
  }

  /* ========================================================
       RESET IMAGE VIEW
    ======================================================== */

  function resetImageView() {
    imageZoom = 1;

    imagePositionX = 0;

    imagePositionY = 0;

    applyImageTransform();
  }

  /* ========================================================
       APPLY IMAGE TRANSFORM
    ======================================================== */

  function applyImageTransform() {
    if (!previewImage) {
      return;
    }

    previewImage.style.transform = `translate(${imagePositionX}px, ${imagePositionY}px) scale(${imageZoom})`;
  }

  /* ========================================================
       IMAGE PAN + MOUSE WHEEL ZOOM
    ======================================================== */

  function setupImagePanAndZoom() {
    if (!previewWrapper) {
      return;
    }

    /*
            Prevent duplicate event registration.
        */

    if (previewWrapper.dataset.zoomReady === "true") {
      return;
    }

    previewWrapper.dataset.zoomReady = "true";

    /* ----------------------------------------------------
           MOUSE WHEEL
        ---------------------------------------------------- */

    previewWrapper.addEventListener(
      "wheel",
      function (event) {
        event.preventDefault();

        const direction = event.deltaY < 0 ? 0.2 : -0.2;

        changeZoom(direction);
      },
      {
        passive: false,
      },
    );

    /* ----------------------------------------------------
           MOUSE DOWN
        ---------------------------------------------------- */

    previewWrapper.addEventListener("mousedown", function (event) {
      if (event.target.closest(".zoom-controls")) {
        return;
      }

      if (imageZoom <= 1) {
        return;
      }

      isDragging = true;

      dragStartX = event.clientX;

      dragStartY = event.clientY;

      startPositionX = imagePositionX;

      startPositionY = imagePositionY;

      previewWrapper.style.cursor = "grabbing";
    });

    /* ----------------------------------------------------
           MOUSE MOVE
        ---------------------------------------------------- */

    document.addEventListener("mousemove", function (event) {
      if (!isDragging) {
        return;
      }

      const deltaX = event.clientX - dragStartX;

      const deltaY = event.clientY - dragStartY;

      imagePositionX = startPositionX + deltaX;

      imagePositionY = startPositionY + deltaY;

      applyImageTransform();
    });

    /* ----------------------------------------------------
           MOUSE UP
        ---------------------------------------------------- */

    document.addEventListener("mouseup", function () {
      if (!isDragging) {
        return;
      }

      isDragging = false;

      previewWrapper.style.cursor = imageZoom > 1 ? "grab" : "default";
    });

    /* ----------------------------------------------------
           TOUCH START
        ---------------------------------------------------- */

    previewWrapper.addEventListener(
      "touchstart",
      function (event) {
        if (imageZoom <= 1 || event.touches.length !== 1) {
          return;
        }

        const touch = event.touches[0];

        isDragging = true;

        dragStartX = touch.clientX;

        dragStartY = touch.clientY;

        startPositionX = imagePositionX;

        startPositionY = imagePositionY;
      },
      {
        passive: true,
      },
    );

    /* ----------------------------------------------------
           TOUCH MOVE
        ---------------------------------------------------- */

    previewWrapper.addEventListener(
      "touchmove",
      function (event) {
        if (!isDragging || event.touches.length !== 1) {
          return;
        }

        event.preventDefault();

        const touch = event.touches[0];

        imagePositionX = startPositionX + (touch.clientX - dragStartX);

        imagePositionY = startPositionY + (touch.clientY - dragStartY);

        applyImageTransform();
      },
      {
        passive: false,
      },
    );

    /* ----------------------------------------------------
           TOUCH END
        ---------------------------------------------------- */

    previewWrapper.addEventListener("touchend", function () {
      isDragging = false;
    });
  }

  /* ========================================================
       RESET SELECTED FILE
    ======================================================== */

  function resetSelectedFile() {
    selectedFile = null;

    imageInput.value = "";

    matchButton.disabled = true;

    if (previewImage) {
      previewImage.src = "";
    }

    if (previewContainer) {
      previewContainer.classList.add("hidden");
    }

    if (uploadContent) {
      uploadContent.classList.remove("hidden");
    }

    resetImageView();
  }

  /* ========================================================
       FORMAT FILE SIZE
    ======================================================== */

  function formatFileSize(bytes) {
    if (bytes === 0) {
      return "0 Bytes";
    }

    const units = ["Bytes", "KB", "MB", "GB"];

    const index = Math.floor(Math.log(bytes) / Math.log(1024));

    return (
      parseFloat((bytes / Math.pow(1024, index)).toFixed(2)) +
      " " +
      units[index]
    );
  }

  /* ========================================================
       ESCAPE HTML
    ======================================================== */

  function escapeHtml(value) {
    if (value === null || value === undefined) {
      return "";
    }

    return String(value)
      .replace(/&/g, "&amp;")

      .replace(/</g, "&lt;")

      .replace(/>/g, "&gt;")

      .replace(/"/g, "&quot;")

      .replace(/'/g, "&#039;");
  }

  /* ========================================================
       ERROR
    ======================================================== */

  function showError(message) {
    if (!errorBox) {
      return;
    }

    errorBox.textContent = message;

    errorBox.classList.remove("hidden");
  }

  function hideError() {
    if (!errorBox) {
      return;
    }

    errorBox.textContent = "";

    errorBox.classList.add("hidden");
  }

  /* ========================================================
       INITIAL STATE
    ======================================================== */

  resetImageView();

  console.log("JewelMatch AI initialized successfully");
});
