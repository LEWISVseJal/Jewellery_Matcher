/* ============================================================
   JEWELMATCH AI
   MAIN PAGE LOGIC
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  console.log("JewelMatch AI - App loaded");

  /* ========================================================
       ELEMENTS
    ======================================================== */

  const uploadArea = document.getElementById("uploadArea");
  const uploadContent = document.getElementById("uploadContent");
  const imageInput = document.getElementById("imageInput");
  const browseButton = document.getElementById("browseButton");

  const previewContainer = document.getElementById("previewContainer");

  const previewImage = document.getElementById("previewImage");

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

  /* ========================================================
       SAFETY CHECK
    ======================================================== */

  if (!imageInput) {
    console.error("JewelMatch: imageInput element not found.");
    return;
  }

  /* ========================================================
       HELPERS
    ======================================================== */

  function showElement(element) {
    if (!element) {
      return;
    }

    element.classList.remove("hidden");
  }

  function hideElement(element) {
    if (!element) {
      return;
    }

    element.classList.add("hidden");
  }

  function showError(message) {
    if (!errorBox) {
      console.error(message);
      return;
    }

    errorBox.textContent = message;

    showElement(errorBox);
  }

  function clearError() {
    if (!errorBox) {
      return;
    }

    errorBox.textContent = "";

    hideElement(errorBox);
  }

  function formatFileSize(bytes) {
    if (!bytes || bytes <= 0) {
      return "0 KB";
    }

    const units = ["Bytes", "KB", "MB", "GB"];

    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
  }

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

  function getSimilarity(result) {
    if (!result || typeof result !== "object") {
      return 0;
    }

    const possibleValues = [
      result.similarity,
      result.score,
      result.final_score,
    ];

    for (const value of possibleValues) {
      const number = Number(value);

      if (Number.isFinite(number)) {
        return number;
      }
    }

    return 0;
  }

  function similarityPercentage(value) {
    let score = Number(value);

    if (!Number.isFinite(score)) {
      return 0;
    }

    /*
     * Most backend responses return similarity
     * between 0 and 1.
     *
     * If backend already sends 0-100,
     * keep it unchanged.
     */

    if (score <= 1) {
      score *= 100;
    }

    return Math.max(0, Math.min(100, score));
  }

  function getResultName(result) {
    return (
      result.name ||
      result.jewellery_name ||
      result.jewelry_name ||
      result.title ||
      "Jewellery"
    );
  }

  function getResultCollection(result) {
    return (
      result.collection ||
      result.source_collection ||
      result.target_collection ||
      ""
    );
  }

  function getResultType(result) {
    return result.type || result.jewellery_type || result.jewelry_type || "";
  }

  function getResultSubtype(result) {
    return result.subtype || "";
  }

  function getResultImage(result) {
    /*
     * Backend may return any of these names.
     */

    const imageUrl =
      result.image_url ||
      result.image ||
      result.image_path ||
      result.path ||
      result.url ||
      "";

    if (!imageUrl) {
      return "";
    }

    /*
     * Already a web URL.
     */

    if (
      imageUrl.startsWith("/") ||
      imageUrl.startsWith("http://") ||
      imageUrl.startsWith("https://") ||
      imageUrl.startsWith("data:")
    ) {
      return imageUrl;
    }

    /*
     * Windows/local paths should not be used directly
     * in the browser. Try the catalogue endpoint.
     */

    const normalized = imageUrl.replace(/\\/g, "/").replace(/^\/+/, "");

    const parts = normalized.split("/");

    const filename = parts[parts.length - 1];

    /*
     * If collection exists, use the backend
     * catalogue image endpoint.
     */

    const collection = getResultCollection(result).toLowerCase();

    if (collection === "gold" || collection === "prototype") {
      return `/catalogue/${collection}/${encodeURIComponent(filename)}`;
    }

    /*
     * Fallback.
     */

    return `/catalogue/gold/${encodeURIComponent(filename)}`;
  }

  /* ========================================================
       IMAGE VALIDATION
    ======================================================== */

  function validateImage(file) {
    if (!file) {
      return {
        valid: false,
        message: "Please select an image.",
      };
    }

    const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

    if (file.type && !allowedTypes.includes(file.type)) {
      return {
        valid: false,
        message: "Unsupported image format. Please use JPG, PNG, WEBP or BMP.",
      };
    }

    /*
     * Keep frontend aligned with the current
     * 20 MB project limit.
     */

    const maxSize = 20 * 1024 * 1024;

    if (file.size > maxSize) {
      return {
        valid: false,
        message: "Image is too large. Maximum allowed size is 20 MB.",
      };
    }

    return {
      valid: true,
      message: "",
    };
  }

  /* ========================================================
       SHOW SELECTED IMAGE
    ======================================================== */

  function handleSelectedFile(file) {
    clearError();

    if (!file) {
      return;
    }

    const validation = validateImage(file);

    if (!validation.valid) {
      selectedFile = null;

      showError(validation.message);

      return;
    }

    selectedFile = file;

    /*
     * Create browser preview.
     */

    const objectUrl = URL.createObjectURL(file);

    if (previewImage) {
      previewImage.onload = () => {
        URL.revokeObjectURL(objectUrl);
      };

      previewImage.src = objectUrl;
    }

    /*
     * File information.
     */

    if (fileName) {
      fileName.textContent = file.name;
    }

    if (fileSize) {
      fileSize.textContent = formatFileSize(file.size);
    }

    /*
     * Switch upload area to preview.
     */

    if (uploadContent) {
      hideElement(uploadContent);
    }

    if (previewContainer) {
      showElement(previewContainer);
    }

    /*
     * Enable match button.
     */

    if (matchButton) {
      matchButton.disabled = false;
    }

    /*
     * Hide old results.
     */

    hideResults();
  }

  /* ========================================================
       RESET IMAGE
    ======================================================== */

  function resetImage() {
    selectedFile = null;

    if (imageInput) {
      imageInput.value = "";
    }

    if (previewImage) {
      previewImage.src = "";
    }

    if (fileName) {
      fileName.textContent = "";
    }

    if (fileSize) {
      fileSize.textContent = "";
    }

    if (previewContainer) {
      hideElement(previewContainer);
    }

    if (uploadContent) {
      showElement(uploadContent);
    }

    if (matchButton) {
      matchButton.disabled = true;
    }

    hideResults();

    clearError();
  }

  /* ========================================================
       BROWSE BUTTON
    ======================================================== */

  if (browseButton) {
    browseButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();

      imageInput.click();
    });
  }

  /* ========================================================
       UPLOAD AREA CLICK
    ======================================================== */

  if (uploadArea) {
    uploadArea.addEventListener("click", (event) => {
      /*
       * Do not trigger another file dialog when
       * clicking the browse button itself.
       */

      if (
        event.target === browseButton ||
        browseButton?.contains(event.target)
      ) {
        return;
      }

      /*
       * When an image is already selected,
       * clicking the area should not unexpectedly
       * replace it.
       */

      if (!selectedFile) {
        imageInput.click();
      }
    });
  }

  /* ========================================================
       FILE INPUT CHANGE
    ======================================================== */

  imageInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];

    handleSelectedFile(file);
  });

  /* ========================================================
       CHANGE IMAGE
    ======================================================== */

  if (changeImageButton) {
    changeImageButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();

      imageInput.click();
    });
  }

  /* ========================================================
       DRAG OVER
    ======================================================== */

  if (uploadArea) {
    uploadArea.addEventListener("dragover", (event) => {
      event.preventDefault();

      uploadArea.classList.add("drag-over");
    });

    uploadArea.addEventListener("dragleave", (event) => {
      event.preventDefault();

      uploadArea.classList.remove("drag-over");
    });

    uploadArea.addEventListener("drop", (event) => {
      event.preventDefault();

      uploadArea.classList.remove("drag-over");

      const file = event.dataTransfer?.files?.[0];

      handleSelectedFile(file);
    });
  }

  /* ========================================================
       HIDE RESULTS
    ======================================================== */

  function hideResults() {
    if (resultsSection) {
      hideElement(resultsSection);
    }

    if (resultsContainer) {
      resultsContainer.innerHTML = "";
    }

    if (bestMatchBadge) {
      bestMatchBadge.textContent = "";
    }
  }

  /* ========================================================
       LOADING STATE
    ======================================================== */

  function setLoadingState(isLoading) {
    if (!matchButton) {
      return;
    }

    if (isLoading) {
      matchButton.disabled = true;

      if (matchButtonText) {
        matchButtonText.textContent = "Finding similar jewellery...";
      }

      if (matchSpinner) {
        showElement(matchSpinner);
      }

      if (loadingBox) {
        showElement(loadingBox);
      }
    } else {
      matchButton.disabled = !selectedFile;

      if (matchButtonText) {
        matchButtonText.textContent = "Find Similar Jewellery";
      }

      if (matchSpinner) {
        hideElement(matchSpinner);
      }

      if (loadingBox) {
        hideElement(loadingBox);
      }
    }
  }

  /* ========================================================
       FETCH MATCH API
    ======================================================== */

  async function submitMatchRequest(formData) {
    let response;

    try {
      response = await fetch("/api/match", {
        method: "POST",
        body: formData,
      });
    } catch (networkError) {
      throw new Error("Could not connect to the server. Please try again.");
    }

    /*
     * Read response according to its content type.
     * This is important because Render may return an
     * HTML error page instead of JSON.
     */

    const contentType = response.headers.get("content-type") || "";

    if (contentType.toLowerCase().includes("application/json")) {
      let data;

      try {
        data = await response.json();
      } catch (jsonError) {
        throw new Error(
          `Server returned invalid JSON (HTTP ${response.status}).`,
        );
      }

      if (!response.ok) {
        throw new Error(
          data.error ||
            data.message ||
            `Server error (HTTP ${response.status}).`,
        );
      }

      return data;
    }

    /*
     * Non-JSON response.
     */

    const responseText = await response.text();

    const cleanText = responseText
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    if (!response.ok) {
      throw new Error(
        cleanText
          ? `Server error (HTTP ${response.status}): ${cleanText.substring(0, 300)}`
          : `Server error (HTTP ${response.status}).`,
      );
    }

    throw new Error(
      cleanText
        ? `The server returned an invalid response: ${cleanText.substring(0, 300)}`
        : "The server returned an invalid response.",
    );
  }

  /* ========================================================
       EXTRACT RESULTS FROM API RESPONSE
    ======================================================== */

  function extractResults(data) {
    if (!data) {
      return [];
    }

    /*
     * Common response formats.
     */

    if (Array.isArray(data)) {
      return data;
    }

    if (Array.isArray(data.results)) {
      return data.results;
    }

    if (Array.isArray(data.matches)) {
      return data.matches;
    }

    if (data.data && Array.isArray(data.data.results)) {
      return data.data.results;
    }

    if (data.data && Array.isArray(data.data.matches)) {
      return data.data.matches;
    }

    return [];
  }

  /* ========================================================
       DISPLAY RESULTS
    ======================================================== */

  function displayResults(data) {
    const results = extractResults(data);

    if (!results.length) {
      if (resultsContainer) {
        resultsContainer.innerHTML = `
                    <div class="no-results">
                        <h3>No similar jewellery found</h3>
                        <p>
                            No sufficiently similar jewellery
                            was found in the catalogue.
                        </p>
                    </div>
                `;
      }

      if (resultsSection) {
        showElement(resultsSection);
      }

      if (bestMatchBadge) {
        bestMatchBadge.textContent = "No match";
      }

      return;
    }

    /*
     * Sort highest similarity first.
     */

    results.sort((a, b) => getSimilarity(b) - getSimilarity(a));

    const bestSimilarity = similarityPercentage(getSimilarity(results[0]));

    if (bestMatchBadge) {
      bestMatchBadge.textContent = `Best match ${bestSimilarity.toFixed(1)}%`;
    }

    if (!resultsContainer) {
      return;
    }

    resultsContainer.innerHTML = "";

    results.forEach((result, index) => {
      const card = createResultCard(result, index);

      resultsContainer.appendChild(card);
    });

    if (resultsSection) {
      showElement(resultsSection);
    }

    /*
     * Scroll to results after successful search.
     */

    setTimeout(() => {
      resultsSection?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 100);
  }

  /* ========================================================
       CREATE RESULT CARD
    ======================================================== */

  function createResultCard(result, index) {
    const card = document.createElement("article");

    card.className = "result-card";

    const name = escapeHtml(getResultName(result));

    const collection = escapeHtml(getResultCollection(result));

    const type = escapeHtml(getResultType(result));

    const subtype = escapeHtml(getResultSubtype(result));

    const similarity = similarityPercentage(getSimilarity(result));

    const imageUrl = getResultImage(result);

    const imageHtml = imageUrl
      ? `
                    <img
                        src="${escapeHtml(imageUrl)}"
                        alt="${name}"
                        class="result-image"
                        loading="lazy"
                        onerror="this.style.display='none'; this.parentElement.classList.add('image-missing');"
                    >
                `
      : `
                    <div class="image-missing">
                        <span>✦</span>
                    </div>
                `;

    const collectionHtml = collection
      ? `
                    <span class="collection-badge">
                        ${collection}
                    </span>
                `
      : "";

    const typeHtml = type
      ? `
                    <p>
                        ${type}${subtype ? ` · ${subtype}` : ""}
                    </p>
                `
      : "";

    card.innerHTML = `

            <div class="result-image-wrapper">

                ${imageHtml}

                <div class="result-score">
                    ${similarity.toFixed(1)}%
                </div>

            </div>


            <div class="result-info">

                <h3 title="${name}">
                    ${name}
                </h3>

                ${typeHtml}

                ${collectionHtml}

            </div>
        `;

    /*
     * First result gets a subtle visual indicator.
     */

    if (index === 0) {
      card.classList.add("best-result");
    }

    return card;
  }

  /* ========================================================
       MATCH BUTTON
    ======================================================== */

  if (matchButton) {
    matchButton.addEventListener("click", async (event) => {
      event.preventDefault();

      clearError();

      /*
       * Validate selected image.
       */

      if (!selectedFile) {
        showError("Please upload a jewellery image first.");

        return;
      }

      const validation = validateImage(selectedFile);

      if (!validation.valid) {
        showError(validation.message);

        return;
      }

      /*
       * Clear previous results.
       */

      hideResults();

      /*
       * Build multipart form.
       *
       * Backend expects:
       * image
       */

      const formData = new FormData();

      formData.append("image", selectedFile, selectedFile.name);

      setLoadingState(true);

      try {
        console.log("Sending jewellery image to /api/match...");

        const data = await submitMatchRequest(formData);

        console.log("Match API response:", data);

        /*
         * Backend may explicitly report an error
         * while still returning HTTP 200.
         */

        if (data && data.error) {
          throw new Error(data.error);
        }

        displayResults(data);
      } catch (error) {
        console.error("Jewellery matching error:", error);

        let message = error?.message || "Unable to perform visual search.";

        /*
         * Make common errors user-friendly.
         */

        if (message.includes("Failed to fetch")) {
          message = "Unable to connect to the server. Please try again.";
        }

        showError(message);
      } finally {
        setLoadingState(false);
      }
    });
  }

  /* ========================================================
       INITIAL STATE
    ======================================================== */

  if (matchButton) {
    matchButton.disabled = true;
  }

  if (loadingBox) {
    hideElement(loadingBox);
  }

  if (errorBox) {
    hideElement(errorBox);
  }

  if (resultsSection) {
    hideElement(resultsSection);
  }

  console.log("JewelMatch AI - Ready");
});
