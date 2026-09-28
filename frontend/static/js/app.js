/* ============================================================
   JEWELMATCH AI
   MAIN SEARCH PAGE
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  console.log("JewelMatch AI - App loaded");

  // ========================================================
  // ELEMENTS
  // ========================================================

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

  // ========================================================
  // SEARCH MODE ELEMENTS
  // ========================================================

  const searchModeInputs = document.querySelectorAll(
    'input[name="searchMode"]',
  );

  // ========================================================
  // STATE
  // ========================================================

  let selectedFile = null;

  let previewObjectUrl = null;

  // ========================================================
  // BASIC HELPERS
  // ========================================================

  function showElement(element) {
    if (element) {
      element.classList.remove("hidden");
    }
  }

  function hideElement(element) {
    if (element) {
      element.classList.add("hidden");
    }
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

  // ========================================================
  // SEARCH MODE
  // ========================================================

  function getSearchMode() {
    for (const input of searchModeInputs) {
      if (input.checked) {
        return input.value;
      }
    }

    return "all";
  }

  function getSearchModeLabel(mode) {
    if (mode === "gold_to_prototype") {
      return "Gold → Prototype";
    }

    if (mode === "prototype_to_gold") {
      return "Prototype → Gold";
    }

    return "All";
  }

  function updateSearchModeUI() {
    const mode = getSearchMode();

    console.log("Search mode:", mode);

    console.log("Search mode label:", getSearchModeLabel(mode));
  }

  searchModeInputs.forEach((input) => {
    input.addEventListener("change", () => {
      updateSearchModeUI();

      hideResults();

      clearError();
    });
  });

  // ========================================================
  // RESULT HELPERS
  // ========================================================

  function getSimilarity(result) {
    if (!result || typeof result !== "object") {
      return 0;
    }

    const values = [
      result.score,
      result.match_score,
      result.final_score,
      result.similarity,
    ];

    for (const value of values) {
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
    return result.collection || result.target_collection || "";
  }

  function getResultType(result) {
    return result.type || result.jewellery_type || result.jewelry_type || "";
  }

  function getResultSubtype(result) {
    return result.subtype || "";
  }

  function getResultImage(result) {
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

    if (
      imageUrl.startsWith("/") ||
      imageUrl.startsWith("http://") ||
      imageUrl.startsWith("https://") ||
      imageUrl.startsWith("data:")
    ) {
      return imageUrl;
    }

    const normalized = imageUrl.replace(/\\/g, "/").replace(/^\/+/, "");

    const filename = normalized.split("/").pop();

    const collection = getResultCollection(result).toLowerCase();

    if (collection === "gold" || collection === "prototype") {
      return (
        `/catalogue-image/` +
        `${collection}/` +
        `${encodeURIComponent(filename)}`
      );
    }

    return `/catalogue-image/gold/` + `${encodeURIComponent(filename)}`;
  }

  // ========================================================
  // IMAGE VALIDATION
  // ========================================================

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

  // ========================================================
  // RESULT DISPLAY
  // ========================================================

  function hideResults() {
    hideElement(resultsSection);

    if (resultsContainer) {
      resultsContainer.innerHTML = "";
    }

    if (bestMatchBadge) {
      bestMatchBadge.textContent = "";
    }
  }

  function displayNoMatch(data) {
    if (resultsContainer) {
      resultsContainer.innerHTML = `
          <div class="no-results">

            <h3>
              No similar jewellery found
            </h3>

            <p>
              ${escapeHtml(
                data?.message ||
                  "No reliable jewellery design match was found.",
              )}
            </p>

          </div>
        `;
    }

    if (bestMatchBadge) {
      bestMatchBadge.textContent = "No match";
    }

    showElement(resultsSection);
  }

  function extractResults(data) {
    if (!data) {
      return [];
    }

    if (Array.isArray(data)) {
      return data;
    }

    if (Array.isArray(data.results)) {
      return data.results;
    }

    if (Array.isArray(data.matches)) {
      return data.matches;
    }

    return [];
  }

  function displayResults(data) {
    /*
     * IMPORTANT:
     * Never display candidates if
     * backend says matched=false.
     */

    if (data && data.matched === false) {
      displayNoMatch(data);

      return;
    }

    const results = extractResults(data);

    if (!results.length) {
      displayNoMatch(data);

      return;
    }

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
      resultsContainer.appendChild(createResultCard(result, index));
    });

    showElement(resultsSection);

    setTimeout(() => {
      resultsSection?.scrollIntoView({
        behavior: "smooth",

        block: "start",
      });
    }, 100);
  }

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
              onerror="
                this.style.display='none';
                this.parentElement.classList.add(
                  'image-missing'
                );
              "
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
              ${type}
              ${subtype ? ` · ${subtype}` : ""}
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

          <h3
            title="${name}"
          >
            ${name}
          </h3>

          ${typeHtml}

          ${collectionHtml}

        </div>
      `;

    if (index === 0) {
      card.classList.add("best-result");
    }

    return card;
  }

  // ========================================================
  // FILE SELECTION
  // ========================================================

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

    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }

    previewObjectUrl = URL.createObjectURL(file);

    if (previewImage) {
      previewImage.src = previewObjectUrl;
    }

    if (fileName) {
      fileName.textContent = file.name;
    }

    if (fileSize) {
      fileSize.textContent = formatFileSize(file.size);
    }

    hideElement(uploadContent);

    showElement(previewContainer);

    if (matchButton) {
      matchButton.disabled = false;
    }

    hideResults();
  }

  // ========================================================
  // UPLOAD BUTTON
  // ========================================================

  if (browseButton) {
    browseButton.addEventListener("click", (event) => {
      event.preventDefault();

      event.stopPropagation();

      imageInput.click();
    });
  }

  // ========================================================
  // UPLOAD AREA
  // ========================================================

  if (uploadArea) {
    uploadArea.addEventListener("click", (event) => {
      if (
        event.target === browseButton ||
        browseButton?.contains(event.target)
      ) {
        return;
      }

      if (!selectedFile) {
        imageInput.click();
      }
    });

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

  // ========================================================
  // FILE INPUT
  // ========================================================

  if (imageInput) {
    imageInput.addEventListener("change", (event) => {
      handleSelectedFile(event.target.files?.[0]);
    });
  }

  // ========================================================
  // CHANGE IMAGE
  // ========================================================

  if (changeImageButton) {
    changeImageButton.addEventListener("click", (event) => {
      event.preventDefault();

      event.stopPropagation();

      imageInput.click();
    });
  }

  // ========================================================
  // LOADING STATE
  // ========================================================

  function setLoadingState(isLoading) {
    if (!matchButton) {
      return;
    }

    matchButton.disabled = isLoading || !selectedFile;

    if (isLoading) {
      if (matchButtonText) {
        matchButtonText.textContent = "Finding similar jewellery...";
      }

      showElement(matchSpinner);

      showElement(loadingBox);
    } else {
      if (matchButtonText) {
        matchButtonText.textContent = "Find Similar Jewellery";
      }

      hideElement(matchSpinner);

      hideElement(loadingBox);
    }
  }

  // ========================================================
  // API
  // ========================================================

  async function submitMatchRequest(formData) {
    let response;

    try {
      response = await fetch("/api/match", {
        method: "POST",

        body: formData,
      });
    } catch (error) {
      throw new Error("Could not connect to the server. Please try again.");
    }

    const contentType = response.headers.get("content-type") || "";

    if (contentType.toLowerCase().includes("application/json")) {
      let data;

      try {
        data = await response.json();
      } catch (error) {
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

    const responseText = await response.text();

    const cleanText = responseText
      .replace(/<[^>]*>/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    throw new Error(
      cleanText
        ? `Server error (HTTP ${response.status}): ${cleanText.substring(
            0,
            300,
          )}`
        : `Server error (HTTP ${response.status}).`,
    );
  }

  // ========================================================
  // MATCH BUTTON
  // ========================================================

  if (matchButton) {
    matchButton.addEventListener("click", async (event) => {
      event.preventDefault();

      clearError();

      if (!selectedFile) {
        showError("Please upload a jewellery image first.");

        return;
      }

      const validation = validateImage(selectedFile);

      if (!validation.valid) {
        showError(validation.message);

        return;
      }

      hideResults();

      // --------------------------------------------------
      // SELECTED SEARCH MODE
      // --------------------------------------------------

      const searchMode = getSearchMode();

      console.log("================================");

      console.log("JEWELLERY SEARCH");

      console.log("Search mode:", searchMode);

      console.log("Mode:", getSearchModeLabel(searchMode));

      console.log("================================");

      // --------------------------------------------------
      // FORM DATA
      // --------------------------------------------------

      const formData = new FormData();

      formData.append("image", selectedFile, selectedFile.name);

      formData.append("search_mode", searchMode);

      setLoadingState(true);

      try {
        const data = await submitMatchRequest(formData);

        console.log("Match API response:", data);

        displayResults(data);
      } catch (error) {
        console.error("Jewellery matching error:", error);

        showError(error?.message || "Unable to perform visual search.");
      } finally {
        setLoadingState(false);
      }
    });
  }

  // ========================================================
  // INITIAL STATE
  // ========================================================

  if (matchButton) {
    matchButton.disabled = true;
  }

  hideElement(loadingBox);

  hideElement(errorBox);

  hideElement(resultsSection);

  updateSearchModeUI();

  console.log("JewelMatch AI - Ready");
});
