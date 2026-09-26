// ============================================================
// JEWELMATCH AI
// CATALOGUE MANAGEMENT
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  console.log("catalogue.js loaded");

  // ========================================================
  // STATE
  // ========================================================

  let catalogue = [];

  let filteredCatalogue = [];

  let activeCollection = "all";

  let searchTerm = "";

  let currentSort = "newest";

  let currentView = "grid";

  let editingItemId = null;

  let deletingItemId = null;

  // ========================================================
  // ELEMENTS
  // ========================================================

  const catalogueGrid = document.getElementById("catalogueGrid");

  const catalogueTableWrapper = document.getElementById(
    "catalogueTableWrapper",
  );

  const catalogueTableBody = document.getElementById("catalogueTableBody");

  const loading = document.getElementById("loading");

  const emptyState = document.getElementById("emptyState");

  const searchInput = document.getElementById("searchInput");

  const clearSearchButton = document.getElementById("clearSearchButton");

  const sortSelect = document.getElementById("sortSelect");

  const refreshButton = document.getElementById("refreshButton");

  const rebuildButton = document.getElementById("rebuildButton");

  const gridViewButton = document.getElementById("gridViewButton");

  const tableViewButton = document.getElementById("tableViewButton");

  const totalCount = document.getElementById("totalCount");

  const goldCount = document.getElementById("goldCount");

  const prototypeCount = document.getElementById("prototypeCount");

  const typeCount = document.getElementById("typeCount");

  const resultCount = document.getElementById("resultCount");

  const activeFilterLabel = document.getElementById("activeFilterLabel");

  const pageStatus = document.getElementById("pageStatus");

  // EDIT MODAL

  const editModal = document.getElementById("editModal");

  const editForm = document.getElementById("editForm");

  const closeModal = document.getElementById("closeModal");

  const cancelEdit = document.getElementById("cancelEdit");

  const modalOverlay = document.getElementById("modalOverlay");

  const editId = document.getElementById("editId");

  const editIdDisplay = document.getElementById("editIdDisplay");

  const editName = document.getElementById("editName");

  const editCollection = document.getElementById("editCollection");

  const editGender = document.getElementById("editGender");

  const editType = document.getElementById("editType");

  const editSubtype = document.getElementById("editSubtype");

  const editDescription = document.getElementById("editDescription");

  const editImage = document.getElementById("editImage");

  const editPreviewImage = document.getElementById("editPreviewImage");

  const editStatus = document.getElementById("editStatus");

  // DELETE MODAL

  const deleteModal = document.getElementById("deleteModal");

  const deleteModalOverlay = document.getElementById("deleteModalOverlay");

  const deleteItemName = document.getElementById("deleteItemName");

  const cancelDelete = document.getElementById("cancelDelete");

  const confirmDelete = document.getElementById("confirmDelete");

  const deleteStatus = document.getElementById("deleteStatus");

  // IMAGE MODAL

  const imageModal = document.getElementById("imageModal");

  const imageModalOverlay = document.getElementById("imageModalOverlay");

  const closeImageModal = document.getElementById("closeImageModal");

  const largePreviewImage = document.getElementById("largePreviewImage");

  const largePreviewName = document.getElementById("largePreviewName");

  // ========================================================
  // INITIAL LOAD
  // ========================================================

  loadCatalogue();

  // ========================================================
  // LOAD CATALOGUE
  // ========================================================

  async function loadCatalogue() {
    showLoading();

    try {
      const response = await fetch("/api/jewellery", {
        method: "GET",
        cache: "no-store",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Unable to load catalogue.");
      }

      if (Array.isArray(data)) {
        catalogue = data;
      } else if (data && Array.isArray(data.items)) {
        catalogue = data.items;
      } else {
        catalogue = [];
      }

      console.log("Catalogue loaded:", catalogue.length);

      updateStatistics();

      applyFilters();

      hideLoading();
    } catch (error) {
      console.error("Catalogue loading error:", error);

      catalogue = [];

      filteredCatalogue = [];

      hideLoading();

      showPageStatus(error.message || "Unable to load catalogue.", "error");

      renderCatalogue();
    }
  }

  // ========================================================
  // STATISTICS
  // ========================================================

  function updateStatistics() {
    const total = catalogue.length;

    const gold = catalogue.filter(
      (item) => String(item.collection || "").toLowerCase() === "gold",
    ).length;

    const prototype = catalogue.filter(
      (item) => String(item.collection || "").toLowerCase() === "prototype",
    ).length;

    const types = new Set(
      catalogue
        .map((item) =>
          String(item.type || item.category || "")
            .trim()
            .toLowerCase(),
        )
        .filter(Boolean),
    ).size;

    if (totalCount) {
      totalCount.textContent = total;
    }

    if (goldCount) {
      goldCount.textContent = gold;
    }

    if (prototypeCount) {
      prototypeCount.textContent = prototype;
    }

    if (typeCount) {
      typeCount.textContent = types;
    }
  }

  // ========================================================
  // FILTER + SEARCH
  // ========================================================

  function applyFilters() {
    let results = [...catalogue];

    // ----------------------------------------------------
    // COLLECTION
    // ----------------------------------------------------

    if (activeCollection !== "all") {
      results = results.filter(
        (item) =>
          String(item.collection || "").toLowerCase() === activeCollection,
      );
    }

    // ----------------------------------------------------
    // SEARCH
    // ----------------------------------------------------

    if (searchTerm) {
      const search = searchTerm.toLowerCase();

      results = results.filter((item) => {
        const values = [
          item.id,

          item.name,

          item.type,

          item.category,

          item.subtype,

          item.gender,

          item.description,

          item.collection,
        ];

        return values.some((value) =>
          String(value || "")
            .toLowerCase()
            .includes(search),
        );
      });
    }

    // ----------------------------------------------------
    // SORT
    // ----------------------------------------------------

    results.sort(sortItems);

    filteredCatalogue = results;

    renderCatalogue();
  }

  // ========================================================
  // SORT
  // ========================================================

  function sortItems(a, b) {
    switch (currentSort) {
      case "name":
        return String(a.name || "").localeCompare(String(b.name || ""));

      case "id":
        return compareIds(a.id, b.id);

      case "type":
        return String(a.type || a.category || "").localeCompare(
          String(b.type || b.category || ""),
        );

      case "oldest":
        return compareIds(a.id, b.id);

      case "newest":

      default:
        return compareIds(b.id, a.id);
    }
  }

  // ========================================================
  // COMPARE JEWELLERY IDs
  // ========================================================

  function compareIds(first, second) {
    const firstNumber = extractIdNumber(first);

    const secondNumber = extractIdNumber(second);

    return firstNumber - secondNumber;
  }

  function extractIdNumber(id) {
    const match = String(id || "").match(/(\d+)/);

    if (!match) {
      return 0;
    }

    return Number(match[1]);
  }

  // ========================================================
  // RENDER CATALOGUE
  // ========================================================

  function renderCatalogue() {
    if (currentView === "table") {
      renderTable();
    } else {
      renderGrid();
    }

    updateResultSummary();
  }

  // ========================================================
  // GRID
  // ========================================================

  function renderGrid() {
    if (!catalogueGrid) {
      return;
    }

    catalogueGrid.innerHTML = "";

    if (filteredCatalogue.length === 0) {
      catalogueGrid.classList.add("hidden");

      if (catalogueTableWrapper) {
        catalogueTableWrapper.classList.add("hidden");
      }

      showEmptyState();

      return;
    }

    hideEmptyState();

    catalogueGrid.classList.remove("hidden");

    if (catalogueTableWrapper) {
      catalogueTableWrapper.classList.add("hidden");
    }

    filteredCatalogue.forEach((item) => {
      const card = createCatalogueCard(item);

      catalogueGrid.appendChild(card);
    });
  }

  // ========================================================
  // CREATE CARD
  // ========================================================

  function createCatalogueCard(item) {
    const card = document.createElement("article");

    card.className = "catalogue-card";

    const collection = String(item.collection || "").toLowerCase();

    const collectionLabel = collection === "prototype" ? "Prototype" : "Gold";

    const filename = item.image || item.filename || "";

    const imageUrl = item.image_url || buildImageUrl(collection, filename);

    const name = item.name || item.design_id || "Unnamed Jewellery";

    const id = item.id || "—";

    const gender = item.gender || "—";

    const type = item.type || item.category || "—";

    const subtype = item.subtype || "—";

    const description = item.description || "No description available.";

    card.innerHTML = `

            <div class="catalogue-card-image">

                ${
                  imageUrl
                    ? `
                        <img
                            src="${escapeAttribute(imageUrl)}"
                            alt="${escapeAttribute(name)}"
                            loading="lazy"
                        >
                    `
                    : `
                        <div class="image-placeholder">
                            ✦
                        </div>
                    `
                }


                <span
                    class="collection-badge ${escapeAttribute(collection)}"
                >
                    ${escapeHtml(collectionLabel)}
                </span>


                <button
                    class="image-expand-button"
                    type="button"
                    data-action="preview"
                    data-id="${escapeAttribute(id)}"
                    title="View image"
                >
                    ⤢
                </button>

            </div>


            <div class="catalogue-card-body">


                <div class="card-top-line">

                    <span class="jewellery-id">
                        ${escapeHtml(id)}
                    </span>

                </div>


                <h3>
                    ${escapeHtml(name)}
                </h3>


                <div class="card-meta">

                    <span>
                        ${escapeHtml(type)}
                    </span>

                    <span>
                        ${escapeHtml(subtype)}
                    </span>

                    <span>
                        ${escapeHtml(gender)}
                    </span>

                </div>


                <p class="card-description">
                    ${escapeHtml(truncateText(description, 110))}
                </p>


                <div class="card-actions">

                    <button
                        type="button"
                        class="card-button edit"
                        data-action="edit"
                        data-id="${escapeAttribute(id)}"
                    >
                        Edit
                    </button>


                    <button
                        type="button"
                        class="card-button delete"
                        data-action="delete"
                        data-id="${escapeAttribute(id)}"
                    >
                        Delete
                    </button>

                </div>

            </div>
        `;

    return card;
  }

  // ========================================================
  // TABLE
  // ========================================================

  function renderTable() {
    if (!catalogueTableWrapper) {
      return;
    }

    if (catalogueGrid) {
      catalogueGrid.classList.add("hidden");
    }

    if (filteredCatalogue.length === 0) {
      catalogueTableWrapper.classList.add("hidden");

      showEmptyState();

      return;
    }

    hideEmptyState();

    catalogueTableWrapper.classList.remove("hidden");

    if (!catalogueTableBody) {
      return;
    }

    catalogueTableBody.innerHTML = "";

    filteredCatalogue.forEach((item) => {
      const row = createTableRow(item);

      catalogueTableBody.appendChild(row);
    });
  }

  // ========================================================
  // TABLE ROW
  // ========================================================

  function createTableRow(item) {
    const row = document.createElement("tr");

    const collection = String(item.collection || "").toLowerCase();

    const filename = item.image || item.filename || "";

    const imageUrl = item.image_url || buildImageUrl(collection, filename);

    const name = item.name || "Unnamed Jewellery";

    const id = item.id || "—";

    const gender = item.gender || "—";

    const type = item.type || item.category || "—";

    const subtype = item.subtype || "—";

    const description = item.description || "—";

    row.innerHTML = `

            <td>

                ${
                  imageUrl
                    ? `
                    <img
                        class="table-image"
                        src="${escapeAttribute(imageUrl)}"
                        alt="${escapeAttribute(name)}"
                        data-action="preview"
                        data-id="${escapeAttribute(id)}"
                    >
                    `
                    : `
                    <div class="table-image-placeholder">
                        ✦
                    </div>
                    `
                }

            </td>


            <td>
                <strong>
                    ${escapeHtml(id)}
                </strong>
            </td>


            <td>

                <div class="table-name">
                    ${escapeHtml(name)}
                </div>

            </td>


            <td>

                <span
                    class="collection-badge ${escapeAttribute(collection)}"
                >
                    ${escapeHtml(
                      collection === "prototype" ? "Prototype" : "Gold",
                    )}
                </span>

            </td>


            <td>
                ${escapeHtml(gender)}
            </td>


            <td>
                ${escapeHtml(type)}
            </td>


            <td>
                ${escapeHtml(subtype)}
            </td>


            <td>

                <div class="table-description">
                    ${escapeHtml(truncateText(description, 80))}
                </div>

            </td>


            <td>

                <div class="table-actions">

                    <button
                        type="button"
                        class="table-action-button"
                        data-action="edit"
                        data-id="${escapeAttribute(id)}"
                    >
                        Edit
                    </button>


                    <button
                        type="button"
                        class="table-action-button delete"
                        data-action="delete"
                        data-id="${escapeAttribute(id)}"
                    >
                        Delete
                    </button>

                </div>

            </td>

        `;

    return row;
  }

  // ========================================================
  // RESULT SUMMARY
  // ========================================================

  function updateResultSummary() {
    const count = filteredCatalogue.length;

    if (resultCount) {
      resultCount.textContent = `${count} ${
        count === 1 ? "jewellery item" : "jewellery items"
      }`;
    }

    if (activeFilterLabel) {
      let label = "Showing all jewellery";

      if (activeCollection === "gold") {
        label = "Showing Gold jewellery";
      } else if (activeCollection === "prototype") {
        label = "Showing Prototype jewellery";
      }

      if (searchTerm) {
        label += ` matching "${searchTerm}"`;
      }

      activeFilterLabel.textContent = label;
    }

    if (clearSearchButton) {
      clearSearchButton.classList.toggle("hidden", !searchTerm);
    }
  }

  // ========================================================
  // COLLECTION FILTERS
  // ========================================================

  document.querySelectorAll(".filter-button").forEach((button) => {
    button.addEventListener("click", () => {
      document
        .querySelectorAll(".filter-button")
        .forEach((item) => item.classList.remove("active"));

      button.classList.add("active");

      activeCollection = button.dataset.collection || "all";

      applyFilters();
    });
  });

  // ========================================================
  // SEARCH
  // ========================================================

  if (searchInput) {
    searchInput.addEventListener("input", (event) => {
      searchTerm = event.target.value.trim().toLowerCase();

      applyFilters();
    });
  }

  if (clearSearchButton) {
    clearSearchButton.addEventListener("click", () => {
      searchTerm = "";

      searchInput.value = "";

      applyFilters();

      searchInput.focus();
    });
  }

  // ========================================================
  // SORT
  // ========================================================

  if (sortSelect) {
    sortSelect.addEventListener("change", (event) => {
      currentSort = event.target.value;

      applyFilters();
    });
  }

  // ========================================================
  // VIEW SWITCH
  // ========================================================

  if (gridViewButton) {
    gridViewButton.addEventListener("click", () => {
      currentView = "grid";

      gridViewButton.classList.add("active");

      if (tableViewButton) {
        tableViewButton.classList.remove("active");
      }

      renderCatalogue();
    });
  }

  if (tableViewButton) {
    tableViewButton.addEventListener("click", () => {
      currentView = "table";

      tableViewButton.classList.add("active");

      if (gridViewButton) {
        gridViewButton.classList.remove("active");
      }

      renderCatalogue();
    });
  }

  // ========================================================
  // CARD / TABLE ACTIONS
  // ========================================================

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-action]");

    if (!target) {
      return;
    }

    const action = target.dataset.action;

    const id = target.dataset.id;

    if (!id) {
      return;
    }

    if (action === "edit") {
      openEditModal(id);
    }

    if (action === "delete") {
      openDeleteModal(id);
    }

    if (action === "preview") {
      openImagePreview(id);
    }
  });

  // ========================================================
  // OPEN EDIT MODAL
  // ========================================================

  function openEditModal(id) {
    const item = findItemById(id);

    if (!item) {
      showPageStatus("Jewellery item not found.", "error");

      return;
    }

    editingItemId = String(item.id);

    editId.value = item.id || "";

    editIdDisplay.value = item.id || "";

    editName.value = item.name || "";

    editCollection.value = String(item.collection || "gold").toLowerCase();

    editGender.value = item.gender || "Unisex";

    editType.value = item.type || item.category || "";

    editSubtype.value = item.subtype || "";

    editDescription.value = item.description || "";

    editImage.value = "";

    const imageUrl =
      item.image_url || buildImageUrl(item.collection, item.image);

    if (editPreviewImage) {
      editPreviewImage.src = imageUrl || "";
    }

    clearEditStatus();

    editModal.classList.remove("hidden");

    document.body.classList.add("modal-open");
  }

  // ========================================================
  // CLOSE EDIT MODAL
  // ========================================================

  function closeEditModal() {
    editModal.classList.add("hidden");

    document.body.classList.remove("modal-open");

    editingItemId = null;

    clearEditStatus();
  }

  if (closeModal) {
    closeModal.addEventListener("click", closeEditModal);
  }

  if (cancelEdit) {
    cancelEdit.addEventListener("click", closeEditModal);
  }

  if (modalOverlay) {
    modalOverlay.addEventListener("click", closeEditModal);
  }

  // ========================================================
  // EDIT IMAGE PREVIEW
  // ========================================================

  if (editImage) {
    editImage.addEventListener("change", (event) => {
      const file = event.target.files[0];

      if (!file) {
        return;
      }

      const reader = new FileReader();

      reader.onload = (event) => {
        editPreviewImage.src = event.target.result;
      };

      reader.readAsDataURL(file);
    });
  }

  // ========================================================
  // SUBMIT EDIT
  // ========================================================

  if (editForm) {
    editForm.addEventListener("submit", async (event) => {
      event.preventDefault();

      if (!editingItemId) {
        return;
      }

      setEditStatus("Saving changes...", "loading");

      const formData = new FormData();

      formData.append("name", editName.value.trim());

      formData.append("collection", editCollection.value);

      formData.append("gender", editGender.value);

      formData.append("type", editType.value.trim());

      formData.append("subtype", editSubtype.value.trim());

      formData.append("description", editDescription.value.trim());

      if (editImage.files && editImage.files.length > 0) {
        formData.append("image", editImage.files[0]);
      }

      try {
        const response = await fetch(
          `/api/jewellery/${encodeURIComponent(editingItemId)}`,
          {
            method: "PUT",
            body: formData,
          },
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.error || data.message || "Unable to update jewellery.",
          );
        }

        setEditStatus("Jewellery updated successfully.", "success");

        await loadCatalogue();

        setTimeout(() => {
          closeEditModal();
        }, 700);
      } catch (error) {
        console.error("Edit error:", error);

        setEditStatus(error.message || "Unable to update jewellery.", "error");
      }
    });
  }

  // ========================================================
  // OPEN DELETE MODAL
  // ========================================================

  function openDeleteModal(id) {
    const item = findItemById(id);

    if (!item) {
      showPageStatus("Jewellery item not found.", "error");

      return;
    }

    deletingItemId = String(item.id);

    deleteItemName.textContent = `${item.id || ""} — ${
      item.name || "Unnamed Jewellery"
    }`;

    deleteStatus.textContent = "";

    deleteStatus.className = "modal-status";

    deleteModal.classList.remove("hidden");

    document.body.classList.add("modal-open");
  }

  // ========================================================
  // CLOSE DELETE MODAL
  // ========================================================

  function closeDeleteModal() {
    deleteModal.classList.add("hidden");

    document.body.classList.remove("modal-open");

    deletingItemId = null;
  }

  if (cancelDelete) {
    cancelDelete.addEventListener("click", closeDeleteModal);
  }

  if (deleteModalOverlay) {
    deleteModalOverlay.addEventListener("click", closeDeleteModal);
  }

  // ========================================================
  // DELETE ITEM
  // ========================================================

  if (confirmDelete) {
    confirmDelete.addEventListener("click", async () => {
      if (!deletingItemId) {
        return;
      }

      deleteStatus.textContent = "Deleting jewellery...";

      deleteStatus.className = "modal-status loading";

      confirmDelete.disabled = true;

      try {
        const response = await fetch(
          `/api/jewellery/${encodeURIComponent(deletingItemId)}`,
          {
            method: "DELETE",
          },
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.error || data.message || "Unable to delete jewellery.",
          );
        }

        closeDeleteModal();

        showPageStatus("Jewellery deleted successfully.", "success");

        await loadCatalogue();
      } catch (error) {
        console.error("Delete error:", error);

        deleteStatus.textContent =
          error.message || "Unable to delete jewellery.";

        deleteStatus.className = "modal-status error";
      } finally {
        confirmDelete.disabled = false;
      }
    });
  }

  // ========================================================
  // IMAGE PREVIEW
  // ========================================================

  function openImagePreview(id) {
    const item = findItemById(id);

    if (!item) {
      return;
    }

    const imageUrl =
      item.image_url || buildImageUrl(item.collection, item.image);

    if (!imageUrl) {
      return;
    }

    largePreviewImage.src = imageUrl;

    largePreviewImage.alt = item.name || "Jewellery";

    largePreviewName.textContent = `${item.id || ""} — ${
      item.name || "Jewellery"
    }`;

    imageModal.classList.remove("hidden");

    document.body.classList.add("modal-open");
  }

  function closeImagePreview() {
    imageModal.classList.add("hidden");

    document.body.classList.remove("modal-open");

    largePreviewImage.src = "";
  }

  if (closeImageModal) {
    closeImageModal.addEventListener("click", closeImagePreview);
  }

  if (imageModalOverlay) {
    imageModalOverlay.addEventListener("click", closeImagePreview);
  }

  // ========================================================
  // REFRESH
  // ========================================================

  if (refreshButton) {
    refreshButton.addEventListener("click", async () => {
      refreshButton.disabled = true;

      refreshButton.textContent = "↻ Loading...";

      await loadCatalogue();

      refreshButton.disabled = false;

      refreshButton.textContent = "↻ Refresh";
    });
  }

  // ========================================================
  // REBUILD INDEX
  // ========================================================

  if (rebuildButton) {
    rebuildButton.addEventListener("click", async () => {
      const confirmed = window.confirm(
        "Rebuild the complete jewellery search index?\n\nThis may take some time because the catalogue images will be processed again.",
      );

      if (!confirmed) {
        return;
      }

      rebuildButton.disabled = true;

      rebuildButton.textContent = "↻ Rebuilding...";

      showPageStatus("Rebuilding search index. Please wait...", "loading");

      try {
        const response = await fetch("/api/jewellery/rebuild-index", {
          method: "POST",
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.error || data.message || "Unable to rebuild search index.",
          );
        }

        showPageStatus(
          data.message || "Search index rebuilt successfully.",
          "success",
        );

        await loadCatalogue();
      } catch (error) {
        console.error("Rebuild error:", error);

        showPageStatus(
          error.message || "Unable to rebuild search index.",
          "error",
        );
      } finally {
        rebuildButton.disabled = false;

        rebuildButton.textContent = "↻ Rebuild Search Index";
      }
    });
  }

  // ========================================================
  // FIND ITEM
  // ========================================================

  function findItemById(id) {
    return catalogue.find((item) => String(item.id) === String(id));
  }

  // ========================================================
  // IMAGE URL
  // ========================================================

  function buildImageUrl(collection, filename) {
    if (!collection || !filename) {
      return "";
    }

    const safeCollection = String(collection).toLowerCase();

    if (safeCollection !== "gold" && safeCollection !== "prototype") {
      return "";
    }

    return (
      "/catalogue/" +
      encodeURIComponent(safeCollection) +
      "/" +
      encodeURIComponent(filename)
    );
  }

  // ========================================================
  // EMPTY STATE
  // ========================================================

  function showEmptyState() {
    if (emptyState) {
      emptyState.classList.remove("hidden");
    }
  }

  function hideEmptyState() {
    if (emptyState) {
      emptyState.classList.add("hidden");
    }
  }

  // ========================================================
  // LOADING
  // ========================================================

  function showLoading() {
    if (loading) {
      loading.classList.remove("hidden");
    }

    if (catalogueGrid) {
      catalogueGrid.classList.add("hidden");
    }

    if (catalogueTableWrapper) {
      catalogueTableWrapper.classList.add("hidden");
    }

    hideEmptyState();
  }

  function hideLoading() {
    if (loading) {
      loading.classList.add("hidden");
    }
  }

  // ========================================================
  // PAGE STATUS
  // ========================================================

  function showPageStatus(message, type = "info") {
    if (!pageStatus) {
      return;
    }

    pageStatus.textContent = message;

    pageStatus.className = `page-status ${type}`;

    pageStatus.classList.remove("hidden");

    if (type === "success") {
      setTimeout(() => {
        pageStatus.classList.add("hidden");
      }, 4000);
    }
  }

  // ========================================================
  // EDIT STATUS
  // ========================================================

  function setEditStatus(message, type = "") {
    if (!editStatus) {
      return;
    }

    editStatus.textContent = message;

    editStatus.className = `modal-status ${type}`;
  }

  function clearEditStatus() {
    if (!editStatus) {
      return;
    }

    editStatus.textContent = "";

    editStatus.className = "modal-status";
  }

  // ========================================================
  // TEXT HELPERS
  // ========================================================

  function truncateText(text, maxLength) {
    const value = String(text || "");

    if (value.length <= maxLength) {
      return value;
    }

    return value.substring(0, maxLength).trim() + "...";
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function escapeAttribute(value) {
    return escapeHtml(value);
  }

  // ========================================================
  // ESC KEY
  // ========================================================

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }

    if (editModal && !editModal.classList.contains("hidden")) {
      closeEditModal();
    }

    if (deleteModal && !deleteModal.classList.contains("hidden")) {
      closeDeleteModal();
    }

    if (imageModal && !imageModal.classList.contains("hidden")) {
      closeImagePreview();
    }
  });

  // ========================================================
  // INITIAL VIEW
  // ========================================================

  if (gridViewButton) {
    gridViewButton.classList.add("active");
  }

  console.log("JewelMatch catalogue manager ready.");
});
