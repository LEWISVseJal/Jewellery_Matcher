let catalogueItems = [];

let activeCollection = "all";

const grid = document.getElementById("catalogueGrid");

const loading = document.getElementById("loading");

const emptyState = document.getElementById("emptyState");

const searchInput = document.getElementById("searchInput");

const statusMessage = document.getElementById("statusMessage");

const editModal = document.getElementById("editModal");

const editForm = document.getElementById("editForm");

const editStatus = document.getElementById("editStatus");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normaliseCollection(value) {
  const collection = String(value || "")
    .trim()
    .toLowerCase();

  if (collection === "gold") {
    return "gold";
  }

  if (collection === "prototype") {
    return "prototype";
  }

  return collection;
}

function collectionLabel(value) {
  return normaliseCollection(value) === "gold" ? "Gold" : "Prototype";
}

function showStatus(message, type = "success") {
  statusMessage.textContent = message;

  statusMessage.className = `status-message ${type}`;

  clearTimeout(showStatus.timer);

  showStatus.timer = setTimeout(() => {
    statusMessage.className = "status-message hidden";
  }, 4500);
}

function updateStats() {
  const gold = catalogueItems.filter(
    (item) => normaliseCollection(item.collection) === "gold",
  ).length;

  const prototype = catalogueItems.filter(
    (item) => normaliseCollection(item.collection) === "prototype",
  ).length;

  document.getElementById("totalCount").textContent = catalogueItems.length;

  document.getElementById("goldCount").textContent = gold;

  document.getElementById("prototypeCount").textContent = prototype;
}

function filteredItems() {
  const query = searchInput.value.trim().toLowerCase();

  return catalogueItems.filter((item) => {
    const collection = normaliseCollection(item.collection);

    if (activeCollection !== "all" && collection !== activeCollection) {
      return false;
    }

    if (!query) {
      return true;
    }

    const searchable = [
      item.id,
      item.name,
      item.type,
      item.subtype,
      item.gender,
      item.collection,
      item.description,
    ]
      .join(" ")
      .toLowerCase();

    return searchable.includes(query);
  });
}

function render() {
  const items = filteredItems();

  loading.classList.add("hidden");

  grid.innerHTML = "";

  if (!items.length) {
    emptyState.classList.remove("hidden");

    return;
  }

  emptyState.classList.add("hidden");

  grid.innerHTML = items
    .map((item) => {
      const collection = normaliseCollection(item.collection);

      const imageUrl =
        item.image_url ||
        (item.filename
          ? `/catalogue/${collection}/${encodeURIComponent(item.filename)}`
          : "");

      return `
                        <article class="catalogue-card">

                            <div class="card-image">

                                ${
                                  imageUrl
                                    ? `
                                            <img
                                                src="${escapeHtml(imageUrl)}"
                                                alt="${escapeHtml(item.name || item.id)}"
                                                loading="lazy"
                                                onerror="this.parentElement.classList.add('image-error'); this.style.display='none';"
                                            >
                                        `
                                    : ""
                                }


                                <span
                                    class="collection-pill ${collection}"
                                >
                                    ${escapeHtml(collectionLabel(collection))}
                                </span>

                            </div>


                            <div class="card-content">

                                <div class="card-topline">

                                    <span class="design-id">
                                        ${escapeHtml(item.id)}
                                    </span>

                                    <span class="type-pill">
                                        ${escapeHtml(item.type || "Jewellery")}
                                    </span>

                                </div>


                                <h3>
                                    ${escapeHtml(item.name || item.id)}
                                </h3>


                                <p class="card-meta">
                                    ${escapeHtml(item.gender || "Unisex")}

                                    ${
                                      item.subtype
                                        ? ` · ${escapeHtml(item.subtype)}`
                                        : ""
                                    }
                                </p>


                                ${
                                  item.description
                                    ? `
                                            <p class="card-description">
                                                ${escapeHtml(item.description)}
                                            </p>
                                        `
                                    : ""
                                }


                                <div class="card-actions">

                                    <button
                                        class="edit-button"
                                        onclick="openEdit('${escapeHtml(item.id)}')"
                                    >
                                        Edit
                                    </button>


                                    <button
                                        class="delete-button"
                                        onclick="deleteItem('${escapeHtml(item.id)}')"
                                    >
                                        Delete
                                    </button>

                                </div>

                            </div>

                        </article>
                    `;
    })
    .join("");
}

async function loadCatalogue() {
  try {
    loading.classList.remove("hidden");

    const response = await fetch("/api/catalogue");

    if (!response.ok) {
      throw new Error("Could not load catalogue.");
    }

    const data = await response.json();

    catalogueItems = data.items || [];

    updateStats();

    render();
  } catch (error) {
    loading.textContent = error.message;

    showStatus(error.message, "error");
  }
}

function openEdit(id) {
  const item = catalogueItems.find((entry) => String(entry.id) === String(id));

  if (!item) {
    return;
  }

  document.getElementById("editId").value = item.id;

  document.getElementById("editName").value = item.name || "";

  document.getElementById("editCollection").value =
    normaliseCollection(item.collection) || "gold";

  document.getElementById("editGender").value = item.gender || "Unisex";

  document.getElementById("editType").value = item.type || "";

  document.getElementById("editSubtype").value = item.subtype || "";

  document.getElementById("editDescription").value = item.description || "";

  document.getElementById("editImage").value = "";

  editStatus.textContent = "";

  editStatus.className = "modal-status";

  editModal.classList.remove("hidden");

  document.body.classList.add("modal-open");
}

function closeEdit() {
  editModal.classList.add("hidden");

  document.body.classList.remove("modal-open");

  editForm.reset();
}

async function deleteItem(id) {
  const item = catalogueItems.find((entry) => String(entry.id) === String(id));

  if (!item) {
    return;
  }

  const confirmed = window.confirm(
    `Delete ${item.name || id} (${id}) from the catalogue?`,
  );

  if (!confirmed) {
    return;
  }

  try {
    const response = await fetch(`/api/jewellery/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Delete failed.");
    }

    showStatus("Jewellery deleted and search index updated.");

    await loadCatalogue();
  } catch (error) {
    showStatus(error.message, "error");
  }
}

editForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const id = document.getElementById("editId").value;

  const formData = new FormData(editForm);

  editStatus.textContent = "Updating catalogue and search index...";

  try {
    const response = await fetch(`/api/jewellery/${encodeURIComponent(id)}`, {
      method: "POST",

      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Update failed.");
    }

    closeEdit();

    showStatus("Jewellery updated and search index rebuilt.");

    await loadCatalogue();
  } catch (error) {
    editStatus.textContent = error.message;

    editStatus.className = "modal-status error";
  }
});

searchInput.addEventListener("input", render);

document.querySelectorAll(".filter-button").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".filter-button")
      .forEach((item) => item.classList.remove("active"));

    button.classList.add("active");

    activeCollection = button.dataset.collection;

    render();
  });
});

document.getElementById("rebuildButton").addEventListener("click", async () => {
  const button = document.getElementById("rebuildButton");

  button.disabled = true;

  button.textContent = "↻ Rebuilding...";

  try {
    const response = await fetch("/api/jewellery/rebuild-index", {
      method: "POST",
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Index rebuild failed.");
    }

    showStatus(
      `Search index rebuilt. ${data.indexed} of ${data.total} designs indexed.`,
    );
  } catch (error) {
    showStatus(error.message, "error");
  } finally {
    button.disabled = false;

    button.textContent = "↻ Rebuild Search Index";
  }
});

document.getElementById("closeModal").addEventListener("click", closeEdit);

document.getElementById("cancelEdit").addEventListener("click", closeEdit);

document.querySelectorAll("[data-close-modal]").forEach((element) => {
  element.addEventListener("click", closeEdit);
});

window.openEdit = openEdit;

window.deleteItem = deleteItem;

loadCatalogue();
