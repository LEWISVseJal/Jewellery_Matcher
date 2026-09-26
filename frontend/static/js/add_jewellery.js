document.addEventListener("DOMContentLoaded", () => {

    const imageInput =
        document.getElementById("jewelleryImage");

    const chooseImageButton =
        document.getElementById("chooseImageButton");

    const imageDropZone =
        document.getElementById("imageDropZone");

    const imageUploadContent =
        document.getElementById("imageUploadContent");

    const imagePreviewContainer =
        document.getElementById("imagePreviewContainer");

    const imagePreview =
        document.getElementById("imagePreview");

    const imageName =
        document.getElementById("imageName");

    const removeImageButton =
        document.getElementById("removeJewelleryImage");

    const jewelleryName =
        document.getElementById("jewelleryName");

    const gender =
        document.getElementById("gender");

    const collection =
        document.getElementById("collection");

    const jewelleryType =
        document.getElementById("jewelleryType");

    const subtype =
        document.getElementById("subtype");

    const description =
        document.getElementById("description");

    const addButton =
        document.getElementById("addJewelleryButton");

    const formStatus =
        document.getElementById("formStatus");


    let selectedFile = null;


    /* ========================================================
       SUBTYPE OPTIONS
    ======================================================== */

    const subtypeOptions = {

        "Ring": [
            "Solitaire",
            "Halo",
            "Cluster",
            "Cocktail",
            "Band",
            "Engagement",
            "Statement",
            "Other"
        ],

        "Necklace": [
            "Pendant Necklace",
            "Choker",
            "Chain Necklace",
            "Layered",
            "Statement",
            "Other"
        ],

        "Earrings": [
            "Stud",
            "Hoop",
            "Drop",
            "Jhumka",
            "Chandbali",
            "Danglers",
            "Other"
        ],

        "Bracelet": [
            "Chain Bracelet",
            "Tennis",
            "Charm",
            "Cuff",
            "Statement",
            "Other"
        ],

        "Bangle": [
            "Plain",
            "Stone Studded",
            "Designer",
            "Kada",
            "Set",
            "Other"
        ],

        "Pendant": [
            "Solitaire",
            "Heart",
            "Religious",
            "Floral",
            "Geometric",
            "Other"
        ],

        "Chain": [
            "Cable",
            "Rope",
            "Box",
            "Figaro",
            "Singapore",
            "Other"
        ],

        "Anklet": [
            "Chain",
            "Charm",
            "Beaded",
            "Traditional",
            "Other"
        ],

        "Nose Pin": [
            "Stud",
            "Hoop",
            "Stone",
            "Floral",
            "Other"
        ],

        "Other": [
            "Other"
        ]

    };


    /* ========================================================
       CHOOSE IMAGE
    ======================================================== */

    chooseImageButton.addEventListener(
        "click",
        () => {

            imageInput.click();

        }
    );


    /* ========================================================
       FILE INPUT
    ======================================================== */

    imageInput.addEventListener(
        "change",
        (event) => {

            const file =
                event.target.files[0];

            handleImage(file);

        }
    );


    /* ========================================================
       IMAGE HANDLER
    ======================================================== */

    function handleImage(file) {

        if (!file) {
            return;
        }


        const allowedTypes = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp"
        ];


        if (
            !allowedTypes.includes(
                file.type
            )
        ) {

            showStatus(
                "Please upload a JPG, PNG, WEBP or BMP image.",
                "error"
            );

            return;
        }


        if (
            file.size >
            10 * 1024 * 1024
        ) {

            showStatus(
                "Image must be smaller than 10 MB.",
                "error"
            );

            return;
        }


        selectedFile =
            file;


        const reader =
            new FileReader();


        reader.onload =
            (event) => {

                imagePreview.src =
                    event.target.result;

                imageName.textContent =
                    file.name;

                imageUploadContent.classList.add(
                    "hidden"
                );

                imagePreviewContainer.classList.remove(
                    "hidden"
                );

                hideStatus();

            };


        reader.readAsDataURL(file);

    }


    /* ========================================================
       DRAG & DROP
    ======================================================== */

    imageDropZone.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

            imageDropZone.style.borderColor =
                "#b27d27";

        }
    );


    imageDropZone.addEventListener(
        "dragleave",
        () => {

            imageDropZone.style.borderColor =
                "";

        }
    );


    imageDropZone.addEventListener(
        "drop",
        (event) => {

            event.preventDefault();

            imageDropZone.style.borderColor =
                "";

            const file =
                event.dataTransfer.files[0];

            handleImage(file);

        }
    );


    /* ========================================================
       REMOVE IMAGE
    ======================================================== */

    removeImageButton.addEventListener(
        "click",
        () => {

            clearImage();

            hideStatus();

        }
    );


    function clearImage() {

        selectedFile = null;

        imageInput.value = "";

        imagePreview.src = "";

        imageName.textContent = "";

        imagePreviewContainer.classList.add(
            "hidden"
        );

        imageUploadContent.classList.remove(
            "hidden"
        );

    }


    /* ========================================================
       TYPE → SUBTYPE
    ======================================================== */

    jewelleryType.addEventListener(
        "change",
        () => {

            const type =
                jewelleryType.value;


            subtype.innerHTML =
                `<option value="">Select subtype</option>`;


            if (!subtypeOptions[type]) {
                return;
            }


            subtypeOptions[type].forEach(
                (option) => {

                    const element =
                        document.createElement(
                            "option"
                        );

                    element.value =
                        option;

                    element.textContent =
                        option;

                    subtype.appendChild(
                        element
                    );

                }
            );

        }
    );


    /* ========================================================
       STATUS
    ======================================================== */

    function showStatus(
        message,
        type
    ) {

        formStatus.textContent =
            message;

        formStatus.className =
            `form-status ${type}`;

    }


    function hideStatus() {

        formStatus.className =
            "form-status hidden";

        formStatus.textContent =
            "";

    }


    /* ========================================================
       RESET FORM
    ======================================================== */

    function resetFormAfterSuccess() {

        clearImage();

        jewelleryName.value = "";

        description.value = "";

        gender.value = "";

        collection.value = "";

        jewelleryType.value = "";

        subtype.innerHTML =
            `<option value="">Select subtype</option>`;

        jewelleryName.focus();

    }


    /* ========================================================
       ADD JEWELLERY
    ======================================================== */

    addButton.addEventListener(
        "click",
        async () => {

            hideStatus();


            if (!selectedFile) {

                showStatus(
                    "Please upload a jewellery image.",
                    "error"
                );

                return;
            }


            if (
                !jewelleryName.value.trim()
            ) {

                showStatus(
                    "Please enter a jewellery name.",
                    "error"
                );

                jewelleryName.focus();

                return;
            }


            if (!gender.value) {

                showStatus(
                    "Please select the gender.",
                    "error"
                );

                gender.focus();

                return;
            }


            if (!collection.value) {

                showStatus(
                    "Please select the collection.",
                    "error"
                );

                collection.focus();

                return;
            }


            if (!jewelleryType.value) {

                showStatus(
                    "Please select the jewellery type.",
                    "error"
                );

                jewelleryType.focus();

                return;
            }


            const formData =
                new FormData();


            formData.append(
                "image",
                selectedFile
            );

            formData.append(
                "name",
                jewelleryName.value.trim()
            );

            formData.append(
                "gender",
                gender.value
            );

            formData.append(
                "type",
                jewelleryType.value
            );

            formData.append(
                "subtype",
                subtype.value
            );

            formData.append(
                "collection",
                collection.value
            );

            formData.append(
                "description",
                description.value.trim()
            );


            addButton.disabled =
                true;


            addButton.innerHTML =
                `<span>◌</span> Processing...`;


            showStatus(
                "Processing image and creating AI embedding...",
                "processing"
            );


            try {

                const response =
                    await fetch(
                        "/api/jewellery/add",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                let data;


                try {

                    data =
                        await response.json();

                } catch (error) {

                    throw new Error(
                        "The server returned an invalid response."
                    );

                }


                if (
                    !response.ok ||
                    !data.success
                ) {

                    throw new Error(
                        data.error ||
                        "Unable to add jewellery."
                    );

                }


                const addedJewellery =
                    data.jewellery || {};


                const addedName =
                    addedJewellery.name ||
                    jewelleryName.value.trim();


                const addedId =
                    addedJewellery.id ||
                    "";


                const addedCollection =
                    addedJewellery.collection ||
                    collection.value;


                showStatus(
                    `✓ ${addedName} added successfully` +
                    (
                        addedId
                            ? ` (${addedId})`
                            : ""
                    ) +
                    ` to ${addedCollection}. You can add another jewellery.`,
                    "success"
                );


                resetFormAfterSuccess();


            } catch (error) {

                console.error(
                    "Add Jewellery Error:",
                    error
                );


                showStatus(
                    error.message ||
                    "Something went wrong while adding the jewellery.",
                    "error"
                );


            } finally {

                addButton.disabled =
                    false;


                addButton.innerHTML = `
                    <span>✦</span>
                    Add Jewellery
                    <strong>→</strong>
                `;

            }

        }
    );

});