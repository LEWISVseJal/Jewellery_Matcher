/*
    Frontend JavaScript for Jewellery AI Matcher.

    Responsibilities:

    1. Read selected image
    2. Display preview
    3. Send image to Flask backend
    4. Receive matching results
    5. Display results
*/


// --------------------------------------------------
// ELEMENTS
// --------------------------------------------------

const imageInput =
    document.getElementById("imageInput");

const matchButton =
    document.getElementById("matchButton");

const previewSection =
    document.getElementById("previewSection");

const previewImage =
    document.getElementById("previewImage");

const loading =
    document.getElementById("loading");

const resultsSection =
    document.getElementById("resultsSection");

const resultsContainer =
    document.getElementById("results");

const errorBox =
    document.getElementById("error");


// --------------------------------------------------
// IMAGE PREVIEW
// --------------------------------------------------

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];


        if (!file) {

            return;
        }


        // Create temporary browser URL
        const imageURL =
            URL.createObjectURL(file);


        previewImage.src =
            imageURL;


        previewSection.classList.remove(
            "hidden"
        );
    }
);


// --------------------------------------------------
// MATCH BUTTON
// --------------------------------------------------

matchButton.addEventListener(
    "click",
    async function () {

        const file =
            imageInput.files[0];


        // Check image
        if (!file) {

            showError(
                "Please select a jewellery image first."
            );

            return;
        }


        // Hide previous results
        resultsSection.classList.add(
            "hidden"
        );


        errorBox.classList.add(
            "hidden"
        );


        loading.classList.remove(
            "hidden"
        );


        // ------------------------------------------------
        // CREATE FORM DATA
        // ------------------------------------------------

        const formData =
            new FormData();


        formData.append(
            "image",
            file
        );


        try {

            // --------------------------------------------
            // SEND IMAGE TO BACKEND
            // --------------------------------------------

            const response =
                await fetch(
                    "http://127.0.0.1:5000/api/match",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            // --------------------------------------------
            // HANDLE ERROR
            // --------------------------------------------

            if (
                !response.ok
                ||
                data.status === "error"
            ) {

                throw new Error(
                    data.message
                    ||
                    "Something went wrong."
                );
            }


            // --------------------------------------------
            // DISPLAY RESULTS
            // --------------------------------------------

            displayResults(
                data.results
            );


        } catch (error) {

            showError(
                error.message
            );


        } finally {

            loading.classList.add(
                "hidden"
            );
        }

    }
);


// --------------------------------------------------
// DISPLAY RESULTS
// --------------------------------------------------

function displayResults(results) {

    resultsContainer.innerHTML = "";


    if (
        !results
        ||
        results.length === 0
    ) {

        resultsContainer.innerHTML =
            "<p>No similar jewellery found.</p>";

        resultsSection.classList.remove(
            "hidden"
        );

        return;
    }


    results.forEach(
        function (item) {

            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "result-card";


            card.innerHTML = `

                <img
                    src="http://127.0.0.1:5000/catalogue/${item.image}"
                    alt="${item.name}"
                >

                <h3>
                    ${item.name}
                </h3>

                <p>
                    ID: ${item.id}
                </p>

                <p>
                    Category: ${item.category}
                </p>

                <div class="similarity">
                    ${item.similarity}%
                </div>

            `;


            resultsContainer.appendChild(
                card
            );
        }
    );


    resultsSection.classList.remove(
        "hidden"
    );
}


// --------------------------------------------------
// ERROR MESSAGE
// --------------------------------------------------

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );
}