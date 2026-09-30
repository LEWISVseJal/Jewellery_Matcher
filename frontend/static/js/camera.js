/*
============================================================
JewelMatch AI
REAL-TIME CAMERA CAPTURE
============================================================

This module does NOT replace the existing upload logic.

It:
1. Opens the device camera.
2. Shows live preview.
3. Captures a frame.
4. Converts it to a File.
5. Places the File into the existing image input.
6. Fires the normal "change" event.

Therefore the existing app.js upload/matching flow
continues to work normally.
*/

document.addEventListener("DOMContentLoaded", () => {
  console.log("JewelMatch AI - Camera module loaded");

  // ========================================================
  // ELEMENTS
  // ========================================================

  const cameraButton =
    document.getElementById("cameraButton");

  const cameraModal =
    document.getElementById("cameraModal");

  const cameraVideo =
    document.getElementById("cameraVideo");

  const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

  const cameraError =
    document.getElementById("cameraError");

  const closeCameraButton =
    document.getElementById("closeCameraButton");

  const cancelCameraButton =
    document.getElementById("cancelCameraButton");

  const captureButton =
    document.getElementById("captureButton");

  const imageInput =
    document.getElementById("imageInput");

  // ========================================================
  // STATE
  // ========================================================

  let cameraStream = null;

  let cameraOpening = false;

  // ========================================================
  // HELPERS
  // ========================================================

  function showModal() {
    if (!cameraModal) {
      return;
    }

    cameraModal.classList.remove("hidden");
  }

  function hideModal() {
    if (!cameraModal) {
      return;
    }

    cameraModal.classList.add("hidden");
  }

  function showCameraError(message) {
    if (!cameraError) {
      console.error(message);
      return;
    }

    cameraError.textContent = message;

    cameraError.classList.remove("hidden");
  }

  function clearCameraError() {
    if (!cameraError) {
      return;
    }

    cameraError.textContent = "";

    cameraError.classList.add("hidden");
  }

  // ========================================================
  // STOP CAMERA
  // ========================================================

  function stopCamera() {
    if (cameraStream) {
      cameraStream
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      cameraStream = null;
    }

    if (cameraVideo) {
      cameraVideo.pause();

      cameraVideo.srcObject = null;
    }
  }

  // ========================================================
  // OPEN CAMERA
  // ========================================================

  async function openCamera() {
    if (cameraOpening) {
      return;
    }

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      showCameraError(
        "Camera access is not supported by this browser. Please use Browse Image."
      );

      showModal();

      return;
    }

    cameraOpening = true;

    clearCameraError();

    showModal();

    if (cameraPlaceholder) {
      cameraPlaceholder.classList.add("hidden");
    }

    try {
      /*
      Prefer the rear camera on mobile devices because
      jewellery will normally be photographed using the
      rear camera.
      */

      cameraStream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: "environment",
            },

            width: {
              ideal: 1280,
            },

            height: {
              ideal: 960,
            },
          },

          audio: false,
        });

      if (!cameraVideo) {
        throw new Error(
          "Camera preview element not found."
        );
      }

      cameraVideo.srcObject =
        cameraStream;

      await cameraVideo.play();

      console.log(
        "JewelMatch AI - Camera started"
      );

    } catch (error) {
      console.error(
        "Camera access error:",
        error
      );

      stopCamera();

      if (cameraPlaceholder) {
        cameraPlaceholder.classList.remove(
          "hidden"
        );
      }

      let message =
        "Unable to access the camera.";

      if (
        error &&
        error.name ===
          "NotAllowedError"
      ) {
        message =
          "Camera permission was denied. Please allow camera access and try again.";
      } else if (
        error &&
        error.name ===
          "NotFoundError"
      ) {
        message =
          "No camera was found on this device.";
      } else if (
        error &&
        error.name ===
          "NotReadableError"
      ) {
        message =
          "The camera is already being used by another application.";
      } else if (
        error &&
        error.name ===
          "SecurityError"
      ) {
        message =
          "Camera access requires a secure connection (HTTPS) or localhost.";
      }

      showCameraError(message);
    } finally {
      cameraOpening = false;
    }
  }

  // ========================================================
  // CAPTURE PHOTO
  // ========================================================

  async function capturePhoto() {
    if (!cameraVideo) {
      return;
    }

    if (
      !cameraVideo.videoWidth ||
      !cameraVideo.videoHeight
    ) {
      showCameraError(
        "Camera is not ready yet. Please wait a moment and try again."
      );

      return;
    }

    clearCameraError();

    const width =
      cameraVideo.videoWidth;

    const height =
      cameraVideo.videoHeight;

    const canvas =
      document.createElement("canvas");

    canvas.width = width;

    canvas.height = height;

    const context =
      canvas.getContext("2d");

    if (!context) {
      showCameraError(
        "Unable to capture the camera image."
      );

      return;
    }

    /*
    Draw the current video frame onto
    the canvas.
    */

    context.drawImage(
      cameraVideo,
      0,
      0,
      width,
      height
    );

    try {
      const blob =
        await new Promise(
          (resolve, reject) => {
            canvas.toBlob(
              (result) => {
                if (result) {
                  resolve(result);
                } else {
                  reject(
                    new Error(
                      "Could not create image."
                    )
                  );
                }
              },
              "image/jpeg",
              0.92
            );
          }
        );

      const timestamp =
        new Date()
          .toISOString()
          .replace(
            /[:.]/g,
            "-"
          );

      const file =
        new File(
          [
            blob,
          ],
          `camera-jewellery-${timestamp}.jpg`,
          {
            type: "image/jpeg",
            lastModified:
              Date.now(),
          }
        );

      /*
      Put the captured File into the
      SAME file input used by Browse.
      */

      if (!imageInput) {
        throw new Error(
          "Image input was not found."
        );
      }

      const dataTransfer =
        new DataTransfer();

      dataTransfer.items.add(
        file
      );

      imageInput.files =
        dataTransfer.files;

      /*
      Trigger the existing app.js
      "change" event.

      app.js will therefore run:
          handleSelectedFile(file)
      */

      imageInput.dispatchEvent(
        new Event(
          "change",
          {
            bubbles: true,
          }
        )
      );

      console.log(
        "JewelMatch AI - Camera photo captured"
      );

      stopCamera();

      hideModal();

    } catch (error) {
      console.error(
        "Camera capture failed:",
        error
      );

      showCameraError(
        "Unable to capture the image. Please try again."
      );
    }
  }

  // ========================================================
  // CLOSE CAMERA
  // ========================================================

  function closeCamera() {
    stopCamera();

    clearCameraError();

    hideModal();

    console.log(
      "JewelMatch AI - Camera closed"
    );
  }

  // ========================================================
  // BUTTON EVENTS
  // ========================================================

  if (cameraButton) {
    cameraButton.addEventListener(
      "click",
      (event) => {
        event.preventDefault();

        event.stopPropagation();

        openCamera();
      }
    );
  }

  if (captureButton) {
    captureButton.addEventListener(
      "click",
      (event) => {
        event.preventDefault();

        capturePhoto();
      }
    );
  }

  if (closeCameraButton) {
    closeCameraButton.addEventListener(
      "click",
      (event) => {
        event.preventDefault();

        closeCamera();
      }
    );
  }

  if (cancelCameraButton) {
    cancelCameraButton.addEventListener(
      "click",
      (event) => {
        event.preventDefault();

        closeCamera();
      }
    );
  }

  // ========================================================
  // BACKDROP CLICK
  // ========================================================

  if (cameraModal) {
    cameraModal.addEventListener(
      "click",
      (event) => {
        if (
          event.target.classList.contains(
            "camera-modal-backdrop"
          )
        ) {
          closeCamera();
        }
      }
    );
  }

  // ========================================================
  // ESC KEY
  // ========================================================

  document.addEventListener(
    "keydown",
    (event) => {
      if (
        event.key === "Escape" &&
        cameraModal &&
        !cameraModal.classList.contains(
          "hidden"
        )
      ) {
        closeCamera();
      }
    }
  );

  // ========================================================
  // PAGE CLEANUP
  // ========================================================

  window.addEventListener(
    "beforeunload",
    () => {
      stopCamera();
    }
  );

  console.log(
    "JewelMatch AI - Camera ready"
  );
});