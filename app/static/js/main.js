document.addEventListener("DOMContentLoaded", () => {
  const alerts = document.querySelectorAll(".alert");
  const uploadForm = document.querySelector("#uploadForm");
  const uploadButton = document.querySelector("#uploadButton");
  const uploadProgressWrap = document.querySelector("#uploadProgressWrap");
  const uploadProgressBar = document.querySelector("#uploadProgressBar");
  const uploadProgressLabel = document.querySelector("#uploadProgressLabel");

  alerts.forEach((alert) => {
    window.setTimeout(() => {
      const instance = bootstrap.Alert.getOrCreateInstance(alert);
      instance.close();
    }, 4500);
  });

  if (!uploadForm) {
    return;
  }

  const setProgress = (value) => {
    const percent = `${Math.max(0, Math.min(100, Math.round(value)))}%`;
    uploadProgressBar.style.width = percent;
    uploadProgressBar.textContent = percent;
    uploadProgressLabel.textContent = percent;
    uploadProgressBar.parentElement.setAttribute("aria-valuenow", parseInt(percent, 10));
  };

  uploadForm.addEventListener("submit", (event) => {
    event.preventDefault();

    const fileInput = uploadForm.querySelector("#dataset");
    if (!fileInput.files.length) {
      uploadForm.reportValidity();
      return;
    }

    const request = new XMLHttpRequest();
    const formData = new FormData(uploadForm);

    uploadProgressWrap.classList.remove("d-none");
    uploadButton.disabled = true;
    setProgress(0);

    request.upload.addEventListener("progress", (progressEvent) => {
      if (progressEvent.lengthComputable) {
        setProgress((progressEvent.loaded / progressEvent.total) * 100);
      }
    });

    request.addEventListener("load", () => {
      setProgress(100);
      document.open();
      document.write(request.responseText);
      document.close();
    });

    request.addEventListener("error", () => {
      uploadButton.disabled = false;
      uploadProgressBar.classList.remove("progress-bar-animated");
      uploadProgressBar.classList.add("bg-danger");
      uploadProgressBar.textContent = "Upload failed";
      uploadProgressLabel.textContent = "Error";
    });

    request.open("POST", uploadForm.action || window.location.href);
    request.send(formData);
  });
});
