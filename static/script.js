async function send() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    document.getElementById("result").innerText =
        "Emotion: " + data.emotion +
        " (confidence: " + data.confidence + ")";
}
