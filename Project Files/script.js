document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('input[type="file"]');
    const preview = document.createElement('img');
    preview.classList.add('result-image');
    fileInput.after(preview);

    fileInput.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
        } else {
            preview.src = '';
            preview.style.display = 'none';
        }
    });
});
