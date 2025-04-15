// static/js/mask_selection.js

// Assuming we have mask data available
fetch('{{ url_for("get_mask_data", filename=original_image) }}')
    .then(response => response.json())
    .then(data => {
        const masks = data.masks;
        const imageElement = document.querySelector('#segmented-image img');
        const imageRect = imageElement.getBoundingClientRect();
        const container = document.getElementById('segmented-image');

        masks.forEach((mask, idx) => {
            const maskElement = document.createElement('div');
            maskElement.classList.add('mask-area');
            maskElement.style.left = mask.bbox[0] + 'px';
            maskElement.style.top = mask.bbox[1] + 'px';
            maskElement.style.width = mask.bbox[2] + 'px';
            maskElement.style.height = mask.bbox[3] + 'px';
            maskElement.dataset.maskId = idx;
            maskElement.addEventListener('click', () => {
                document.getElementById('selected-mask-id').value = idx;
                alert('Selected mask ID: ' + idx);
            });
            container.appendChild(maskElement);
        });
    })
    .catch(error => console.error('Error fetching mask data:', error));