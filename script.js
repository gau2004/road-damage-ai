// ---------- Predict page ----------
const predictForm = document.getElementById('predict-form');
if (predictForm) {
  const imageInput = document.getElementById('image-input');
  const previewImg = document.getElementById('preview-img');
  const uploadLabelText = document.getElementById('upload-label-text');
  const loading = document.getElementById('loading');
  const resultBox = document.getElementById('result');
  const errorBox = document.getElementById('error-box');

  imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.classList.remove('hidden');
      uploadLabelText.textContent = file.name;
    };
    reader.readAsDataURL(file);
  });

  predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.classList.add('hidden');
    resultBox.classList.add('hidden');

    const file = imageInput.files[0];
    if (!file) {
      errorBox.textContent = 'Please choose an image first.';
      errorBox.classList.remove('hidden');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    loading.classList.remove('hidden');

    try {
      const res = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();
      loading.classList.add('hidden');

      if (!res.ok) {
        errorBox.textContent = data.error || 'Something went wrong.';
        errorBox.classList.remove('hidden');
        return;
      }

      document.getElementById('result-label').textContent =
        data.prediction === 'Broken Road' ? '🚧 Broken Road' : '✅ Not Broken Road';
      document.getElementById('result-confidence').textContent =
        `Confidence: ${data.confidence}%`;
      document.getElementById('confidence-bar').style.width = `${data.confidence}%`;
      resultBox.classList.remove('hidden');
    } catch (err) {
      loading.classList.add('hidden');
      errorBox.textContent = 'Request failed: ' + err.message;
      errorBox.classList.remove('hidden');
    }
  });
}

// ---------- Retrain page ----------
const retrainForm = document.getElementById('retrain-form');
if (retrainForm) {
  const statusBox = document.getElementById('training-status');
  const errorBox = document.getElementById('retrain-error');
  const resultsSection = document.getElementById('results-section');

  retrainForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.classList.add('hidden');
    resultsSection.classList.add('hidden');
    statusBox.classList.remove('hidden');

    const formData = new FormData(retrainForm);
    const payload = {
      optimizer: formData.get('optimizer'),
      activation: formData.get('activation'),
      learning_rate: parseFloat(formData.get('learning_rate')),
      batch_size: parseInt(formData.get('batch_size')),
      epochs: parseInt(formData.get('epochs')),
      dropout: parseFloat(formData.get('dropout')),
      image_size: parseInt(formData.get('image_size')),
      augmentation: formData.get('augmentation') === 'on'
    };

    try {
      const res = await fetch('/retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      statusBox.classList.add('hidden');

      if (!res.ok) {
        errorBox.textContent = data.error || 'Training failed.';
        errorBox.classList.remove('hidden');
        return;
      }

      document.getElementById('m-train-acc').textContent = data.training_accuracy + '%';
      document.getElementById('m-val-acc').textContent = data.validation_accuracy + '%';
      document.getElementById('m-precision').textContent = data.precision + '%';
      document.getElementById('m-recall').textContent = data.recall + '%';
      document.getElementById('m-f1').textContent = data.f1_score + '%';
      document.getElementById('m-time').textContent = data.training_time_sec;

      document.getElementById('graph-accuracy').src = data.accuracy_graph;
      document.getElementById('graph-loss').src = data.loss_graph;
      document.getElementById('graph-cm').src = data.confusion_matrix_graph;

      resultsSection.classList.remove('hidden');
    } catch (err) {
      statusBox.classList.add('hidden');
      errorBox.textContent = 'Request failed: ' + err.message;
      errorBox.classList.remove('hidden');
    }
  });
}
