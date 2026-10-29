/**
 * HeartGuard – Client-Side Interactivity & Chart.js Visualizations
 */

document.addEventListener('DOMContentLoaded', function () {
  initPresets();
  initFormValidation();
});

/**
 * Quick patient profile presets for testing and demonstrations
 */
function initPresets() {
  const btnHealthy = document.getElementById('btnPresetHealthy');
  const btnAtRisk = document.getElementById('btnPresetAtRisk');
  const btnClear = document.getElementById('btnClearForm');
  const form = document.getElementById('predictForm');

  if (!form) return;

  const healthyProfile = {
    age: 42,
    sex: '0',        // Female
    cp: '2',         // Atypical Angina
    trestbps: 118,   // Optimal BP
    chol: 185,       // Optimal cholesterol
    fbs: '0',        // No diabetes
    restecg: '0',    // Normal ECG
    thalach: 172,    // High max heart rate
    exang: '0',      // No exercise angina
    oldpeak: 0.2,    // Minimal ST depression
    slope: '1',      // Upsloping
    ca: '0',         // 0 vessels
    thal: '3',       // Normal perfusion
  };

  const highRiskProfile = {
    age: 67,
    sex: '1',        // Male
    cp: '4',         // Asymptomatic / Silent
    trestbps: 160,   // Stage 2 Hypertension
    chol: 286,       // High cholesterol
    fbs: '0',
    restecg: '2',    // LV Hypertrophy
    thalach: 108,    // Low exercise HR
    exang: '1',      // Exercise angina present
    oldpeak: 2.5,    // Severe ST depression
    slope: '2',      // Flat ST slope
    ca: '3',         // 3 vessels narrowed
    thal: '7',       // Reversible perfusion defect
  };

  function populateForm(data) {
    Object.keys(data).forEach(function (key) {
      const el = document.getElementById(key);
      if (el) {
        el.value = data[key];
        el.dispatchEvent(new Event('change'));
      }
    });
  }

  if (btnHealthy) {
    btnHealthy.addEventListener('click', function () {
      populateForm(healthyProfile);
    });
  }

  if (btnAtRisk) {
    btnAtRisk.addEventListener('click', function () {
      populateForm(highRiskProfile);
    });
  }

  if (btnClear) {
    btnClear.addEventListener('click', function () {
      form.reset();
    });
  }
}

/**
 * Handle form submission state and client-side input validation
 */
function initFormValidation() {
  const form = document.getElementById('predictForm');
  const submitBtn = document.getElementById('btnSubmit');

  if (!form || !submitBtn) return;

  form.addEventListener('submit', function (e) {
    const inputs = form.querySelectorAll('input[required], select[required]');
    let hasError = false;

    inputs.forEach(function (input) {
      if (!input.value || input.value.trim() === '') {
        input.style.borderColor = '#ef4444';
        hasError = true;
      } else {
        input.style.borderColor = '';
      }
    });

    if (hasError) {
      e.preventDefault();
      alert('Please fill in all 13 clinical biomarkers before submitting.');
      return;
    }

    // Show loading state
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <svg class="animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20" style="animation: spin 1s linear infinite;">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
        <path d="M12 2a10 10 0 0 1 10 10"/>
      </svg>
      <span>Evaluating Clinical Risk...</span>
    `;
  });
}

/**
 * Semi-circular Speedometer Gauge Chart (Chart.js)
 */
function initRiskGauge() {
  const canvas = document.getElementById('riskGaugeChart');
  if (!canvas) return;

  const probability = parseFloat(canvas.getAttribute('data-probability') || '0');
  const percent = Math.min(Math.max(probability * 100, 0), 100);

  // Determine active color based on probability
  let needleColor = '#10b981';
  if (percent >= 70) {
    needleColor = '#ef4444';
  } else if (percent >= 30) {
    needleColor = '#f59e0b';
  }

  // Draw gauge chart
  new Chart(canvas.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Risk', 'Remaining'],
      datasets: [
        {
          data: [percent, 100 - percent],
          backgroundColor: [needleColor, '#e2e8f0'],
          borderWidth: 0,
          circumference: 180,
          rotation: -90,
          cutout: '75%',
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: { enabled: false },
        legend: { display: false },
      },
      animation: {
        animateRotate: true,
        duration: 1200,
      },
    },
  });
}

/**
 * Benchmark Comparison Chart for About Page (Chart.js)
 */
function initModelBenchmarkChart(models, accuracies, precisions, recalls, f1s, rocAucs) {
  const canvas = document.getElementById('modelComparisonChart');
  if (!canvas) return;

  new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        {
          label: 'ROC-AUC (%)',
          data: rocAucs,
          backgroundColor: '#dc2626',
          borderRadius: 6,
        },
        {
          label: 'Accuracy (%)',
          data: accuracies,
          backgroundColor: '#3b82f6',
          borderRadius: 6,
        },
        {
          label: 'F1-Score (%)',
          data: f1s,
          backgroundColor: '#10b981',
          borderRadius: 6,
        },
        {
          label: 'Recall (%)',
          data: recalls,
          backgroundColor: '#f59e0b',
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 75,
          max: 100,
          ticks: {
            callback: function (val) {
              return val + '%';
            },
          },
          grid: {
            color: '#f1f5f9',
          },
        },
        x: {
          grid: {
            display: false,
          },
        },
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: { family: 'Plus Jakarta Sans', weight: '600', size: 12 },
            boxWidth: 14,
            usePointStyle: true,
          },
        },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              return `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`;
            },
          },
        },
      },
    },
  });
}
