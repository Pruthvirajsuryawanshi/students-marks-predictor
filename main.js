// main.js – Frontend logic for Student Marks Predictor
// Loads the CSV data, populates the table, and draws a scatter chart.

// Helper: parse CSV string into array of objects
function parseCSV(csvText) {
  const lines = csvText.trim().split(/\r?\n/);
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj = {};
    headers.forEach((h, i) => {
      // Convert numeric fields to numbers
      const val = values[i];
      obj[h] = isNaN(Number(val)) ? val : Number(val);
    });
    return obj;
  });
}

// Render data table
function renderTable(data) {
  const tbody = document.querySelector('#dataTable tbody');
  tbody.innerHTML = '';
  data.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.number_courses}</td>
      <td>${row.time_study}</td>
      <td>${row.Marks}</td>
      <td></td>
    `;
    tbody.appendChild(tr);
  });
}

// Create scatter chart using Chart.js
function createScatterChart(data) {
  const ctx = document.getElementById('scatterChart').getContext('2d');
  const chartData = data.map(d => ({x: d.time_study, y: d.Marks, r: d.number_courses * 2}));
  new Chart(ctx, {
    type: 'bubble', // bubble lets us encode a third variable (courses) via radius
    data: {
      datasets: [{
        label: 'Marks vs Study Hours',
        data: chartData,
        backgroundColor: 'rgba(255, 180, 0, 0.6)',
        borderColor: 'rgba(255, 180, 0, 1)'
      }]
    },
    options: {
      scales: {
        x: {
          title: {display: true, text: 'Study Hours'},
          beginAtZero: true
        },
        y: {
          title: {display: true, text: 'Marks'},
          beginAtZero: true
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => {
              const point = ctx.raw;
              return `Study: ${point.x}, Marks: ${point.y}, Courses: ${Math.round(point.r / 2)}`;
            }
          }
        }
      }
    }
  });
}

// Compute multiple linear regression coefficients (Intercept, coeff for courses, coeff for study)
function computeRegressionCoefficients(data) {
  // X matrix with columns [1, number_courses, time_study]
  const n = data.length;
  let sumX0 = n; // sum of intercept column (all 1s)
  let sumX1 = 0, sumX2 = 0;
  let sumY = 0;
  let sumX1Y = 0, sumX2Y = 0;
  let sumX1X1 = 0, sumX2X2 = 0, sumX1X2 = 0;
  data.forEach(p => {
    const x1 = p.number_courses;
    const x2 = p.time_study;
    const y = p.Marks;
    sumX1 += x1;
    sumX2 += x2;
    sumY += y;
    sumX1Y += x1 * y;
    sumX2Y += x2 * y;
    sumX1X1 += x1 * x1;
    sumX2X2 += x2 * x2;
    sumX1X2 += x1 * x2;
  });
  // Normal equation matrix (3x3)
  const A = [
    [sumX0, sumX1, sumX2],
    [sumX1, sumX1X1, sumX1X2],
    [sumX2, sumX1X2, sumX2X2]
  ];
  const B = [sumY, sumX1Y, sumX2Y];
  // Solve Ax = B using Cramer's rule (small 3x3 system)
  const det = (m) => {
    return m[0][0]* (m[1][1]*m[2][2] - m[1][2]*m[2][1])
         - m[0][1]* (m[1][0]*m[2][2] - m[1][2]*m[2][0])
         + m[0][2]* (m[1][0]*m[2][1] - m[1][1]*m[2][0]);
  };
  const D = det(A);
  if (Math.abs(D) < 1e-12) return null; // singular
  // Replace columns
  const replaceCol = (colIdx, vec) => {
    const m = A.map((row, i) => row.slice());
    m.forEach((row, i) => row[colIdx] = vec[i]);
    return det(m);
  };
  const a0 = replaceCol(0, B) / D; // intercept
  const a1 = replaceCol(1, B) / D; // coeff courses
  const a2 = replaceCol(2, B) / D; // coeff study
  return {intercept: a0, coeffCourses: a1, coeffStudy: a2};
}

function predictMarks(coeffs, courses, study) {
  return coeffs.intercept + coeffs.coeffCourses * courses + coeffs.coeffStudy * study;
}

// Populate predictions in the table
function populatePredictions(coeffs) {
  const rows = document.querySelectorAll('#dataTable tbody tr');
  rows.forEach((tr, idx) => {
    const cells = tr.children;
    const courses = Number(cells[0].textContent);
    const study = Number(cells[1].textContent);
    const pred = predictMarks(coeffs, courses, study).toFixed(2);
    cells[3].textContent = pred;
  });
}

// UI Handlers
function setupUI(coeffs) {
  const predictBtn = document.getElementById('predictBtn');
  const resultSpan = document.getElementById('predictionResult');
  predictBtn.addEventListener('click', () => {
    const courses = Number(document.getElementById('inputCourses').value);
    const study = Number(document.getElementById('inputStudy').value);
    if (isNaN(courses) || isNaN(study)) {
      resultSpan.textContent = 'Please enter valid numbers.';
      return;
    }
    const pred = predictMarks(coeffs, courses, study).toFixed(2);
    resultSpan.textContent = `Predicted Marks: ${pred}`;
  });

  // File upload handling
  const fileInput = document.getElementById('fileInput');
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const csv = event.target.result;
      const newData = parseCSV(csv);
      // Append to existing table
      const tbody = document.querySelector('#dataTable tbody');
      newData.forEach(row => {
        const tr = document.createElement('tr');
        const pred = predictMarks(coeffs, row.number_courses, row.time_study).toFixed(2);
        tr.innerHTML = `
          <td>${row.number_courses}</td>
          <td>${row.time_study}</td>
          <td>${row.Marks}</td>
          <td>${pred}</td>
        `;
        tbody.appendChild(tr);
      });
    };
    reader.readAsText(file);
  });
}

// Main execution
fetch('Student_Marks.csv')
  .then(res => res.text())
  .then(csv => {
    const data = parseCSV(csv);
    renderTable(data);
    createScatterChart(data);
    const coeffs = computeRegressionCoefficients(data);
    if (coeffs) {
      populatePredictions(coeffs);
      setupUI(coeffs);
    } else {
      console.error('Regression could not be computed.');
    }
  })
  .catch(err => console.error('Failed to load CSV:', err));
