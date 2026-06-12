// Global App State
const state = {
    dashboardStats: {
        total: 0,
        positive: 0,
        neutral: 0,
        negative: 0,
        score: 0.0,
        top_words: { positive: [], negative: [] }
    },
    // Chart.js references to prevent canvas reuse errors
    charts: {
        dashDist: null,
        dashGauge: null,
        sandboxVader: null,
        sandboxML: null,
        sandboxProbs: null,
        bulkDist: null
    },
    currentFile: null
};

// Initialize Application on DOM Load
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initSandbox();
    initBulkUpload();
    fetchModelInfo();
    
    // Set initial mock dashboard values
    updateDashboardUI();
});

// 1. NAVIGATION CONTROL
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const pages = document.querySelectorAll(".view-page");
    const pageTitle = document.getElementById("page-title");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            
            // Remove active class from all nav items
            navItems.forEach(nav => nav.classList.remove("active"));
            
            // Add active class to clicked item
            item.classList.add("active");
            
            // Show corresponding page
            const targetPageId = item.getAttribute("data-target");
            pages.forEach(page => {
                if (page.id === targetPageId) {
                    page.classList.add("active");
                } else {
                    page.classList.remove("active");
                }
            });
            
            // Update Page Title
            if (item.querySelector("span")) {
                pageTitle.innerText = item.querySelector("span").innerText;
            }
        });
    });
}

// 2. REAL-TIME SANDBOX LOGIC
function initSandbox() {
    const textInput = document.getElementById("sandbox-text");
    const charCount = document.getElementById("char-count");
    const btnClear = document.getElementById("btn-clear-sandbox");
    let debounceTimer;

    textInput.addEventListener("input", (e) => {
        const text = e.target.value;
        charCount.innerText = `${text.length} characters`;
        
        clearTimeout(debounceTimer);
        
        if (!text.trim()) {
            resetSandboxUI();
            return;
        }

        debounceTimer = setTimeout(() => {
            analyzeSingle(text);
        }, 400); // 400ms debounce
    });

    btnClear.addEventListener("click", () => {
        textInput.value = "";
        charCount.innerText = "0 characters";
        resetSandboxUI();
    });

    // Initialize blank charts for Sandbox
    initSandboxCharts();
}

async function analyzeSingle(text) {
    try {
        const response = await fetch("/api/analyze/single", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) throw new Error("API analysis failed");
        const data = await response.json();
        
        updateSandboxUI(data);
        
        // Add to Dashboard aggregates dynamically to feel alive
        updateDashboardAggregates(data);
        
    } catch (error) {
        console.error("Single analysis error:", error);
    }
}

function updateSandboxUI(data) {
    // 1. Update Pipeline Preprocessing steps
    const prep = data.preprocessing;
    document.getElementById("step-lower").innerText = prep.step1_lowercase || "-";
    document.getElementById("step-tokens").innerText = JSON.stringify(prep.step2_tokenized) || "[]";
    document.getElementById("step-nopunct").innerText = JSON.stringify(prep.step3_no_punctuation) || "[]";
    document.getElementById("step-nostop").innerText = JSON.stringify(prep.step4_no_stopwords) || "[]";
    document.getElementById("step-lemma").innerText = prep.final_cleaned || "-";
    
    // 2. Update VADER Card details
    const vader = data.vader;
    const compound = vader.compound;
    document.getElementById("vader-compound-val").innerText = compound.toFixed(2);
    
    const vBadge = document.getElementById("vader-badge");
    vBadge.className = `sentiment-badge ${vader.sentiment.substring(0, 3)}`;
    vBadge.innerText = vader.sentiment;
    
    // Update VADER Gauge chart
    updateHalfGauge(state.charts.sandboxVader, (compound + 1) / 2); // Map -1..1 to 0..1
    
    // 3. Update ML Model Card details
    const ml = data.machine_learning;
    const mlConfidence = Math.round(ml.confidence * 100);
    document.getElementById("ml-confidence-val").innerText = `${mlConfidence}%`;
    
    const mlBadge = document.getElementById("ml-badge");
    mlBadge.className = `sentiment-badge ${ml.sentiment.substring(0, 3)}`;
    mlBadge.innerText = ml.sentiment;
    
    // Update ML Gauge chart
    updateHalfGauge(state.charts.sandboxML, ml.confidence);
    
    // 4. Update Probability distribution bar chart
    const probs = [
        Math.round(ml.probabilities.positive * 100),
        Math.round(ml.probabilities.neutral * 100),
        Math.round(ml.probabilities.negative * 100)
    ];
    state.charts.sandboxProbs.data.datasets[0].data = probs;
    state.charts.sandboxProbs.update();
}

function resetSandboxUI() {
    // Clear trace table
    document.getElementById("step-lower").innerText = "-";
    document.getElementById("step-tokens").innerText = "-";
    document.getElementById("step-nopunct").innerText = "-";
    document.getElementById("step-nostop").innerText = "-";
    document.getElementById("step-lemma").innerText = "-";
    
    // Clear VADER details
    document.getElementById("vader-compound-val").innerText = "0.00";
    const vBadge = document.getElementById("vader-badge");
    vBadge.className = "sentiment-badge neu";
    vBadge.innerText = "Neutral";
    updateHalfGauge(state.charts.sandboxVader, 0.5); // centered
    
    // Clear ML details
    document.getElementById("ml-confidence-val").innerText = "0%";
    const mlBadge = document.getElementById("ml-badge");
    mlBadge.className = "sentiment-badge neu";
    mlBadge.innerText = "Neutral";
    updateHalfGauge(state.charts.sandboxML, 0.33); // baseline probability
    
    // Clear probs chart
    state.charts.sandboxProbs.data.datasets[0].data = [0, 0, 0];
    state.charts.sandboxProbs.update();
}

// 3. BULK CSV UPLOAD LOGIC
function initBulkUpload() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("csv-file-input");
    const btnRemove = document.getElementById("btn-remove-file");
    const btnProcess = document.getElementById("btn-process-bulk");
    
    // Drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleSelectedFile(files[0]);
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) handleSelectedFile(e.target.files[0]);
    });
    
    btnRemove.addEventListener("click", () => {
        state.currentFile = null;
        fileInput.value = "";
        document.getElementById("file-info-bar").style.display = "none";
        document.getElementById("column-config-panel").style.display = "none";
        document.getElementById("bulk-results-section").style.display = "none";
        dropZone.style.display = "flex";
    });
    
    btnProcess.addEventListener("click", () => {
        if (state.currentFile) processBulkCSV();
    });
}

function handleSelectedFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert("Please upload a CSV file only.");
        return;
    }
    
    state.currentFile = file;
    
    // Display File Details Bar
    document.getElementById("file-name").innerText = file.name;
    document.getElementById("file-size").innerText = `(${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById("file-info-bar").style.display = "flex";
    document.getElementById("drop-zone").style.display = "none";
    
    // Parse headers on client using FileReader for instant feedback
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const firstLine = text.split('\n')[0];
        const headers = firstLine.split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
        
        const colSelect = document.getElementById("csv-text-column");
        colSelect.innerHTML = "";
        
        // Auto-select text columns using a scoring system
        let matchedIndex = 0;
        let highestScore = -1;
        
        headers.forEach((h, index) => {
            if (!h) return;
            const option = document.createElement("option");
            option.value = h;
            option.innerText = h;
            colSelect.appendChild(option);
            
            const colLower = h.toLowerCase();
            // Filter out ID, time, and date columns
            if (colLower.endsWith('id') || colLower.includes('_id') || colLower.includes('date') || colLower.includes('time')) {
                return;
            }
            
            let score = 0;
            if (colLower.includes('text')) score += 10;
            if (colLower.includes('review')) score += 10;
            if (colLower.includes('comment')) score += 10;
            if (colLower.includes('feedback')) score += 8;
            if (colLower.includes('message')) score += 8;
            if (colLower.includes('body')) score += 8;
            if (colLower.includes('content')) score += 8;
            
            if (score > highestScore) {
                highestScore = score;
                matchedIndex = index;
            }
        });
        
        colSelect.selectedIndex = matchedIndex;
        document.getElementById("column-config-panel").style.display = "flex";
    };
    reader.readAsText(file.slice(0, 4096)); // read first 4KB to extract headers
}

async function processBulkCSV() {
    const file = state.currentFile;
    const colName = document.getElementById("csv-text-column").value;
    const progressWrapper = document.getElementById("bulk-progress-wrapper");
    const progressBar = document.getElementById("bulk-progress");
    
    progressWrapper.style.display = "block";
    progressBar.style.width = "40%";
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("column", colName);
    
    try {
        progressBar.style.width = "70%";
        
        const response = await fetch("/api/analyze/bulk", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Bulk processing failed");
        }
        
        progressBar.style.width = "100%";
        const summary = await response.json();
        
        setTimeout(() => {
            progressWrapper.style.display = "none";
            progressBar.style.width = "0%";
            displayBulkResults(summary);
        }, 300);
        
    } catch (error) {
        progressBar.style.width = "0%";
        progressWrapper.style.display = "none";
        alert(error.message);
    }
}

async function displayBulkResults(summary) {
    // 1. Show results section
    document.getElementById("bulk-results-section").style.display = "block";
    
    // 2. Populate stats indicators
    const posPercent = Math.round((summary.vader_distribution.positive / summary.total_records) * 100) || 0;
    const negPercent = Math.round((summary.vader_distribution.negative / summary.total_records) * 100) || 0;
    
    document.getElementById("bulk-total").innerText = summary.total_records;
    document.getElementById("bulk-pos").innerText = `${posPercent}%`;
    document.getElementById("bulk-pos-count").innerText = `${summary.vader_distribution.positive} reviews`;
    document.getElementById("bulk-neg").innerText = `${negPercent}%`;
    document.getElementById("bulk-neg-count").innerText = `${summary.vader_distribution.negative} reviews`;
    document.getElementById("bulk-score").innerText = summary.average_sentiment_score.toFixed(2);
    document.getElementById("bulk-agreement-rate").innerText = `Model Agreement: ${Math.round(summary.model_agreement_rate * 100)}%`;
    
    // 3. Setup export download button link
    const downloadBtn = document.getElementById("btn-download-results");
    downloadBtn.href = `/api/export/${summary.job_id}`;
    
    // 4. Update the bulk doughnut distribution chart
    const distData = [
        summary.vader_distribution.positive,
        summary.vader_distribution.neutral,
        summary.vader_distribution.negative
    ];
    initBulkChart(distData);
    
    // 5. Populate top terms list bars
    populateKeywordLists(summary.top_words, "bulk-pos-words", "bulk-neg-words");
    
    // 6. Fetch preview table lines from server using the exported file
    fetchPreviewTable(summary.job_id, summary.detected_column);
    
    // 7. Sync dashboard figures to show these CSV metrics
    syncCSVStatsToDashboard(summary);
}

async function fetchPreviewTable(jobId, textColumnName) {
    try {
        const response = await fetch(`/api/export/${jobId}`);
        if (!response.ok) return;
        
        const csvText = await response.text();
        const lines = csvText.split(/\r?\n/).filter(l => l.trim() !== "");
        const headers = lines[0].split(',');
        
        // Find indices of needed columns
        const textIdx = headers.indexOf(textColumnName);
        const compIdx = headers.indexOf("VADER_Compound");
        const vSentIdx = headers.indexOf("VADER_Sentiment");
        const mSentIdx = headers.indexOf("ML_Sentiment");
        
        const tbody = document.querySelector("#bulk-preview-table tbody");
        tbody.innerHTML = "";
        
        // Show up to 100 rows in preview
        const rowsToShow = Math.min(lines.length - 1, 100);
        
        for (let i = 1; i <= rowsToShow; i++) {
            // Helper to handle commas inside quotes in CSV line parsing
            const fields = parseCSVLine(lines[i]);
            if (fields.length <= Math.max(textIdx, compIdx, vSentIdx, mSentIdx)) continue;
            
            const tr = document.createElement("tr");
            
            const tdText = document.createElement("td");
            tdText.className = "text-truncate-cell";
            tdText.innerText = fields[textIdx] || "";
            tdText.title = fields[textIdx] || "";
            tr.appendChild(tdText);
            
            const tdComp = document.createElement("td");
            tdComp.innerText = parseFloat(fields[compIdx]).toFixed(2) || "0.00";
            tr.appendChild(tdComp);
            
            const tdVader = document.createElement("td");
            const vVal = fields[vSentIdx] ? fields[vSentIdx].toLowerCase() : "neutral";
            tdVader.innerHTML = `<span class="sentiment-badge ${vVal.substring(0, 3)}" style="padding: 0.15rem 0.5rem; font-size: 0.7rem;">${fields[vSentIdx]}</span>`;
            tr.appendChild(tdVader);
            
            const tdML = document.createElement("td");
            const mlVal = fields[mSentIdx] ? fields[mSentIdx].toLowerCase() : "neutral";
            tdML.innerHTML = `<span class="sentiment-badge ${mlVal.substring(0, 3)}" style="padding: 0.15rem 0.5rem; font-size: 0.7rem;">${fields[mSentIdx]}</span>`;
            tr.appendChild(tdML);
            
            tbody.appendChild(tr);
        }
        
    } catch (e) {
        console.error("Preview table fetch error:", e);
    }
}

// Simple csv line parser helper to handle quoted values containing commas
function parseCSVLine(line) {
    const result = [];
    let insideQuote = false;
    let entry = "";
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
            insideQuote = !insideQuote;
        } else if (char === ',' && !insideQuote) {
            result.push(entry.trim().replace(/^["']|["']$/g, ''));
            entry = "";
        } else {
            entry += char;
        }
    }
    result.push(entry.trim().replace(/^["']|["']$/g, ''));
    return result;
}

// 4. MODEL DETAILS VIEW
async function fetchModelInfo() {
    try {
        const response = await fetch("/api/model-info");
        if (!response.ok) return;
        const details = await response.json();
        
        // Update Accuracy Metrics
        const accPct = Math.round(details.accuracy * 100);
        document.getElementById("model-accuracy").innerText = `${accPct}%`;
        document.getElementById("model-samples").innerText = details.training_samples;
        
        // Draw weights lists
        const container = document.getElementById("model-features-container");
        container.innerHTML = "";
        
        const classes = details.classes || ["positive", "neutral", "negative"];
        
        classes.forEach(cls => {
            const classFeatures = details.top_features[cls] || [];
            
            const card = document.createElement("div");
            card.className = "card feature-class-card";
            
            const labelShort = cls.substring(0, 3);
            
            let icon = "fa-face-smile";
            if (cls === "negative") icon = "fa-face-frown";
            if (cls === "neutral") icon = "fa-face-meh";
            
            card.innerHTML = `
                <div class="feature-class-title ${labelShort}">
                    <span><i class="fa-solid ${icon}"></i> Class: ${cls.toUpperCase()}</span>
                    <span>Weights</span>
                </div>
                <div class="feature-list">
                    ${classFeatures.map(f => `
                        <div class="feature-item">
                            <span>${f.word}</span>
                            <span class="feature-weight">${f.weight.toFixed(4)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
            
            container.appendChild(card);
        });
        
    } catch (e) {
        console.error("Model info fetch failed:", e);
    }
}

// 5. SYNCHRONIZE METRICS TO DASHBOARD
function syncCSVStatsToDashboard(summary) {
    state.dashboardStats.total = summary.total_records;
    state.dashboardStats.positive = summary.vader_distribution.positive;
    state.dashboardStats.neutral = summary.vader_distribution.neutral;
    state.dashboardStats.negative = summary.vader_distribution.negative;
    state.dashboardStats.score = summary.average_sentiment_score;
    state.dashboardStats.top_words = summary.top_words;
    
    updateDashboardUI();
}

function updateDashboardAggregates(singleData) {
    // Dynamic integration of sandbox typing actions into overall stats
    state.dashboardStats.total += 1;
    const sent = singleData.vader.sentiment;
    state.dashboardStats[sent] += 1;
    
    // Recalculate rolling average score
    const total = state.dashboardStats.total;
    const currentScore = state.dashboardStats.score;
    const newCompound = singleData.vader.compound;
    state.dashboardStats.score = currentScore + (newCompound - currentScore) / total;
    
    // Add dynamic words
    const tokens = singleData.preprocessing.step5_lemmatized;
    const wordList = sent === "positive" ? state.dashboardStats.top_words.positive : state.dashboardStats.top_words.negative;
    
    tokens.forEach(t => {
        const found = wordList.find(w => w.word === t);
        if (found) {
            found.count += 1;
        } else {
            wordList.push({ word: t, count: 1 });
        }
    });
    
    // Sort words
    wordList.sort((a,b) => b.count - a.count);
    if (wordList.length > 10) wordList.length = 10;
    
    updateDashboardUI();
}

function updateDashboardUI() {
    const stats = state.dashboardStats;
    const total = stats.total;
    
    // Calculate percentages
    const posPct = total > 0 ? Math.round((stats.positive / total) * 100) : 0;
    const neuPct = total > 0 ? Math.round((stats.neutral / total) * 100) : 0;
    const negPct = total > 0 ? Math.round((stats.negative / total) * 100) : 0;
    
    // Update elements
    document.getElementById("dash-total").innerText = total;
    document.getElementById("dash-pos").innerText = `${posPct}%`;
    document.getElementById("dash-neu").innerText = `${neuPct}%`;
    document.getElementById("dash-neg").innerText = `${negPct}%`;
    
    document.getElementById("sat-score-val").innerText = stats.score.toFixed(2);
    
    // Draw/update dashboard charts
    initDashboardCharts([stats.positive, stats.neutral, stats.negative], stats.score);
    
    // Draw top keywords
    const posWordsContainer = document.getElementById("pos-word-list");
    const negWordsContainer = document.getElementById("neg-word-list");
    
    if (total === 0) {
        posWordsContainer.innerHTML = `<div style="color: var(--text-secondary); font-size: 0.875rem;">No stats loaded yet.</div>`;
        negWordsContainer.innerHTML = `<div style="color: var(--text-secondary); font-size: 0.875rem;">No stats loaded yet.</div>`;
    } else {
        populateKeywordLists(stats.top_words, "pos-word-list", "neg-word-list");
    }
}

function populateKeywordLists(topWords, posContainerId, negContainerId) {
    const posContainer = document.getElementById(posContainerId);
    const negContainer = document.getElementById(negContainerId);
    
    const maxVal = Math.max(
        ...(topWords.positive.map(w => w.count) || [1]), 
        ...(topWords.negative.map(w => w.count) || [1]),
        1
    );
    
    posContainer.innerHTML = topWords.positive.length === 0 ? '<span style="color: var(--text-secondary); font-size:0.85rem;">None</span>' : "";
    topWords.positive.slice(0, 7).forEach(item => {
        const pct = Math.round((item.count / maxVal) * 100);
        posContainer.innerHTML += `
            <div class="word-row">
                <div class="word-label">${item.word}</div>
                <div class="word-bar-wrapper">
                    <div class="word-bar pos" style="width: ${pct}%"></div>
                </div>
                <div class="word-count">${item.count}</div>
            </div>
        `;
    });
    
    negContainer.innerHTML = topWords.negative.length === 0 ? '<span style="color: var(--text-secondary); font-size:0.85rem;">None</span>' : "";
    topWords.negative.slice(0, 7).forEach(item => {
        const pct = Math.round((item.count / maxVal) * 100);
        negContainer.innerHTML += `
            <div class="word-row">
                <div class="word-label">${item.word}</div>
                <div class="word-bar-wrapper">
                    <div class="word-bar neg" style="width: ${pct}%"></div>
                </div>
                <div class="word-count">${item.count}</div>
            </div>
        `;
    });
}

// 6. CHART.JS DRAWING HANDLERS
function initDashboardCharts(distributionData, satisfactionScore) {
    // 1. Doughnut chart
    const distCanvas = document.getElementById("distribution-chart");
    if (state.charts.dashDist) {
        state.charts.dashDist.data.datasets[0].data = distributionData;
        state.charts.dashDist.update();
    } else {
        state.charts.dashDist = new Chart(distCanvas, {
            type: "doughnut",
            data: {
                labels: ["Positive", "Neutral", "Negative"],
                datasets: [{
                    data: distributionData.every(v => v === 0) ? [1,1,1] : distributionData, // avoid blank chart initially
                    backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
                    borderColor: "#0f172a",
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { color: "#94a3b8", font: { family: "Outfit" } }
                    }
                },
                cutout: "65%"
            }
        });
    }
    
    // 2. Gauge chart
    const gaugeCanvas = document.getElementById("satisfaction-gauge");
    const mappedScore = (satisfactionScore + 1) / 2; // Map -1..1 to 0..1
    if (state.charts.dashGauge) {
        updateHalfGauge(state.charts.dashGauge, mappedScore);
    } else {
        state.charts.dashGauge = createHalfGaugeChart(gaugeCanvas, mappedScore, ["#ef4444", "#f59e0b", "#10b981"]);
    }
}

function initSandboxCharts() {
    const vaderCanvas = document.getElementById("vader-gauge");
    const mlCanvas = document.getElementById("ml-gauge");
    const probsCanvas = document.getElementById("ml-probs-chart");
    
    state.charts.sandboxVader = createHalfGaugeChart(vaderCanvas, 0.5, ["#ef4444", "#f59e0b", "#10b981"]);
    state.charts.sandboxML = createHalfGaugeChart(mlCanvas, 0.33, ["#ef4444", "#f59e0b", "#10b981"]);
    
    // Horizontal bars for probabilities
    state.charts.sandboxProbs = new Chart(probsCanvas, {
        type: "bar",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ["rgba(16, 185, 129, 0.85)", "rgba(245, 158, 11, 0.85)", "rgba(239, 68, 68, 0.85)"],
                borderColor: ["#10b981", "#f59e0b", "#ef4444"],
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(255,255,255,0.03)" },
                    ticks: { color: "#94a3b8", font: { family: "Outfit" } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f8fafc", font: { family: "Outfit", weight: "500" } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function initBulkChart(distributionData) {
    const canvas = document.getElementById("bulk-dist-chart");
    
    if (state.charts.bulkDist) {
        state.charts.bulkDist.data.datasets[0].data = distributionData;
        state.charts.bulkDist.update();
    } else {
        state.charts.bulkDist = new Chart(canvas, {
            type: "doughnut",
            data: {
                labels: ["Positive", "Neutral", "Negative"],
                datasets: [{
                    data: distributionData,
                    backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
                    borderColor: "#0f172a",
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                cutout: "70%"
            }
        });
    }
}

// Reusable Half-Doughnut gauge chart creator
function createHalfGaugeChart(canvas, valNormalized, colors) {
    const fillVal = valNormalized * 100;
    const remaining = 100 - fillVal;
    
    return new Chart(canvas, {
        type: "doughnut",
        data: {
            datasets: [{
                data: [fillVal, remaining],
                backgroundColor: [colors[2], "rgba(255,255,255,0.03)"],
                borderWidth: 0
            }]
        },
        options: {
            rotation: -90,
            circumference: 180,
            cutout: "82%",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            }
        }
    });
}

function updateHalfGauge(chart, valNormalized) {
    if (!chart) return;
    const fillVal = Math.max(0, Math.min(100, valNormalized * 100));
    const remaining = 100 - fillVal;
    
    // Choose dynamic color depending on score thresholds
    let activeColor = "#f59e0b"; // Amber (neutral)
    if (valNormalized >= 0.55) {
        activeColor = "#10b981"; // Green (positive)
    } else if (valNormalized <= 0.45) {
        activeColor = "#ef4444"; // Red (negative)
    }
    
    chart.data.datasets[0].backgroundColor[0] = activeColor;
    chart.data.datasets[0].data = [fillVal, remaining];
    chart.update();
}
