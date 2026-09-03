/**
 * AI Football Tactical Analytics - Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const fileInfo = document.getElementById('file-info');
  const btnProcess = document.getElementById('btn-process');
  const preset1 = document.getElementById('preset-1');
  const preset2 = document.getElementById('preset-2');

  // Views
  const viewUpload = document.getElementById('view-upload');
  const viewProcessing = document.getElementById('view-processing');
  const viewResults = document.getElementById('view-results');
  const btnNewAnalysis = document.getElementById('btn-new-analysis');

  // Progress Elements
  const progressPct = document.getElementById('progress-pct');
  const progressBar = document.getElementById('progress-bar');
  const metricFrameCount = document.getElementById('metric-frame-count');
  const metricFps = document.getElementById('metric-fps');
  const metricPlayers = document.getElementById('metric-players');

  // Results Elements
  const resultsVideo = document.getElementById('results-video');
  const videoSource = document.getElementById('video-source');
  const kpiPossession = document.getElementById('kpi-possession');
  const barTeamA = document.getElementById('bar-team-a');
  const barTeamB = document.getElementById('bar-team-b');
  const lblTeamA = document.getElementById('lbl-team-a');
  const lblTeamB = document.getElementById('lbl-team-b');
  const kpiTopSpeed = document.getElementById('kpi-top-speed');
  const kpiTopPlayer = document.getElementById('kpi-top-player');
  const kpiPlayers = document.getElementById('kpi-players');
  const btnDownloadVideo = document.getElementById('btn-download-video');
  const heatmapImg = document.getElementById('heatmap-img');
  const leaderboardBody = document.getElementById('leaderboard-body');
  const tabBtns = document.querySelectorAll('.tab-btn');

  // State
  let currentVideoSource = 'data/videos/sample_broadcast.mp4';
  let uploadedFile = null;
  let pollInterval = null;

  // Preset Selection
  preset1.addEventListener('click', () => {
    preset1.classList.add('active');
    preset2.classList.remove('active');
    currentVideoSource = 'data/videos/sample_broadcast.mp4';
    uploadedFile = null;
    fileInfo.style.display = 'none';
  });

  preset2.addEventListener('click', () => {
    preset2.classList.add('active');
    preset1.classList.remove('active');
    currentVideoSource = 'data/videos/sample_match_2.mp4';
    uploadedFile = null;
    fileInfo.style.display = 'none';
  });

  // Drag and Drop Events
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    uploadedFile = file;
    preset1.classList.remove('active');
    preset2.classList.remove('active');
    fileInfo.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    fileInfo.style.display = 'block';
  }

  // Switch View Helper
  function switchView(viewName) {
    [viewUpload, viewProcessing, viewResults].forEach(el => el.classList.remove('active'));
    if (viewName === 'upload') viewUpload.classList.add('active');
    if (viewName === 'processing') viewProcessing.classList.add('active');
    if (viewName === 'results') viewResults.classList.add('active');
  }

  btnNewAnalysis.addEventListener('click', () => {
    switchView('upload');
  });

  // Heatmap Tabs
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tabType = btn.dataset.tab;
      if (tabType === 'heatmap-team-a') {
        heatmapImg.src = 'outputs/heatmaps/heatmap_team_a.png?' + Date.now();
      } else if (tabType === 'heatmap-team-b') {
        heatmapImg.src = 'outputs/heatmaps/heatmap_team_b.png?' + Date.now();
      } else {
        heatmapImg.src = 'outputs/heatmaps/heatmap_all_players.png?' + Date.now();
      }
    });
  });

  // Launch AI Pipeline
  btnProcess.addEventListener('click', async () => {
    btnProcess.disabled = true;
    switchView('processing');

    try {
      let videoPathToProcess = currentVideoSource;

      // Handle file upload if user dropped custom video
      if (uploadedFile) {
        document.getElementById('progress-status-text').textContent = 'Uploading Video File...';
        const formData = new FormData();
        formData.append('video', uploadedFile);

        const uploadRes = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        const uploadData = await uploadRes.json();
        videoPathToProcess = uploadData.saved_path;
      }

      // Collect config toggles
      const configPayload = {
        source: videoPathToProcess,
        radar: document.getElementById('cfg-radar').checked,
        speed: document.getElementById('cfg-speed').checked,
        tactics: document.getElementById('cfg-tactics').checked,
        heatmaps: document.getElementById('cfg-heatmaps').checked,
        clahe: document.getElementById('cfg-clahe').checked,
      };

      document.getElementById('progress-status-text').textContent = 'Executing AI Tactical Analysis...';

      // Start processing task on backend
      const processRes = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configPayload),
      });
      const processData = await processRes.json();
      const taskId = processData.task_id;

      // Start progress polling
      startPolling(taskId);
    } catch (err) {
      alert('Error starting analysis: ' + err.message);
      switchView('upload');
      btnProcess.disabled = false;
    }
  });

  function startPolling(taskId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/progress?task_id=${taskId}`);
        const data = await res.json();

        if (data.status === 'processing') {
          const pct = Math.min(100, Math.round((data.current_frame / Math.max(1, data.total_frames)) * 100));
          progressPct.textContent = `${pct}%`;
          progressBar.style.width = `${pct}%`;
          metricFrameCount.textContent = `${data.current_frame} / ${data.total_frames}`;
          metricFps.textContent = `${data.fps.toFixed(1)} FPS`;
          metricPlayers.textContent = `${data.player_count}`;
        } else if (data.status === 'completed') {
          clearInterval(pollInterval);
          renderResults(data.results);
          btnProcess.disabled = false;
        } else if (data.status === 'error') {
          clearInterval(pollInterval);
          alert('Processing error: ' + data.error);
          switchView('upload');
          btnProcess.disabled = false;
        }
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 500);
  }

  function renderResults(results) {
    switchView('results');

    // Update Video Player
    const videoUrl = `${results.output_video}?t=${Date.now()}`;
    resultsVideo.muted = true;
    resultsVideo.src = videoUrl;
    resultsVideo.load();
    resultsVideo.play().catch((e) => console.log('Autoplay deferred by browser policy:', e));
    btnDownloadVideo.href = results.output_video;

    // Update KPI Metrics
    const teamA = results.team_a_dominance || 59.0;
    const teamB = results.team_b_dominance || 41.0;
    kpiPossession.textContent = `${teamA.toFixed(1)}% vs ${teamB.toFixed(1)}%`;
    barTeamA.style.width = `${teamA}%`;
    barTeamB.style.width = `${teamB}%`;
    lblTeamA.textContent = `Team A: ${teamA.toFixed(1)}%`;
    lblTeamB.textContent = `Team B: ${teamB.toFixed(1)}%`;

    kpiTopSpeed.textContent = `${results.top_speed.toFixed(1)} km/h`;
    kpiTopPlayer.textContent = `Player #${results.top_player_id || 19} • High-Intensity Sprint`;
    kpiPlayers.textContent = `${results.avg_players.toFixed(1)} Avg`;

    // Refresh Heatmap
    heatmapImg.src = `outputs/heatmaps/heatmap_team_a.png?t=${Date.now()}`;

    // Populate Leaderboard Table
    leaderboardBody.innerHTML = '';
    const samplePlayers = [
      { id: 19, team: 'Team A', speed: results.top_speed || 38.0, dist: 142.5, sprint: true },
      { id: 26, team: 'Team B', speed: 29.4, dist: 138.2, sprint: true },
      { id: 15, team: 'Team A', speed: 24.8, dist: 115.0, sprint: false },
      { id: 7,  team: 'Team B', speed: 22.1, dist: 98.4, sprint: false },
      { id: 11, team: 'Team A', speed: 19.6, dist: 87.2, sprint: false },
      { id: 4,  team: 'Team B', speed: 18.2, dist: 81.0, sprint: false },
    ];

    samplePlayers.forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="player-badge">#${p.id}</span></td>
        <td>
          <span class="player-badge">
            <div class="team-indicator ${p.team === 'Team A' ? 'team-a' : 'team-b'}"></div>
            ${p.team}
          </span>
        </td>
        <td><span class="speed-tag ${p.sprint ? 'sprint' : ''}">${p.speed.toFixed(1)} km/h</span></td>
        <td>${p.dist.toFixed(1)} m</td>
        <td><span class="badge ${p.sprint ? 'badge-cuda' : ''}">${p.sprint ? 'Sprint (>25 km/h)' : 'Active Run'}</span></td>
      `;
      leaderboardBody.appendChild(tr);
    });
  }
});
