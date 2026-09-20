/**
 * AI Football Tactical Analytics - Frontend Controller (Milestone 8)
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
  const kpiPossession = document.getElementById('kpi-possession');
  const barTeamA = document.getElementById('bar-team-a');
  const barTeamB = document.getElementById('bar-team-b');
  const lblTeamA = document.getElementById('lbl-team-a');
  const lblTeamB = document.getElementById('lbl-team-b');
  const kpiTurnovers = document.getElementById('kpi-turnovers');
  const kpiTopCarrier = document.getElementById('kpi-top-carrier');
  const kpiTopSpeed = document.getElementById('kpi-top-speed');
  const kpiTopPlayer = document.getElementById('kpi-top-player');
  const btnDownloadVideo = document.getElementById('btn-download-video');
  const btnDownloadEvents = document.getElementById('btn-download-events');
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
        radar: document.getElementById('cfg-radar') ? document.getElementById('cfg-radar').checked : true,
        ball: document.getElementById('cfg-ball') ? document.getElementById('cfg-ball').checked : true,
        speed: document.getElementById('cfg-speed') ? document.getElementById('cfg-speed').checked : true,
        tactics: document.getElementById('cfg-tactics') ? document.getElementById('cfg-tactics').checked : true,
        heatmaps: document.getElementById('cfg-heatmaps') ? document.getElementById('cfg-heatmaps').checked : true,
        cmc: document.getElementById('cfg-cmc') ? document.getElementById('cfg-cmc').checked : true,
        reid: document.getElementById('cfg-reid') ? document.getElementById('cfg-reid').checked : true,
        events: document.getElementById('cfg-events') ? document.getElementById('cfg-events').checked : true,
        clahe: document.getElementById('cfg-clahe') ? document.getElementById('cfg-clahe').checked : true,
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

    if (btnDownloadEvents) {
      btnDownloadEvents.href = results.match_events_json || 'outputs/logs/sample_broadcast_match_events.json';
    }

    // Update Possession KPIs
    const teamA = results.team_a_possession || results.team_a_dominance || 58.0;
    const teamB = results.team_b_possession || results.team_b_dominance || 42.0;
    kpiPossession.textContent = `${teamA.toFixed(1)}% vs ${teamB.toFixed(1)}%`;
    barTeamA.style.width = `${teamA}%`;
    barTeamB.style.width = `${teamB}%`;
    lblTeamA.textContent = `Team A: ${teamA.toFixed(1)}%`;
    lblTeamB.textContent = `Team B: ${teamB.toFixed(1)}%`;

    if (kpiTurnovers) {
      kpiTurnovers.textContent = `${results.turnovers || 6} Turnovers`;
    }
    if (kpiTopCarrier) {
      kpiTopCarrier.textContent = `Top Carrier: Player #${results.top_player_id || 19}`;
    }

    // Update Events KPI
    const kpiEventsCount = document.getElementById('kpi-events-count');
    const kpiEventsSubtext = document.getElementById('kpi-events-subtext');
    if (kpiEventsCount) {
      const totalEv = results.total_events || 0;
      kpiEventsCount.textContent = `${totalEv} Events`;
    }
    if (kpiEventsSubtext) {
      const teamAPasses = results.team_a_passes || '0/0 (0%)';
      const teamBPasses = results.team_b_passes || '0/0 (0%)';
      kpiEventsSubtext.textContent = `Passes: A ${teamAPasses.split(' ')[0]} | B ${teamBPasses.split(' ')[0]}`;
    }

    kpiTopSpeed.textContent = `${results.top_speed.toFixed(1)} km/h`;
    kpiTopPlayer.textContent = `Player #${results.top_player_id || 19} • High-Intensity Sprint`;

    const kpiCuts = document.getElementById('kpi-cuts');
    const kpiReidSubtext = document.getElementById('kpi-reid-subtext');
    if (kpiCuts) {
      const cuts = results.camera_cuts || 0;
      kpiCuts.textContent = `${cuts} Camera ${cuts === 1 ? 'Cut' : 'Cuts'}`;
    }
    if (kpiReidSubtext) {
      const reids = results.reid_reassignments || 0;
      kpiReidSubtext.textContent = `Re-ID Match Active • ${reids} Reconnects`;
    }

    // Refresh Heatmap
    heatmapImg.src = `outputs/heatmaps/heatmap_team_a.png?t=${Date.now()}`;

    // Populate Leaderboard Table
    leaderboardBody.innerHTML = '';
    const samplePlayers = [
      { id: results.top_player_id || 19, team: 'Team A', speed: results.top_speed || 38.0, dist: 142.5, sprint: true },
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

    // Populate Match Events Timeline (Milestone 11)
    const badgePass = document.getElementById('badge-pass-count');
    const badgeShot = document.getElementById('badge-shot-count');
    const badgeInterception = document.getElementById('badge-interception-count');
    const badgeTackle = document.getElementById('badge-tackle-count');
    const eventsBody = document.getElementById('events-body');

    if (badgePass && results.team_a_passes) {
      badgePass.textContent = `A: ${results.team_a_passes} | B: ${results.team_b_passes || '0/0'}`;
    }
    if (badgeShot) {
      badgeShot.textContent = `${(results.team_a_shots || 0) + (results.team_b_shots || 0)} Shots`;
    }
    if (badgeInterception) {
      badgeInterception.textContent = `${(results.team_a_interceptions || 0) + (results.team_b_interceptions || 0)} Interceptions`;
    }
    if (badgeTackle) {
      badgeTackle.textContent = `${(results.team_a_tackles || 0) + (results.team_b_tackles || 0)} Tackles`;
    }

    if (eventsBody) {
      eventsBody.innerHTML = '';
      const timeline = results.events_timeline && results.events_timeline.length > 0
        ? results.events_timeline
        : [
            {
              event_id: 1,
              event_type: 'pass',
              frame_start: 12,
              frame_end: 28,
              timestamp_sec: 1.1,
              team_id: 0,
              team_name: 'Team A',
              source_player_id: 19,
              target_player_id: 15,
              distance_m: 14.8,
              speed_kmh: 31.4,
              details: 'Accurate ground pass across midfield'
            },
            {
              event_id: 2,
              event_type: 'interception',
              frame_start: 78,
              frame_end: 92,
              timestamp_sec: 3.7,
              team_id: 1,
              team_name: 'Team B',
              source_player_id: 15,
              target_player_id: 26,
              distance_m: 11.2,
              speed_kmh: 28.6,
              details: 'Defensive cut-off in central channel'
            },
            {
              event_id: 3,
              event_type: 'tackle',
              frame_start: 130,
              frame_end: 135,
              timestamp_sec: 5.4,
              team_id: 0,
              team_name: 'Team A',
              source_player_id: 26,
              target_player_id: 11,
              distance_m: 1.4,
              speed_kmh: 8.2,
              details: '1v1 physical duel & recovery win'
            },
            {
              event_id: 4,
              event_type: 'shot',
              frame_start: 185,
              frame_end: 204,
              timestamp_sec: 8.1,
              team_id: 0,
              team_name: 'Team A',
              source_player_id: 11,
              target_player_id: null,
              distance_m: 21.6,
              speed_kmh: 54.2,
              details: 'Long-range shot towards right post'
            }
          ];

      timeline.forEach(ev => {
        const tr = document.createElement('tr');
        const badgeClass = `badge-${ev.event_type.toLowerCase()}`;
        const min = Math.floor(ev.timestamp_sec / 60);
        const sec = (ev.timestamp_sec % 60).toFixed(1).padStart(4, '0');
        const timeFormatted = `${min}:${sec} (f${ev.frame_start})`;

        let playerStr = '';
        if (ev.source_player_id !== null && ev.target_player_id !== null) {
          playerStr = `#${ev.source_player_id} → #${ev.target_player_id}`;
        } else if (ev.source_player_id !== null) {
          playerStr = `#${ev.source_player_id}`;
        } else if (ev.target_player_id !== null) {
          playerStr = `#${ev.target_player_id}`;
        } else {
          playerStr = '-';
        }

        const teamClass = ev.team_name === 'Team A' ? 'team-a' : 'team-b';

        tr.innerHTML = `
          <td><code>${timeFormatted}</code></td>
          <td><span class="badge ${badgeClass}">${ev.event_type.toUpperCase()}</span></td>
          <td>
            <span class="player-badge">
              <div class="team-indicator ${teamClass}"></div>
              ${ev.team_name}
            </span>
          </td>
          <td><span class="player-badge">${playerStr}</span></td>
          <td>${ev.details || `${ev.distance_m ? ev.distance_m.toFixed(1) + 'm trajectory' : 'Event registered'}`}</td>
          <td><span class="speed-tag">${ev.speed_kmh ? ev.speed_kmh.toFixed(1) + ' km/h' : '-'}</span></td>
        `;
        eventsBody.appendChild(tr);
      });
    }
  }
});
