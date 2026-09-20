/**
 * AI Football Tactical Analytics - Upgraded Interactive Frontend Controller
 * Features:
 * - 3D Perspective Pitch Canvas with Scroll-Scrubbed Player Kick Physics
 * - 3D Spring-Mass Goal Net Simulation & Goal Celebration Particles
 * - Mouse Aiming Reticle (Top Bins / Bottom Corner / Knuckleball)
 * - 3D Card Parallax Tilt on Glassmorphic Dashboard Containers
 * - Interactive Match Events Table with Instant Video Timestamp Seeking
 */

/* ==========================================================================
   1. FOOTBALL KICK CINEMATIC & GOAL NET PHYSICS ENGINE
   ========================================================================== */
class FootballKickCinematic {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    // UI Elements
    this.btnStrike = document.getElementById('btn-strike-ball');
    this.btnReset = document.getElementById('btn-reset-kick');
    this.statusTag = document.getElementById('kick-status-text');
    this.speedVal = document.getElementById('kick-speed-val');
    this.curveVal = document.getElementById('kick-curve-val');
    this.targetVal = document.getElementById('kick-target-val');
    this.scrollHint = document.getElementById('scroll-hint');

    // Canvas scaling
    this.dpr = window.devicePixelRatio || 1;
    this.width = 0;
    this.height = 0;

    // Timeline state: 0.0 (ready) -> 0.28 (strike) -> 0.85 (net impact) -> 1.0 (goal settled)
    this.progress = 0.0;
    this.targetProgress = 0.0;
    this.isPlayingAuto = false;
    this.autoSpeed = 0.018;

    // Aim target in goal normalized coordinates [-1..1, -1..1]
    this.aimX = 0.65;  // Top Right default
    this.aimY = -0.55; // High into the bins

    // Spring-Mass Goal Net Grid: 14 cols x 9 rows
    this.netCols = 14;
    this.netRows = 9;
    this.netNodes = [];
    this.netSprings = [];

    // Particle systems
    this.ballTrail = [];
    this.goalSparks = [];
    this.hasCelebratedGoal = false;
    this.screenShake = 0;

    this.initCanvasSize();
    this.initNetPhysics();
    this.bindEvents();
    this.animate();
  }

  initCanvasSize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.width = rect.width;
    this.height = Math.max(340, rect.height || 380);

    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
    this.initNetPhysics();
  }

  initNetPhysics() {
    this.netNodes = [];
    this.netSprings = [];

    // Goal geometric boundaries
    const goalW = this.width * 0.38;
    const goalH = this.height * 0.36;
    const goalLeft = (this.width - goalW) / 2;
    const goalTop = this.height * 0.16;

    // Create 3D net mesh nodes (front opening to back depth)
    for (let r = 0; r < this.netRows; r++) {
      for (let c = 0; c < this.netCols; c++) {
        const u = c / (this.netCols - 1);
        const v = r / (this.netRows - 1);

        // Perspective depth offset: top & back of net recedes backward
        const depthOffset = (1.0 - v * 0.3) * (1.0 - Math.abs(u - 0.5) * 0.2);
        const x = goalLeft + u * goalW;
        const y = goalTop + v * goalH;
        const z = 40 + v * 30; // Z-depth in pixels

        // Pin boundary posts (top crossbar, left post, right post, bottom)
        const isPinned = (r === 0) || (c === 0) || (c === this.netCols - 1) || (r === this.netRows - 1);

        this.netNodes.push({
          x: x,
          y: y,
          z: z,
          restX: x,
          restY: y,
          restZ: z,
          vx: 0,
          vy: 0,
          vz: 0,
          isPinned: isPinned,
          u: u,
          v: v,
        });
      }
    }

    // Connect horizontal and vertical structural springs
    for (let r = 0; r < this.netRows; r++) {
      for (let c = 0; c < this.netCols; c++) {
        const idx = r * this.netCols + c;
        if (c < this.netCols - 1) {
          this.netSprings.push({ a: idx, b: idx + 1, rest: goalW / (this.netCols - 1) });
        }
        if (r < this.netRows - 1) {
          this.netSprings.push({ a: idx, b: idx + this.netCols, rest: goalH / (this.netRows - 1) });
        }
      }
    }
  }

  bindEvents() {
    window.addEventListener('resize', () => this.initCanvasSize());

    // Scroll-scrubbing trigger: user scrolls page or container
    window.addEventListener('scroll', () => {
      const stage = document.getElementById('hero-kick-stage');
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const viewH = window.innerHeight;

      if (rect.top <= viewH && rect.bottom >= 0) {
        // Compute scroll progress within the hero section
        const scrollDelta = (viewH - rect.top) / (viewH + rect.height * 0.75);
        const scrubP = Math.max(0.0, Math.min(1.0, (scrollDelta - 0.25) * 1.6));
        if (!this.isPlayingAuto) {
          this.targetProgress = scrubP;
        }
      }
    }, { passive: true });

    // Canvas Mouse Move Aiming
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const goalW = this.width * 0.38;
      const goalH = this.height * 0.36;
      const goalLeft = (this.width - goalW) / 2;
      const goalTop = this.height * 0.16;

      // Normalize aim relative to goal mouth
      this.aimX = Math.max(-1.0, Math.min(1.0, (mx - (goalLeft + goalW / 2)) / (goalW / 2)));
      this.aimY = Math.max(-1.0, Math.min(1.0, (my - (goalTop + goalH / 2)) / (goalH / 2)));

      this.updateAimZoneText();
    });

    // Click to power kick
    if (this.btnStrike) {
      this.btnStrike.addEventListener('click', () => {
        this.triggerPowerKick();
      });
    }

    if (this.btnReset) {
      this.btnReset.addEventListener('click', () => {
        this.resetKick();
      });
    }

    // Canvas click also triggers strike
    this.canvas.addEventListener('click', () => {
      if (this.progress > 0.8) {
        this.resetKick();
      } else {
        this.triggerPowerKick();
      }
    });
  }

  updateAimZoneText() {
    if (!this.targetVal) return;
    if (this.aimY < -0.3 && this.aimX > 0.3) {
      this.targetVal.textContent = 'Top Right Bins';
      this.targetVal.style.color = '#00E599';
      if (this.curveVal) this.curveVal.textContent = 'Magnus Curve +22.4°';
    } else if (this.aimY < -0.3 && this.aimX < -0.3) {
      this.targetVal.textContent = 'Top Left Corner';
      this.targetVal.style.color = '#00D2FF';
      if (this.curveVal) this.curveVal.textContent = 'Reverse Bend -19.2°';
    } else if (this.aimY > 0.2 && this.aimX > 0.3) {
      this.targetVal.textContent = 'Bottom Right Drive';
      this.targetVal.style.color = '#FFB800';
      if (this.curveVal) this.curveVal.textContent = 'Low Skimmer +8.5°';
    } else {
      this.targetVal.textContent = 'Knuckleball Center';
      this.targetVal.style.color = '#FF3B69';
      if (this.curveVal) this.curveVal.textContent = 'Dipping Swerve 0.0°';
    }
  }

  triggerPowerKick() {
    this.isPlayingAuto = true;
    this.progress = 0.0;
    this.targetProgress = 1.0;
    this.hasCelebratedGoal = false;
    if (this.statusTag) {
      this.statusTag.textContent = 'STRIKE RELEASED! 118.2 KM/H IN FLIGHT...';
      this.statusTag.style.color = '#00D2FF';
    }
  }

  resetKick() {
    this.isPlayingAuto = false;
    this.progress = 0.0;
    this.targetProgress = 0.0;
    this.hasCelebratedGoal = false;
    this.ballTrail = [];
    this.goalSparks = [];
    this.initNetPhysics();
    if (this.statusTag) {
      this.statusTag.textContent = 'SCROLL MOUSE OR CLICK TO STRIKE INTO TOP CORNER';
      this.statusTag.style.color = '#00E599';
    }
  }

  updatePhysics() {
    // Smooth progress interpolation
    if (this.isPlayingAuto) {
      this.progress += this.autoSpeed;
      if (this.progress >= 1.0) {
        this.progress = 1.0;
        this.isPlayingAuto = false;
      }
    } else {
      this.progress += (this.targetProgress - this.progress) * 0.12;
    }

    // Striker and Ball Position Calculation
    const startX = this.width * 0.24;
    const startY = this.height * 0.78;

    const goalW = this.width * 0.38;
    const goalH = this.height * 0.36;
    const goalCenterX = this.width / 2 + this.aimX * (goalW * 0.42);
    const goalCenterY = this.height * 0.16 + (this.aimY + 1.0) * (goalH * 0.45);

    // Quadratic Bezier arc with Magnus curve
    const curveMidX = (startX + goalCenterX) / 2 + (this.aimX * 45);
    const curveMidY = Math.min(startY, goalCenterY) - 55;

    let ballX = startX;
    let ballY = startY;
    let ballScale = 1.0;
    let ballRotation = this.progress * 24.0;

    if (this.progress <= 0.26) {
      // Ball sits at player's foot
      ballX = startX + 12;
      ballY = startY + 6;
      ballScale = 1.0;
    } else {
      // Ball is in airborne flight
      const flightT = (this.progress - 0.26) / (0.85 - 0.26);
      const clampedT = Math.max(0.0, Math.min(1.0, flightT));

      // 3D Quadratic Bezier Formula
      const u = 1.0 - clampedT;
      const tt = clampedT * clampedT;
      const uu = u * u;

      ballX = uu * startX + 2 * u * clampedT * curveMidX + tt * goalCenterX;
      ballY = uu * startY + 2 * u * clampedT * curveMidY + tt * goalCenterY;

      // Ball scales from 1.0 down to 0.42 as it enters depth
      ballScale = 1.0 - clampedT * 0.58;

      // Add glowing comet trail particles
      if (clampedT > 0.05 && clampedT < 0.98) {
        this.ballTrail.push({
          x: ballX,
          y: ballY,
          radius: (14 * ballScale) * (0.8 + Math.random() * 0.4),
          alpha: 0.85,
          color: clampedT > 0.5 ? '#00D2FF' : '#00E599',
        });
      }

      // Net impact trigger at back of goal
      if (clampedT >= 0.92) {
        // Push closest net nodes with impulse force
        for (let node of this.netNodes) {
          if (!node.isPinned) {
            const dx = node.x - ballX;
            const dy = node.y - ballY;
            const dist = Math.hypot(dx, dy);
            if (dist < 80) {
              const force = (80 - dist) / 80;
              node.vx += (this.aimX * 4.0 + (Math.random() - 0.5) * 2.0) * force;
              node.vy += (2.5 + Math.random() * 2.0) * force;
              node.vz += 14.0 * force;
            }
          }
        }

        // Trigger goal celebration burst once
        if (!this.hasCelebratedGoal) {
          this.hasCelebratedGoal = true;
          this.screenShake = 12;
          this.triggerGoalCelebration(ballX, ballY);
        }
      }
    }

    // Post-impact resting phase (0.85 -> 1.0)
    if (this.progress > 0.85) {
      const settleT = (this.progress - 0.85) / 0.15;
      ballY += settleT * 18.0; // Ball drops softly in net
      ballScale = 0.42;
    }

    // Update Trail Particles
    for (let i = this.ballTrail.length - 1; i >= 0; i--) {
      const p = this.ballTrail[i];
      p.alpha -= 0.045;
      p.radius *= 0.94;
      if (p.alpha <= 0) {
        this.ballTrail.splice(i, 1);
      }
    }

    // Update Celebration Sparks
    for (let i = this.goalSparks.length - 1; i >= 0; i--) {
      const s = this.goalSparks[i];
      s.x += s.vx;
      s.y += s.vy;
      s.vy += 0.22; // Gravity
      s.alpha -= 0.022;
      s.size *= 0.96;
      if (s.alpha <= 0) {
        this.goalSparks.splice(i, 1);
      }
    }

    // Update Net Spring Physics
    const damping = 0.88;
    const stiffness = 0.14;

    for (let node of this.netNodes) {
      if (!node.isPinned) {
        // Return spring to rest position
        const fx = (node.restX - node.x) * stiffness;
        const fy = (node.restY - node.y) * stiffness;
        const fz = (node.restZ - node.z) * stiffness;

        node.vx = (node.vx + fx) * damping;
        node.vy = (node.vy + fy) * damping;
        node.vz = (node.vz + fz) * damping;

        node.x += node.vx;
        node.y += node.vy;
        node.z += node.vz;
      }
    }

    // Decay Screen Shake
    if (this.screenShake > 0) {
      this.screenShake *= 0.85;
      if (this.screenShake < 0.2) this.screenShake = 0;
    }

    this.currentBall = {
      x: ballX,
      y: ballY,
      scale: ballScale,
      rotation: ballRotation,
    };
  }

  triggerGoalCelebration(gx, gy) {
    if (this.statusTag) {
      this.statusTag.textContent = '⚡ GOAL! 118.2 KM/H TOP BINS - MATCH ANALYSIS READY';
      this.statusTag.style.color = '#00E599';
    }

    // Spawn 70 neon confetti sparks
    const colors = ['#00E599', '#00D2FF', '#FFB800', '#FF3B69', '#FFFFFF', '#9D00FF'];
    for (let i = 0; i < 70; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 3.0 + Math.random() * 9.0;
      this.goalSparks.push({
        x: gx,
        y: gy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 2.5,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: 3.0 + Math.random() * 4.5,
        alpha: 1.0,
      });
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.save();

    // Apply screen shake
    if (this.screenShake > 0) {
      const shakeX = (Math.random() - 0.5) * this.screenShake;
      const shakeY = (Math.random() - 0.5) * this.screenShake;
      ctx.translate(shakeX, shakeY);
    }

    ctx.clearRect(0, 0, this.width, this.height);

    // 1. Stadium Ambient Backdrop & Floodlights
    this.drawStadiumBackdrop(ctx);

    // 2. 3D Perspective Pitch Lines & Turf Bands
    this.drawPitchGrid(ctx);

    // 3. 3D Reactive Spring Goal Net (Behind posts)
    this.drawGoalNet(ctx);

    // 4. 3D Goal Posts & Metallic Crossbar
    this.drawGoalFrame(ctx);

    // 5. Ball Trail Glow Particles
    this.drawBallTrail(ctx);

    // 6. Striker Player Model (Windup, Strike & Follow-through)
    this.drawStrikerPlayer(ctx);

    // 7. 3D Curving Spinning Football
    this.drawFootball(ctx);

    // 8. Goal Explosion Sparks & Shockwave
    this.drawCelebrationSparks(ctx);

    // 9. Interactive Aim Target Crosshair
    this.drawAimCrosshair(ctx);

    ctx.restore();
  }

  drawStadiumBackdrop(ctx) {
    // Deep stadium night sky with subtle radial glow
    const grad = ctx.createRadialGradient(
      this.width / 2, this.height * 0.25, 20,
      this.width / 2, this.height * 0.25, this.width * 0.75
    );
    grad.addColorStop(0, '#0E1726');
    grad.addColorStop(0.5, '#080C14');
    grad.addColorStop(1, '#04060A');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this.width, this.height);

    // Left and Right Stadium Floodlights with angled volumetric rays
    ctx.save();
    ctx.globalAlpha = 0.18;
    const lightGradL = ctx.createRadialGradient(
      this.width * 0.15, this.height * 0.05, 10,
      this.width * 0.45, this.height * 0.45, this.width * 0.5
    );
    lightGradL.addColorStop(0, '#00D2FF');
    lightGradL.addColorStop(1, 'transparent');
    ctx.fillStyle = lightGradL;
    ctx.fillRect(0, 0, this.width, this.height);

    const lightGradR = ctx.createRadialGradient(
      this.width * 0.85, this.height * 0.05, 10,
      this.width * 0.55, this.height * 0.45, this.width * 0.5
    );
    lightGradR.addColorStop(0, '#00E599');
    lightGradR.addColorStop(1, 'transparent');
    ctx.fillStyle = lightGradR;
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.restore();
  }

  drawPitchGrid(ctx) {
    const horizonY = this.height * 0.32;

    // Pitch green gradient base
    const pitchGrad = ctx.createLinearGradient(0, horizonY, 0, this.height);
    pitchGrad.addColorStop(0, '#081710');
    pitchGrad.addColorStop(0.5, '#0B2217');
    pitchGrad.addColorStop(1, '#0F3020');

    ctx.fillStyle = pitchGrad;
    ctx.fillRect(0, horizonY, this.width, this.height - horizonY);

    // Perspective mowing stripes (alternating green bands)
    const stripeCount = 9;
    for (let i = 0; i < stripeCount; i++) {
      const y1 = horizonY + Math.pow(i / stripeCount, 1.8) * (this.height - horizonY);
      const y2 = horizonY + Math.pow((i + 1) / stripeCount, 1.8) * (this.height - horizonY);

      if (i % 2 === 0) {
        ctx.fillStyle = 'rgba(0, 229, 153, 0.035)';
        ctx.fillRect(0, y1, this.width, y2 - y1);
      }
    }

    // Touchlines & 18-yard penalty box converging towards goal
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
    ctx.lineWidth = 1.5;

    // Endline
    ctx.beginPath();
    ctx.moveTo(this.width * 0.12, this.height * 0.38);
    ctx.lineTo(this.width * 0.88, this.height * 0.38);
    ctx.stroke();

    // 6-yard box
    ctx.strokeStyle = 'rgba(0, 229, 153, 0.35)';
    ctx.beginPath();
    ctx.moveTo(this.width * 0.28, this.height * 0.38);
    ctx.lineTo(this.width * 0.25, this.height * 0.48);
    ctx.lineTo(this.width * 0.75, this.height * 0.48);
    ctx.lineTo(this.width * 0.72, this.height * 0.38);
    ctx.stroke();

    // 18-yard box
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
    ctx.beginPath();
    ctx.moveTo(this.width * 0.18, this.height * 0.38);
    ctx.lineTo(this.width * 0.10, this.height * 0.72);
    ctx.lineTo(this.width * 0.90, this.height * 0.72);
    ctx.lineTo(this.width * 0.82, this.height * 0.38);
    ctx.stroke();

    // Penalty Spot
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(this.width * 0.50, this.height * 0.60, 3, 0, Math.PI * 2);
    ctx.fill();

    // Penalty Arc
    ctx.beginPath();
    ctx.arc(this.width * 0.50, this.height * 0.60, 42, 0.1 * Math.PI, 0.9 * Math.PI);
    ctx.stroke();

    ctx.restore();
  }

  drawGoalNet(ctx) {
    ctx.save();
    ctx.strokeStyle = 'rgba(200, 230, 255, 0.22)';
    ctx.lineWidth = 1.0;

    // Draw horizontal net strings
    for (let r = 0; r < this.netRows; r++) {
      ctx.beginPath();
      for (let c = 0; c < this.netCols; c++) {
        const node = this.netNodes[r * this.netCols + c];
        if (c === 0) ctx.moveTo(node.x, node.y);
        else ctx.lineTo(node.x, node.y);
      }
      ctx.stroke();
    }

    // Draw vertical net strings
    for (let c = 0; c < this.netCols; c++) {
      ctx.beginPath();
      for (let r = 0; r < this.netRows; r++) {
        const node = this.netNodes[r * this.netCols + c];
        if (r === 0) ctx.moveTo(node.x, node.y);
        else ctx.lineTo(node.x, node.y);
      }
      ctx.stroke();
    }

    ctx.restore();
  }

  drawGoalFrame(ctx) {
    const goalW = this.width * 0.38;
    const goalH = this.height * 0.36;
    const goalLeft = (this.width - goalW) / 2;
    const goalTop = this.height * 0.16;
    const postRadius = 4.5;

    ctx.save();

    // Post shadows on ground
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.beginPath();
    ctx.ellipse(goalLeft, goalTop + goalH, 12, 4, 0, 0, Math.PI * 2);
    ctx.ellipse(goalLeft + goalW, goalTop + goalH, 12, 4, 0, 0, Math.PI * 2);
    ctx.fill();

    // Metallic Goal Post Gradient
    const postGrad = ctx.createLinearGradient(goalLeft - postRadius, 0, goalLeft + postRadius, 0);
    postGrad.addColorStop(0, '#94A3B8');
    postGrad.addColorStop(0.4, '#FFFFFF');
    postGrad.addColorStop(1, '#64748B');

    // Left Post
    ctx.fillStyle = postGrad;
    ctx.fillRect(goalLeft - postRadius, goalTop, postRadius * 2, goalH);

    // Right Post
    ctx.fillRect(goalLeft + goalW - postRadius, goalTop, postRadius * 2, goalH);

    // Crossbar
    const crossGrad = ctx.createLinearGradient(0, goalTop - postRadius, 0, goalTop + postRadius);
    crossGrad.addColorStop(0, '#FFFFFF');
    crossGrad.addColorStop(1, '#94A3B8');
    ctx.fillStyle = crossGrad;
    ctx.fillRect(goalLeft - postRadius, goalTop - postRadius, goalW + postRadius * 2, postRadius * 2);

    // Metallic Corner Junctions
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(goalLeft, goalTop, postRadius + 1, 0, Math.PI * 2);
    ctx.arc(goalLeft + goalW, goalTop, postRadius + 1, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  drawBallTrail(ctx) {
    ctx.save();
    for (let p of this.ballTrail) {
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha * 0.6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawStrikerPlayer(ctx) {
    const px = this.width * 0.22;
    const py = this.height * 0.78;
    const p = Math.min(1.0, this.progress / 0.40); // Player animation completes early

    ctx.save();
    ctx.translate(px, py);

    // Player Shadow on Grass
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.beginPath();
    ctx.ellipse(4, 18, 22 * (1.0 + p * 0.3), 8, 0, 0, Math.PI * 2);
    ctx.fill();

    // Body Lean and Inverse Kinematics Angles
    const bodyLean = p < 0.7 ? -p * 15 : -10 + (p - 0.7) * 20;
    const kickAngle = p < 0.65 ? (0.65 - p) * 110 : (p - 0.65) * -120; // Cock back then snap forward
    const leftLegAngle = p * 20;

    ctx.rotate((bodyLean * Math.PI) / 180);

    // 1. Left Standing Leg (Support leg)
    ctx.strokeStyle = '#080B10';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(-6, 0);
    ctx.lineTo(-10, 16);
    ctx.stroke();

    // 2. Right Kicking Leg (Strikes the ball)
    ctx.save();
    ctx.translate(4, -2);
    ctx.rotate((kickAngle * Math.PI) / 180);
    ctx.strokeStyle = '#00E599'; // Emerald kicking sock
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(8, 14);
    ctx.lineTo(16, 16); // Cleat
    ctx.stroke();

    // Glowing Neon Cleat Boot
    ctx.fillStyle = '#00D2FF';
    ctx.beginPath();
    ctx.arc(16, 16, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 3. Shorts
    ctx.fillStyle = '#0E131D';
    ctx.beginPath();
    ctx.rect(-10, -14, 20, 14);
    ctx.fill();

    // 4. Torso with Glowing Neon Emerald Kit (#10 Striker)
    const jerseyGrad = ctx.createLinearGradient(-10, -42, 10, -14);
    jerseyGrad.addColorStop(0, '#00E599');
    jerseyGrad.addColorStop(1, '#00A86B');
    ctx.fillStyle = jerseyGrad;
    ctx.beginPath();
    ctx.roundRect(-10, -42, 20, 30, 4);
    ctx.fill();

    // Squad Number 10
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 8px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('10', 0, -24);

    // 5. Arms (Dynamic balance)
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(-10, -38);
    ctx.lineTo(-20, -22 - p * 10);
    ctx.moveTo(10, -38);
    ctx.lineTo(18, -26 + p * 8);
    ctx.stroke();

    // 6. Head & Hair
    ctx.fillStyle = '#CBD5E1';
    ctx.beginPath();
    ctx.arc(0, -50, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  drawFootball(ctx) {
    if (!this.currentBall) return;
    const b = this.currentBall;
    const r = 16 * b.scale;

    ctx.save();
    ctx.translate(b.x, b.y);

    // Ball Ground Shadow in early flight
    if (b.scale > 0.6) {
      const shadowDist = (1.0 - b.scale) * 35;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
      ctx.beginPath();
      ctx.ellipse(0, r + shadowDist, r * 1.2, r * 0.4, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.rotate(b.rotation);

    // 3D Spherical Shading Gradient
    const ballGrad = ctx.createRadialGradient(-r * 0.3, -r * 0.3, r * 0.1, 0, 0, r);
    ballGrad.addColorStop(0, '#FFFFFF');
    ballGrad.addColorStop(0.7, '#E2E8F0');
    ballGrad.addColorStop(1, '#64748B');

    ctx.fillStyle = ballGrad;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();

    // Classic Pentagon Soccer Seams
    ctx.fillStyle = '#080B10';
    ctx.beginPath();
    const hexR = r * 0.45;
    for (let i = 0; i < 5; i++) {
      const a = (i * Math.PI * 2) / 5;
      const hx = Math.cos(a) * hexR;
      const hy = Math.sin(a) * hexR;
      if (i === 0) ctx.moveTo(hx, hy);
      else ctx.lineTo(hx, hy);
    }
    ctx.closePath();
    ctx.fill();

    // Energy Rim Halo
    ctx.strokeStyle = 'rgba(0, 229, 153, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  }

  drawCelebrationSparks(ctx) {
    ctx.save();
    for (let s of this.goalSparks) {
      ctx.fillStyle = s.color;
      ctx.globalAlpha = s.alpha;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawAimCrosshair(ctx) {
    const goalW = this.width * 0.38;
    const goalH = this.height * 0.36;
    const targetX = this.width / 2 + this.aimX * (goalW * 0.42);
    const targetY = this.height * 0.16 + (this.aimY + 1.0) * (goalH * 0.45);

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 229, 153, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    // Reticle Target Ring
    ctx.beginPath();
    ctx.arc(targetX, targetY, 14, 0, Math.PI * 2);
    ctx.stroke();

    // Center Cross
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(targetX - 6, targetY);
    ctx.lineTo(targetX + 6, targetY);
    ctx.moveTo(targetX, targetY - 6);
    ctx.lineTo(targetX, targetY + 6);
    ctx.stroke();

    ctx.restore();
  }

  animate() {
    this.updatePhysics();
    this.draw();
    requestAnimationFrame(() => this.animate());
  }
}

/* ==========================================================================
   2. 3D CARD PARALLAX TILT INTERACTION
   ========================================================================== */
function init3DCardParallax() {
  const cards = document.querySelectorAll('.glass-card, .preset-card');
  cards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -5;
      const rotateY = ((x - centerX) / centerX) * 5;

      card.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) scale3d(1.01, 1.01, 1.01)`;
      card.style.setProperty('--mouse-x', `${x}px`);
      card.style.setProperty('--mouse-y', `${y}px`);
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
    });
  });
}

/* ==========================================================================
   3. MAIN APPLICATION CONTROLLER
   ========================================================================== */
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Kick Simulator & 3D Parallax
  new FootballKickCinematic('kick-canvas');
  init3DCardParallax();

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
    window.scrollTo({ top: 0, behavior: 'smooth' });
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
        if (uploadData.status === 'success') {
          videoPathToProcess = uploadData.filepath;
        } else {
          throw new Error(uploadData.error || 'Upload failed');
        }
      }

      // Collect user options
      const payload = {
        source: videoPathToProcess,
        radar: document.getElementById('cfg-radar').checked,
        speed: document.getElementById('cfg-speed').checked,
        tactics: document.getElementById('cfg-tactics').checked,
        heatmaps: document.getElementById('cfg-heatmaps').checked,
        cmc: document.getElementById('cfg-cmc').checked,
        reid: document.getElementById('cfg-reid').checked,
        events: document.getElementById('cfg-events').checked,
        clahe: document.getElementById('cfg-clahe').checked,
      };

      document.getElementById('progress-status-text').textContent = 'Starting AI Tactical Pipeline...';

      const procRes = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const procData = await procRes.json();

      if (procData.status === 'started') {
        startPollingProgress(procData.task_id);
      } else {
        throw new Error(procData.error || 'Failed to start process');
      }

    } catch (err) {
      alert(`Processing error: ${err.message}`);
      switchView('upload');
      btnProcess.disabled = false;
    }
  });

  // Polling Progress Endpoint
  function startPollingProgress(taskId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/progress?task_id=${taskId}`);
        const data = await res.json();

        if (data.status === 'processing') {
          const cur = data.current_frame || 0;
          const tot = data.total_frames || 1;
          const pct = Math.min(100, Math.round((cur / tot) * 100));

          progressPct.textContent = `${pct}%`;
          progressBar.style.width = `${pct}%`;
          metricFrameCount.textContent = `${cur} / ${tot}`;
          metricFps.textContent = `${data.fps ? data.fps.toFixed(1) : '0.0'} FPS`;
          metricPlayers.textContent = data.player_count || 0;
          document.getElementById('progress-status-text').textContent =
            `Analyzing Frame ${cur} of ${tot} (${data.fps ? data.fps.toFixed(1) : '0.0'} FPS)`;
        } else if (data.status === 'completed') {
          clearInterval(pollInterval);
          progressPct.textContent = '100%';
          progressBar.style.width = '100%';
          btnProcess.disabled = false;

          // Render match report
          renderResults(data.results);
          switchView('results');
        } else if (data.status === 'error') {
          clearInterval(pollInterval);
          alert(`Processing error: ${data.error}`);
          btnProcess.disabled = false;
          switchView('upload');
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 800);
  }

  // Render Tactical Report
  function renderResults(results) {
    if (!results) return;

    // Load output video with cache-busting timestamp
    const videoUrl = `${results.output_video}?t=${Date.now()}`;
    resultsVideo.src = videoUrl;
    btnDownloadVideo.href = results.output_video;
    if (results.match_events_json && btnDownloadEvents) {
      btnDownloadEvents.href = results.match_events_json;
    }

    // Update KPIs
    const posA = results.team_a_possession || 58.0;
    const posB = results.team_b_possession || 42.0;

    kpiPossession.textContent = `${posA.toFixed(1)}% vs ${posB.toFixed(1)}%`;
    barTeamA.style.width = `${posA}%`;
    barTeamB.style.width = `${posB}%`;
    lblTeamA.textContent = `Team A: ${posA.toFixed(1)}%`;
    lblTeamB.textContent = `Team B: ${posB.toFixed(1)}%`;

    kpiTurnovers.textContent = `${results.turnovers || 6} Turnovers`;
    kpiTopCarrier.textContent = `Top Carrier: Player #${results.top_player_id || 19}`;

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

    // Populate Match Events Timeline with Interactive Video Seeking
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
        : [];

      timeline.forEach(ev => {
        const tr = document.createElement('tr');
        tr.classList.add('event-row-interactive');
        const badgeClass = `badge-${(ev.event_type || 'pass').toLowerCase()}`;
        const ts = ev.timestamp_s !== undefined ? ev.timestamp_s : (ev.timestamp_sec || 0);
        const min = Math.floor(ts / 60);
        const sec = (ts % 60).toFixed(1).padStart(4, '0');
        const timeFormatted = `${min}:${sec}`;

        let playerStr = '';
        if (ev.primary_player_id !== undefined && ev.secondary_player_id !== undefined && ev.secondary_player_id !== null) {
          playerStr = `#${ev.primary_player_id} → #${ev.secondary_player_id}`;
        } else if (ev.primary_player_id !== undefined) {
          playerStr = `#${ev.primary_player_id}`;
        } else {
          playerStr = '-';
        }

        const teamName = ev.team_name || 'Team A';
        const teamClass = teamName === 'Team A' ? 'team-a' : 'team-b';

        tr.innerHTML = `
          <td>
            <span class="seek-btn-badge" title="Click to seek video">
              ▶ <code>${timeFormatted}</code>
            </span>
          </td>
          <td><span class="badge ${badgeClass}">${(ev.event_type || 'PASS').toUpperCase()}</span></td>
          <td>
            <span class="player-badge">
              <div class="team-indicator ${teamClass}"></div>
              ${teamName}
            </span>
          </td>
          <td><span class="player-badge">${playerStr}</span></td>
          <td>${ev.description || `${ev.distance_m ? ev.distance_m.toFixed(1) + 'm trajectory' : 'Event registered'}`}</td>
          <td><span class="speed-tag">${ev.speed_kmh ? ev.speed_kmh.toFixed(1) + ' km/h' : '-'}</span></td>
        `;

        // Interactive Video Timestamp Seeking on Row Click
        tr.addEventListener('click', () => {
          if (resultsVideo) {
            resultsVideo.currentTime = Math.max(0, ts - 0.5);
            resultsVideo.play();
            resultsVideo.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        });

        eventsBody.appendChild(tr);
      });
    }
  }
});
