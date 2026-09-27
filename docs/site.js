(() => {
  'use strict';
  const videos = [...document.querySelectorAll('video')];
  const status = document.querySelector('#playback-status');
  const announce = text => { status.textContent = text; };
  const pauseOthers = current => videos.forEach(video => {
    if (video !== current && !video.paused) video.pause();
  });
  const play = async video => {
    try { await video.play(); }
    catch { announce('Use the video play button to begin playback.'); }
  };

  videos.forEach(video => {
    const shell = video.closest('.video-shell');
    if (shell) {
      const button = document.createElement('button');
      button.className = 'play-video';
      button.type = 'button';
      button.setAttribute('aria-label', `Play ${video.getAttribute('aria-label')}`);
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7z"/></svg>';
      video.controls = false;
      shell.append(button);
      button.addEventListener('click', () => {
        video.controls = true;
        play(video);
      });
      video.addEventListener('play', () => { button.hidden = true; video.controls = true; });
      video.addEventListener('error', () => { button.hidden = true; video.controls = true; });
    }
    video.addEventListener('play', () => pauseOthers(video));
    video.addEventListener('error', () => {
      announce('The video could not load. Use its download link to open it directly.');
    });
  });

  document.querySelectorAll('[data-pause-all]').forEach(button => {
    button.addEventListener('click', () => {
      pauseOthers(null);
      announce('All videos paused.');
    });
  });

  document.querySelectorAll('.sequence-panel').forEach(panel => {
    const video = panel.querySelector('video');
    const steps = [...panel.querySelectorAll('.prompt-step')];
    const progress = panel.querySelector('.video-progress span');
    const update = () => {
      const time = video.currentTime;
      let active = steps.findIndex(step => time < Number(step.dataset.end));
      if (active < 0) active = steps.length - 1;
      steps.forEach((step, index) => {
        if (index === active) step.setAttribute('aria-current', 'step');
        else step.removeAttribute('aria-current');
      });
      if (Number.isFinite(video.duration) && video.duration > 0) {
        progress.style.width = `${Math.min(100, time / video.duration * 100)}%`;
      }
    };
    video.addEventListener('timeupdate', update);
    video.addEventListener('loadedmetadata', update);
    steps.forEach(step => step.addEventListener('click', () => {
      const seek = () => {
        video.currentTime = Number(step.dataset.start);
        update();
        play(video);
      };
      if (video.readyState >= 1) seek();
      else {
        video.addEventListener('loadedmetadata', seek, {once: true});
        video.load();
      }
    }));
    update();
  });

  document.querySelectorAll('[data-example-group]').forEach(group => {
    const picker = group.querySelector('.example-picker');
    const buttons = [...picker.querySelectorAll('.example-choice')];
    const panels = [...group.querySelectorAll('.sequence-panel')];
    const select = id => {
      panels.forEach(panel => {
        panel.hidden = panel.id !== id;
        if (panel.hidden) panel.querySelector('video').pause();
      });
      buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.target === id)));
    };
    picker.hidden = false;
    buttons.forEach(button => button.addEventListener('click', () => select(button.dataset.target)));
    select(group.dataset.default);
    const initial = location.hash.slice(1);
    if (panels.some(panel => panel.id === initial)) select(initial);
    window.addEventListener('hashchange', () => {
      const id = location.hash.slice(1);
      if (panels.some(panel => panel.id === id)) select(id);
    });
  });

  const copy = document.querySelector('#copy-citation');
  copy.hidden = false;
  copy.addEventListener('click', async () => {
    const text = document.querySelector('#citation-text').textContent;
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = 'Copied';
      announce('Citation copied.');
      window.setTimeout(() => { copy.textContent = 'Copy BibTeX'; }, 2000);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.querySelector('#citation-text'));
      selection.removeAllRanges();
      selection.addRange(range);
      announce('Citation selected. Copy it with your keyboard.');
    }
  });

  if ('IntersectionObserver' in window) {
    const navigation = [...document.querySelectorAll('nav a[href^="#"]')];
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) navigation.forEach(link => {
          if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      });
    }, {rootMargin: '-15% 0px -65% 0px'});
    navigation.forEach(link => {
      const section = document.querySelector(link.hash);
      if (section) observer.observe(section);
    });
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pauseOthers(null);
  });
})();
