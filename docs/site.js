(() => {
  'use strict';
  const videos = [...document.querySelectorAll('video:not(#hero-video)')];
  const film = document.querySelector('#overview-video');
  const viewer = document.querySelector('#motion-viewer');
  let filmNeedsGesture = false;
  const status = document.querySelector('#playback-status');
  const announce = text => { status.textContent = text; };
  const playback = new Map(videos.map(video => [video, {
    visible: false, pausedByUser: false, scriptedPauses: 0
  }]));
  const pause = video => {
    if (!video.paused) {
      playback.get(video).scriptedPauses += 1;
      video.pause();
    }
  };
  const play = async video => {
    playback.get(video).pausedByUser = false;
    try { await video.play(); }
    catch { announce('Use the video play button to begin playback.'); }
  };
  const autoplay = async video => {
    try { await video.play(); }
    catch (error) {
      if (video !== film || error.name !== 'NotAllowedError' || film.muted) return;
      const state = playback.get(film);
      if (!state.visible || document.hidden || state.pausedByUser) return;
      filmNeedsGesture = true;
      film.muted = true;
      film.play().catch(() => {});
    }
  };
  const updatePlayback = video => {
    const state = playback.get(video);
    const visible = viewer.open ? viewer.contains(video) : state.visible;
    if (visible && !document.hidden && !video.closest('[hidden]') &&
        !state.pausedByUser && !video.ended) {
      if (video.paused) autoplay(video);
    } else pause(video);
  };

  const hero = document.querySelector('#hero-video');
  const header = document.querySelector('.site-header');
  const worksMenu = document.querySelector('.works-menu');
  const credits = document.querySelector('#team');
  const research = document.querySelector('.research-content');
  credits.addEventListener('focusin', event => {
    if (event.target.matches(':focus-visible') &&
        event.target.getBoundingClientRect().bottom > research.getBoundingClientRect().top) {
      window.scrollBy({top: credits.getBoundingClientRect().top, behavior: 'instant'});
    }
  });
  document.addEventListener('click', event => {
    if (!worksMenu.contains(event.target) || event.target.closest('a')) worksMenu.open = false;
  });
  worksMenu.addEventListener('keydown', event => {
    if (event.key === 'Escape' && worksMenu.open) {
      event.preventDefault();
      worksMenu.open = false;
      worksMenu.querySelector('summary').focus();
    }
  });
  worksMenu.addEventListener('focusout', event => {
    if (!worksMenu.contains(event.relatedTarget)) worksMenu.open = false;
  });
  new IntersectionObserver(([entry]) => {
    header.classList.toggle('is-visible', !entry.isIntersecting && entry.boundingClientRect.top < 0);
    if (entry.isIntersecting) worksMenu.open = false;
  }, {rootMargin: `-${header.offsetHeight}px 0px 0px 0px`}).observe(hero);
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let heroVisible = false;
  const updateHero = () => {
    if (heroVisible && !viewer.open && !document.hidden && !reducedMotion.matches) {
      hero.play().catch(() => {});
    } else hero.pause();
  };
  reducedMotion.addEventListener('change', updateHero);
  new IntersectionObserver(([entry]) => {
    heroVisible = entry.isIntersecting;
    updateHero();
  }).observe(hero);
  document.addEventListener('visibilitychange', updateHero);

  const soundButton = document.querySelector('#film-sound');
  const updateSoundButton = () => {
    const audible = !film.muted && film.volume > 0;
    if (audible || film.volume === 0) filmNeedsGesture = false;
    soundButton.dataset.audible = String(audible);
    soundButton.querySelector('.sound-label').textContent = audible ? 'Sound off' : 'Sound on';
    soundButton.setAttribute('aria-label', audible ? 'Turn sound off' : 'Turn sound on');
  };
  soundButton.addEventListener('click', () => {
    const audible = !film.muted && film.volume > 0;
    filmNeedsGesture = false;
    film.muted = audible;
    if (!audible && film.volume === 0) film.volume = 1;
    if (!audible && !playback.get(film).pausedByUser) play(film);
  });
  film.addEventListener('volumechange', updateSoundButton);
  updateSoundButton();
  soundButton.hidden = false;
  document.querySelectorAll('#watch-film, .site-header a[href="#overview"]').forEach(link => {
    link.addEventListener('click', () => {
      if (filmNeedsGesture) {
        filmNeedsGesture = false;
        film.muted = false;
      }
      play(film);
    });
  });

  videos.forEach(video => {
    const state = playback.get(video);
    video.muted = video !== film;
    video.controls = true;
    video.addEventListener('play', () => {
      if (!video.paused) state.pausedByUser = false;
    });
    video.addEventListener('pause', () => {
      if (state.scriptedPauses) state.scriptedPauses -= 1;
      else if (!video.ended) state.pausedByUser = true;
    });
    video.addEventListener('error', () => {
      announce('The video could not load. Please reload the page and try again.');
    });
  });
  const videoObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const video = entry.target.matches('video') ? entry.target : entry.target.querySelector('video');
      playback.get(video).visible = entry.isIntersecting && entry.intersectionRatio >= 0.1;
      updatePlayback(video);
    });
  }, {threshold: 0.1});
  const preloadObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const video = entry.target.matches('video') ? entry.target : entry.target.querySelector('video');
      video.preload = 'auto';
      preloadObserver.unobserve(entry.target);
    });
  }, {rootMargin: `${window.innerHeight}px 0px`});
  videos.forEach(video => {
    const frame = video.closest('.showcase-player') || video;
    videoObserver.observe(frame);
    if (video !== film) preloadObserver.observe(frame);
  });

  const showcase = document.querySelector('#continuous-transitions');
  const tabs = document.querySelector('.showcase-tabs');
  const categoryButtons = [...document.querySelectorAll('.category-choice')];
  const groups = [...document.querySelectorAll('.result-group')];
  const description = document.querySelector('#motion-description');
  const mobileView = window.matchMedia('(max-width: 720px)');
  const results = [...document.querySelectorAll('.motion-result')];
  let wasInDemo = false;
  let scrollReference = window.scrollY;
  const updateDemoNavigation = () => {
    const bounds = showcase.getBoundingClientRect();
    const inDemo = mobileView.matches && bounds.top <= header.offsetHeight &&
      bounds.bottom > header.offsetHeight;
    const keyboardFocus = header.contains(document.activeElement) &&
      document.activeElement.matches(':focus-visible');
    const distance = window.scrollY - scrollReference;
    if (!inDemo || worksMenu.open || keyboardFocus) {
      document.documentElement.classList.remove('demo-nav-collapsed');
      scrollReference = window.scrollY;
    } else if (!wasInDemo || Math.abs(distance) > 12) {
      document.documentElement.classList.toggle('demo-nav-collapsed', !wasInDemo || distance > 0);
      scrollReference = window.scrollY;
    }
    wasInDemo = inDemo;
  };

  const updateFraming = result => {
    const focused = result.dataset.fullView !== 'true';
    result.classList.add('has-framing');
    result.classList.toggle('is-focused', focused);
    result.querySelector('video').controls = !focused;
  };
  let selectedGroup = groups.find(group => !group.hidden);
  let tabsVisible = false;
  let rotationTimer = null;
  let rotationProgress = null;

  const stopRotation = () => {
    clearTimeout(rotationTimer);
    rotationTimer = null;
    rotationProgress?.cancel();
    rotationProgress = null;
  };
  const firstRowTop = () => selectedGroup.querySelector('.motion-result').getBoundingClientRect().top;
  const resultsViewportTop = () => tabs.getBoundingClientRect().bottom;
  const canRotate = () => !viewer.open && tabsVisible && !document.hidden && !reducedMotion.matches &&
    firstRowTop() >= resultsViewportTop() - 16 && firstRowTop() < window.innerHeight &&
    !document.fullscreenElement && !videos.some(video => video.webkitDisplayingFullscreen) &&
    !selectedGroup.matches(':hover') &&
    !selectedGroup.contains(document.activeElement) &&
    !selectedGroup.querySelector('details[open]') &&
    ![...selectedGroup.querySelectorAll('video')].some(video => playback.get(video).pausedByUser);
  const updateRotation = () => {
    if (!canRotate()) {
      stopRotation();
      return;
    }
    if (rotationTimer !== null) return;
    const duration = Number(selectedGroup.dataset.rotationSeconds || 20) * 1000;
    rotationProgress = tabs.querySelector('[aria-pressed="true"] .category-progress').animate(
      [{transform: 'scaleX(0)'}, {transform: 'scaleX(1)'}],
      {duration, easing: 'linear', fill: 'forwards'}
    );
    rotationTimer = setTimeout(() => {
      stopRotation();
      if (canRotate()) {
        const next = groups[(groups.indexOf(selectedGroup) + 1) % groups.length];
        selectCategory(next.dataset.categoryPanel);
      }
    }, duration);
  };
  let viewerResult = null;
  let viewerPlaceholder = null;
  let viewerScroll = 0;
  const openViewer = result => {
    stopRotation();
    viewerResult = result;
    viewerScroll = window.scrollY;
    const video = result.querySelector('video');
    pause(video);
    viewerPlaceholder = document.createElement('div');
    viewerPlaceholder.className = 'motion-placeholder';
    viewerPlaceholder.style.height = `${result.getBoundingClientRect().height}px`;
    result.replaceWith(viewerPlaceholder);
    viewer.querySelector('.viewer-content').append(result);
    result.dataset.fullView = 'true';
    updateFraming(result);
    viewer.querySelector('#viewer-title').textContent = result.getAttribute('aria-label');
    document.documentElement.style.setProperty('--viewer-scrollbar', `${window.innerWidth - document.documentElement.clientWidth}px`);
    document.documentElement.classList.add('viewer-open');
    viewer.showModal();
    videos.forEach(updatePlayback);
    updateHero();
  };
  const closeViewer = (restorePosition = true) => {
    if (!viewerResult) return;
    const result = viewerResult;
    pause(result.querySelector('video'));
    viewerPlaceholder.replaceWith(result);
    result.dataset.fullView = 'false';
    updateFraming(result);
    viewerResult = null;
    viewerPlaceholder = null;
    document.documentElement.classList.remove('viewer-open');
    document.documentElement.style.removeProperty('--viewer-scrollbar');
    if (restorePosition) window.scrollTo({top: viewerScroll, behavior: 'instant'});
    result.querySelector('.framing-toggle').focus({preventScroll: true});
    videos.forEach(updatePlayback);
    updateHero();
    updateRotation();
  };
  viewer.querySelector('.viewer-close').addEventListener('click', () => viewer.close());
  viewer.addEventListener('close', () => closeViewer());
  viewer.addEventListener('click', event => {
    const bounds = viewer.getBoundingClientRect();
    if (event.target === viewer && (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom)) viewer.close();
  });
  const selectCategory = id => {
    stopRotation();
    const next = groups.find(group => group.dataset.categoryPanel === id);
    const changed = next !== selectedGroup;
    selectedGroup = next;
    groups.forEach(group => { group.hidden = group !== selectedGroup; });
    categoryButtons.forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.category === id));
    });
    description.textContent = selectedGroup.dataset.description;
    if (changed && !reducedMotion.matches) {
      selectedGroup.animate([{opacity: 0}, {opacity: 1}], {duration: 220, easing: 'ease-out'});
    }
    videos.forEach(updatePlayback);
    updateRotation();
  };
  const selectFromHash = () => {
    const id = location.hash.slice(1);
    const example = document.getElementById(id);
    const group = groups.find(item => item.dataset.categoryPanel === id) ||
      (example?.matches('.motion-result') && example.closest('.result-group'));
    if (!group) return false;
    selectCategory(group.dataset.categoryPanel);
    if (mobileView.matches) document.documentElement.classList.add('demo-nav-collapsed');
    (example?.matches('.motion-result') ? example : showcase)
      .scrollIntoView({block: 'start', behavior: 'instant'});
    scrollReference = window.scrollY;
    wasInDemo = mobileView.matches;
    return true;
  };

  categoryButtons.forEach(button => button.addEventListener('click', () => {
    const returnToTop = firstRowTop() < resultsViewportTop() - 16;
    selectCategory(button.dataset.category);
    history.replaceState(null, '', `#${button.dataset.category}`);
    if (returnToTop) {
      document.documentElement.classList.remove('demo-nav-collapsed');
      showcase.scrollIntoView({block: 'start', behavior: reducedMotion.matches ? 'instant' : 'smooth'});
    }
  }));
  groups.forEach(group => {
    group.addEventListener('pointerenter', updateRotation);
    group.addEventListener('pointerleave', updateRotation);
    group.addEventListener('focusin', updateRotation);
    group.addEventListener('focusout', () => queueMicrotask(updateRotation));
    group.querySelectorAll('details').forEach(details => {
      details.addEventListener('toggle', updateRotation);
    });
    group.querySelectorAll('.motion-result').forEach(result => {
      const video = result.querySelector('video');
      const steps = [...result.querySelectorAll('.prompt-step')];
      const mobilePrompt = result.querySelector('.mobile-prompt');
      const playButton = result.querySelector('.motion-play');
      const framingButton = result.querySelector('.framing-toggle');
      const subjects = [...result.querySelectorAll('[data-subject]')];
      const staticPrompts = [...result.querySelectorAll('.prompt-static')];
      let selectedSubject = 0;
      let pendingSeek = null;
      const updatePlayButton = () => {
        playButton.dataset.playing = String(!video.paused);
        playButton.setAttribute('aria-label', `${video.paused ? 'Play' : 'Pause'} ${video.getAttribute('aria-label')}`);
      };
      const updatePrompts = () => {
        steps.forEach(step => {
          if (video.currentTime >= Number(step.dataset.start) &&
              video.currentTime < Number(step.dataset.end)) {
            step.setAttribute('aria-current', 'step');
          } else step.removeAttribute('aria-current');
        });
        const current = result.querySelector('.prompt-step[aria-current="step"] span:last-child');
        const text = current?.textContent || staticPrompts[selectedSubject]?.textContent || '';
        if (mobilePrompt.textContent !== text) mobilePrompt.textContent = text;
      };
      playButton.addEventListener('click', () => {
        if (video.paused) play(video);
        else video.pause();
      });
      framingButton.setAttribute('aria-haspopup', 'dialog');
      framingButton.setAttribute('aria-controls', 'motion-viewer');
      framingButton.addEventListener('click', () => openViewer(result));
      subjects.forEach(button => button.addEventListener('click', () => {
        selectedSubject = Number(button.dataset.subject);
        subjects.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
        result.style.setProperty('--crop-x', button.dataset.cropX);
        updatePrompts();
      }));
      updateFraming(result);
      updatePlayButton();
      result.addEventListener('click', event => {
        const step = event.target.closest('.prompt-step');
        if (!step) return;
        pendingSeek = Number(step.dataset.start);
        if (video.readyState >= 1) {
          video.currentTime = pendingSeek;
          pendingSeek = null;
          updatePrompts();
        }
        play(video);
      });
      video.addEventListener('timeupdate', updatePrompts);
      video.addEventListener('loadedmetadata', () => {
        if (pendingSeek !== null) {
          video.currentTime = pendingSeek;
          pendingSeek = null;
        }
        updatePrompts();
      });
      video.addEventListener('play', updateRotation);
      video.addEventListener('pause', updateRotation);
      video.addEventListener('play', updatePlayButton);
      video.addEventListener('pause', updatePlayButton);
      video.addEventListener('webkitbeginfullscreen', stopRotation);
      video.addEventListener('webkitendfullscreen', updateRotation);
      updatePrompts();
    });
  });
  new IntersectionObserver(([entry]) => {
    tabsVisible = entry.isIntersecting && entry.intersectionRatio >= 0.5;
    updateRotation();
  }, {threshold: 0.5}).observe(tabs);
  let rotationFrame = null;
  const scheduleRotationUpdate = () => {
    if (rotationFrame !== null) return;
    rotationFrame = requestAnimationFrame(() => {
      rotationFrame = null;
      updateDemoNavigation();
      updateRotation();
    });
  };
  window.addEventListener('scroll', scheduleRotationUpdate, {passive: true});
  window.addEventListener('resize', scheduleRotationUpdate);
  tabs.addEventListener('transitionend', scheduleRotationUpdate);
  header.addEventListener('focusin', scheduleRotationUpdate);
  worksMenu.addEventListener('toggle', scheduleRotationUpdate);
  document.addEventListener('visibilitychange', updateRotation);
  document.addEventListener('fullscreenchange', updateRotation);
  reducedMotion.addEventListener('change', updateRotation);
  window.addEventListener('hashchange', () => {
    if (viewer.open) {
      viewer.close();
      closeViewer(false);
    }
    selectFromHash();
  });
  if (!selectFromHash()) selectCategory(selectedGroup.dataset.categoryPanel);

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
    videos.forEach(updatePlayback);
  });
})();
