(() => {
  'use strict';
  const videos = [...document.querySelectorAll('video:not(#hero-video)')];
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
  const updatePlayback = video => {
    const state = playback.get(video);
    if (state.visible && !document.hidden && !video.closest('[hidden]') &&
        !state.pausedByUser && !video.ended) {
      if (video.paused) video.play().catch(() => {});
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
    if (heroVisible && !document.hidden && !reducedMotion.matches) {
      hero.play().catch(() => {});
    } else hero.pause();
  };
  reducedMotion.addEventListener('change', updateHero);
  new IntersectionObserver(([entry]) => {
    heroVisible = entry.isIntersecting;
    updateHero();
  }).observe(hero);
  document.addEventListener('visibilitychange', updateHero);

  document.querySelector('#watch-film').addEventListener('click', () => {
    const film = document.querySelector('#overview-video');
    play(film);
  });

  videos.forEach(video => {
    const state = playback.get(video);
    video.muted = true;
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
      playback.get(entry.target).visible = entry.isIntersecting && entry.intersectionRatio >= 0.1;
      updatePlayback(entry.target);
    });
  }, {threshold: 0.1});
  videos.forEach(video => videoObserver.observe(video));

  const showcase = document.querySelector('#continuous-transitions');
  const tabs = document.querySelector('.showcase-tabs');
  const categoryButtons = [...document.querySelectorAll('.category-choice')];
  const groups = [...document.querySelectorAll('.result-group')];
  const description = document.querySelector('#motion-description');
  let selectedGroup = groups.find(group => !group.hidden);
  let tabsVisible = false;
  let rotationTimer = null;

  const stopRotation = () => {
    clearTimeout(rotationTimer);
    rotationTimer = null;
  };
  const canRotate = () => tabsVisible && !document.hidden && !reducedMotion.matches &&
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
    rotationTimer = setTimeout(() => {
      stopRotation();
      if (canRotate()) {
        const next = groups[(groups.indexOf(selectedGroup) + 1) % groups.length];
        selectCategory(next.dataset.categoryPanel);
      }
    }, Number(selectedGroup.dataset.rotationSeconds || 20) * 1000);
  };
  const selectCategory = id => {
    stopRotation();
    selectedGroup = groups.find(group => group.dataset.categoryPanel === id);
    groups.forEach(group => { group.hidden = group !== selectedGroup; });
    categoryButtons.forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.category === id));
    });
    description.textContent = selectedGroup.dataset.description;
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
    (example?.matches('.motion-result') ? example : showcase)
      .scrollIntoView({block: 'start', behavior: 'instant'});
    return true;
  };

  categoryButtons.forEach(button => button.addEventListener('click', () => {
    selectCategory(button.dataset.category);
    history.replaceState(null, '', `#${button.dataset.category}`);
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
      let pendingSeek = null;
      const updatePrompts = () => {
        steps.forEach(step => {
          if (video.currentTime >= Number(step.dataset.start) &&
              video.currentTime < Number(step.dataset.end)) {
            step.setAttribute('aria-current', 'step');
          } else step.removeAttribute('aria-current');
        });
      };
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
      video.addEventListener('webkitbeginfullscreen', stopRotation);
      video.addEventListener('webkitendfullscreen', updateRotation);
      updatePrompts();
    });
  });
  new IntersectionObserver(([entry]) => {
    tabsVisible = entry.isIntersecting && entry.intersectionRatio >= 0.5;
    updateRotation();
  }, {threshold: 0.5, rootMargin: `-${header.offsetHeight}px 0px 0px 0px`}).observe(tabs);
  document.addEventListener('visibilitychange', updateRotation);
  document.addEventListener('fullscreenchange', updateRotation);
  reducedMotion.addEventListener('change', updateRotation);
  window.addEventListener('hashchange', selectFromHash);
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
