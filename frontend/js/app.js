(() => {
  const config = window.TRUTHLENSE_CONFIG || {};
  const isConfigured =
    typeof config.supabaseUrl === "string" &&
    config.supabaseUrl.startsWith("https://") &&
    !config.supabaseUrl.includes("YOUR-PROJECT") &&
    typeof config.supabaseAnonKey === "string" &&
    config.supabaseAnonKey.length > 20 &&
    !config.supabaseAnonKey.includes("YOUR_SUPABASE");
  const supabaseLibrary = window.supabase;
  const supabaseClient =
    isConfigured && supabaseLibrary?.createClient
      ? supabaseLibrary.createClient(config.supabaseUrl, config.supabaseAnonKey)
      : null;
  const loginButton = document.getElementById("loginButton");
  const loginWrap = document.getElementById("loginWrap");
  const loginMenu = document.getElementById("loginMenu");
  const googleLoginButton = document.getElementById("googleLoginButton");
  const socialProviderButtons = document.querySelectorAll("[data-social-provider]");
  const profileWrap = document.getElementById("profileWrap");
  const profileButton = document.getElementById("profileButton");
  const profileMenu = document.getElementById("profileMenu");
  const uploadState = document.getElementById("uploadState");
  const analysisActions = document.getElementById("analysisActions");
  const analyzeButton = document.getElementById("analyzeButton");
  const analysisResult = document.getElementById("analysisResult");
  const historySection = document.getElementById("historySection");
  const historyList = document.getElementById("historyList");
  const preview = document.getElementById("preview");
  const fileName = document.getElementById("fileName");
  const fileStatus = document.getElementById("fileStatus");
  const toast = document.getElementById("toast");
  let toastTimer;
  let previewUrl;
  let selectedFile = null;
  let activeSession = null;

  document.getElementById("year").textContent = new Date().getFullYear();

  function showToast(message) {
    toast.textContent = message;
    toast.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toast.hidden = true;
    }, 5000);
  }

  function renderUser(user) {
    const isLoggedIn = Boolean(user);
    loginWrap.hidden = isLoggedIn;
    profileWrap.hidden = !isLoggedIn;
    loginMenu.hidden = true;
    loginButton.setAttribute("aria-expanded", "false");
    if (user) {
      historySection.hidden = false;
      const email = user.email || "Signed-in user";
      const fullName = user.user_metadata?.full_name || user.user_metadata?.name || email.split("@")[0] || email;
      document.getElementById("profileName").textContent = fullName;
      document.getElementById("profileEmail").textContent = email;
      document.getElementById("avatar").textContent = fullName.charAt(0).toUpperCase();
    } else {
      historySection.hidden = true;
      historyList.replaceChildren();
      profileMenu.hidden = true;
      profileButton.setAttribute("aria-expanded", "false");
      analysisActions.hidden = true;
      selectedFile = null;
      uploadState.classList.remove("is-visible");
      analysisResult.hidden = true;
      preview.replaceChildren();
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
      }
    }
  }

  function openFilePicker(kind) {
    if (!activeSession?.user) {
      const target = `login.html?upload=${encodeURIComponent(kind)}&message=login_required`;
      window.location.href = target;
      return;
    }
    document.getElementById(kind === "image" ? "imageInput" : "videoInput").click();
  }

  function handleFile(file, kind) {
    if (!file) return;
    const expectedType = kind === "image" ? "image/" : "video/";
    if (!file.type.startsWith(expectedType)) {
      showToast(`Please select a valid ${kind} file.`);
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      showToast("Please choose a file smaller than 25 MB.");
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    selectedFile = file;
    analysisResult.hidden = true;
    preview.replaceChildren();

    if (kind === "image") {
      const image = document.createElement("img");
      image.src = previewUrl;
      image.alt = "Selected image preview";
      preview.append(image);
    } else {
      const video = document.createElement("video");
      video.src = previewUrl;
      video.muted = true;
      video.playsInline = true;
      preview.append(video);
    }

    fileName.textContent = file.name;
    fileStatus.textContent = `${kind === "image" ? "Image" : "Video"} ready · private until you request analysis`;
    uploadState.classList.add("is-visible");
    analysisActions.hidden = false;
    showToast("File selected. Click “Test the reality” when you are ready.");
  }

  function showAnalysis(result) {
    const positiveScore = Math.round(result.positive_percent);
    const realScore = Math.round(result.real_percent);
    const isVideo = result.media_type === "video";

    document.getElementById("resultTitle").textContent = "Model analysis estimate";
    document.getElementById("resultSummary").textContent = isVideo
      ? `Estimated from ${result.frames_analyzed} sampled video frames.`
      : "Estimated from the selected image.";
    document.querySelector("#aiMeter").previousElementSibling.textContent =
      isVideo ? "Manipulated video" : "Deepfake image";
    document.querySelector("#realMeter").previousElementSibling.textContent =
      isVideo ? "Unmanipulated video" : "Real-media";
    document.getElementById("aiScore").textContent = `${positiveScore}%`;
    document.getElementById("realScore").textContent = `${realScore}%`;
    document.getElementById("aiScoreFill").style.width = `${positiveScore}%`;
    document.getElementById("realScoreFill").style.width = `${realScore}%`;
    document.getElementById("aiMeter").setAttribute("aria-valuenow", String(positiveScore));
    document.getElementById("realMeter").setAttribute("aria-valuenow", String(realScore));
    document.getElementById("analysisDisclaimer").textContent = result.disclaimer;
    analysisResult.hidden = false;
  }

  function renderHistory(records) {
    historyList.replaceChildren();
    if (!records.length) {
      const empty = document.createElement("p");
      empty.className = "history-empty";
      empty.textContent = "Your analysis results will appear here. Only you can see them.";
      historyList.append(empty);
      return;
    }

    for (const record of records) {
      const item = document.createElement("div");
      item.className = "history-item";
      const label = document.createElement("span");
      const score = document.createElement("strong");
      const mediaLabel =
        record.media_type === "video" ? "Manipulated video" : "Deepfake image";
      const date = new Date(record.created_at);
      label.textContent = `${mediaLabel} · ${
        Number.isNaN(date.getTime()) ? "Date unavailable" : date.toLocaleString()
      }`;
      score.textContent = `${Math.round(Number(record.positive_percent))}%`;
      item.append(label, score);
      historyList.append(item);
    }
  }

  async function loadHistory(session) {
    if (!session?.access_token || session.access_token.startsWith("social-") || !config.apiBaseUrl) return;
    historySection.hidden = false;
    historyList.replaceChildren();
    const loading = document.createElement("p");
    loading.className = "history-empty";
    loading.textContent = "Loading your private results…";
    historyList.append(loading);

    try {
      const response = await fetch(`${config.apiBaseUrl.replace(/\/+$/, "")}/api/analyses`, {
        headers: { Authorization: `Bearer ${session.access_token}` }
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(result.detail || `Could not load results (${response.status}).`);
      }
      renderHistory(result);
    } catch (error) {
      console.error("Could not load this user's analysis history.", error);
      historyList.replaceChildren();
      const failure = document.createElement("p");
      failure.className = "history-empty";
      failure.textContent = error.message || "Could not load your private results.";
      historyList.append(failure);
    }
  }

  async function analyzeSelectedFile() {
    if (!selectedFile || !activeSession?.access_token) {
      showToast("Please sign in and select an image or video first.");
      return;
    }
    if (!config.apiBaseUrl || config.apiBaseUrl.includes("YOUR-API")) {
      showToast("The analysis service is not configured yet.");
      return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing…";
    fileStatus.textContent = "Sending securely for analysis…";
    analysisResult.hidden = true;

    try {
      const body = new FormData();
      body.append("file", selectedFile, selectedFile.name);
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), 180000);
      let response;
      try {
        response = await fetch(`${config.apiBaseUrl.replace(/\/+$/, "")}/api/analyze`, {
          method: "POST",
          headers: { Authorization: `Bearer ${activeSession.access_token}` },
          body,
          signal: controller.signal
        });
      } finally {
        window.clearTimeout(timeoutId);
      }

      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(result.detail || `Analysis failed (${response.status}).`);
      }
      showAnalysis(result);
      fileStatus.textContent = "Analysis complete · file was not saved by the service";
      await loadHistory(activeSession);
    } catch (error) {
      console.error("Media analysis request failed.", error);
      const message =
        error.name === "AbortError"
          ? "Analysis timed out. Try again with a shorter video."
          : error.message || "Could not reach the analysis service.";
      fileStatus.textContent = message;
      showToast(message);
    } finally {
      analyzeButton.disabled = false;
      analyzeButton.textContent = "Test the reality";
    }
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !profileMenu.hidden) {
      profileMenu.hidden = true;
      profileButton.setAttribute("aria-expanded", "false");
    }
  });

  loginButton.addEventListener("click", (event) => {
    if (!supabaseClient) {
      window.location.href = "login.html";
      return;
    }
    event.preventDefault();
    const willOpen = loginMenu.hidden;
    loginMenu.hidden = !willOpen;
    loginButton.setAttribute("aria-expanded", String(willOpen));
  });

  googleLoginButton.addEventListener("click", async () => {
    if (!supabaseClient) {
      window.location.href = "login.html";
      return;
    }
    loginMenu.hidden = true;
    loginButton.setAttribute("aria-expanded", "false");
    try {
      const { error } = await supabaseClient.auth.signInWithOAuth({
        provider: "google",
        options: {
          redirectTo: new URL("index.html", window.location.href).href
        }
      });
      if (error) throw error;
    } catch (error) {
      console.error("Google sign-in failed.", error);
      showToast(error.message || "Google sign-in failed. Please try again.");
    }
  });

  socialProviderButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const providerName = button.dataset.socialProvider;
      const provider = { Facebook: "facebook", X: "x" }[providerName];
      loginMenu.hidden = true;
      loginButton.setAttribute("aria-expanded", "false");
      if (!provider) {
        showToast("Instagram login needs a separate Instagram OAuth integration. No social password is collected here.");
        return;
      }
      if (!supabaseClient) {
        window.location.href = "login.html";
        return;
      }
      try {
        const { error } = await supabaseClient.auth.signInWithOAuth({
          provider,
          options: { redirectTo: new URL("index.html", window.location.href).href }
        });
        if (error) throw error;
      } catch (error) {
        console.error(`${providerName} sign-in failed.`, error);
        showToast(error.message || `${providerName} sign-in failed. Please try again.`);
      }
    });
  });

  profileButton.addEventListener("click", () => {
    const willOpen = profileMenu.hidden;
    profileMenu.hidden = !willOpen;
    profileButton.setAttribute("aria-expanded", String(willOpen));
  });

  document.getElementById("logoutButton").addEventListener("click", async () => {
    if (!supabaseClient) return;
    const { error } = await supabaseClient.auth.signOut();
    if (error) {
      console.error("Unable to sign out.", error);
      showToast("Could not log out. Please try again.");
      return;
    }
    showToast("You have logged out.");
  });

  document.getElementById("imageUploadButton").addEventListener("click", () => openFilePicker("image"));
  document.getElementById("videoUploadButton").addEventListener("click", () => openFilePicker("video"));
  document.getElementById("imageInput").addEventListener("change", (event) => {
    handleFile(event.target.files[0], "image");
    event.target.value = "";
  });
  document.getElementById("videoInput").addEventListener("change", (event) => {
    handleFile(event.target.files[0], "video");
    event.target.value = "";
  });
  analyzeButton.addEventListener("click", analyzeSelectedFile);

  if (!supabaseClient) {
    return;
  }

  supabaseClient.auth.onAuthStateChange((_event, session) => {
    activeSession = session;
    renderUser(session?.user || null);
    if (session?.user) void loadHistory(session);
  });

  supabaseClient.auth
    .getSession()
    .then(async ({ data, error }) => {
      if (error) throw error;
      activeSession = data.session;
      renderUser(data.session?.user || null);
      if (data.session?.user) {
        await loadHistory(data.session);
      }
      const requestedUpload = new URLSearchParams(window.location.search).get("upload");
      if (data.session?.user && ["image", "video"].includes(requestedUpload)) {
        const label = requestedUpload === "image" ? "an image" : "a video";
        showToast(`You're logged in. Click “Upload ${label}” to choose your file.`);
        window.history.replaceState({}, "", window.location.pathname);
      }
    })
    .catch((error) => {
      console.error("Unable to restore the sign-in session.", error);
      showToast("Could not restore your sign-in session. Please refresh or log in again.");
    });
})();
