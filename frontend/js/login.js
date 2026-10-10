(() => {
  const config = window.TRUTHLENSE_CONFIG || {};
  const form = document.getElementById("loginForm");
  const emailInput = document.getElementById("emailInput");
  const passwordInput = document.getElementById("passwordInput");
  const nameInput = document.getElementById("nameInput");
  const confirmPasswordInput = document.getElementById("confirmPasswordInput");
  const createAccountFields = document.getElementById("createAccountFields");
  const confirmPasswordWrap = document.getElementById("confirmPasswordWrap");
  const formMessage = document.getElementById("formMessage");
  const submitButton = document.getElementById("submitButton");
  const createAccountButton = document.getElementById("createAccountButton");
  const socialProviderButtons = document.querySelectorAll("[data-provider]");
  const params = new URLSearchParams(window.location.search);
  const requestedUpload = params.get("upload");
  const uploadKind = ["image", "video"].includes(requestedUpload) ? requestedUpload : null;
  const loginPrompt = params.get("message") === "login_required";
  const startInCreateMode = params.get("mode") === "signup";
  const isConfigured =
    typeof config.supabaseUrl === "string" &&
    config.supabaseUrl.startsWith("https://") &&
    !config.supabaseUrl.includes("YOUR-PROJECT") &&
    typeof config.supabaseAnonKey === "string" &&
    config.supabaseAnonKey.length > 20 &&
    !config.supabaseAnonKey.includes("YOUR_SUPABASE");
  const client =
    isConfigured && window.supabase?.createClient
      ? window.supabase.createClient(config.supabaseUrl, config.supabaseAnonKey)
      : null;
  let isCreateMode = false;

  function showError(message) {
    formMessage.textContent = message;
    formMessage.style.color = "#f1a99d";
  }

  function showMessage(message) {
    formMessage.textContent = message;
    formMessage.style.color = "var(--mint)";
  }

  function getHomeUrl() {
    return uploadKind ? `index.html?upload=${uploadKind}` : "index.html";
  }

  function setCreateMode(enabled) {
    isCreateMode = enabled;
    createAccountFields.hidden = !enabled;
    confirmPasswordWrap.hidden = !enabled;
    nameInput.required = enabled;
    confirmPasswordInput.required = enabled;
    submitButton.textContent = enabled ? "Create account" : "Log in with email";
    createAccountButton.textContent = enabled ? "Back to login" : "Create an account";

    if (!enabled) {
      nameInput.value = "";
      confirmPasswordInput.value = "";
      formMessage.textContent = "";
    }
  }

  async function saveUserProfile(user, fullName) {
    if (!user || !user.id || !client) return;
    const profileName = (fullName || user.user_metadata?.full_name || user.user_metadata?.name || user.email || "User").trim() || "User";
    const { error } = await client.from("profiles").upsert(
      {
        id: user.id,
        name: profileName,
        email: user.email,
        created_at: new Date().toISOString()
      },
      { onConflict: "id" }
    );

    if (error) {
      console.warn("Profile save warning:", error.message || error);
    }
  }

  if (loginPrompt) {
    showError("Please log in to upload an image or video.");
  }

  if (startInCreateMode) {
    setCreateMode(true);
    document.querySelector("h1").textContent = "Create your account";
    document.querySelector(".intro").textContent = "Create an account to sign in and privately test images or videos.";
  }

  if (!client) {
    showError("Authentication is not configured yet. Check frontend/config.js for your Supabase project URL and publishable key.");
    form.querySelector("button[type=submit]").disabled = true;
    createAccountButton.disabled = true;
    socialProviderButtons.forEach((button) => { button.disabled = true; });
    return;
  }

  client.auth
    .getSession()
    .then(async ({ data, error }) => {
      if (error) throw error;
      if (data.session) {
        await saveUserProfile(data.session.user, data.session.user?.user_metadata?.full_name || data.session.user?.user_metadata?.name || data.session.user?.email);
        window.location.replace(getHomeUrl());
      }
    })
    .catch((error) => {
      console.error("Unable to check the authentication session.", error);
      showError("Could not check your sign-in session. Please refresh and try again.");
    });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;

    submitButton.disabled = true;
    formMessage.textContent = "";

    try {
      const email = emailInput.value.trim();
      const password = passwordInput.value;

      if (isCreateMode) {
        const fullName = nameInput.value.trim();
        const confirmPassword = confirmPasswordInput.value;

        if (!fullName) {
          throw new Error("Please enter your full name.");
        }
        if (password.length < 6) {
          throw new Error("Password must be at least 6 characters long.");
        }
        if (password !== confirmPassword) {
          throw new Error("Passwords do not match.");
        }

        const { data, error } = await client.auth.signUp({
          email,
          password,
          options: {
            data: { full_name: fullName },
            emailRedirectTo: new URL(getHomeUrl(), window.location.href).href
          }
        });

        if (error) throw error;

        if (data.user) {
          await saveUserProfile(data.user, fullName);
        }

        if (data.session) {
          passwordInput.value = "";
          confirmPasswordInput.value = "";
          nameInput.value = "";
          window.location.assign(getHomeUrl());
          return;
        }

        showMessage("Account created, but Supabase requires email confirmation before sign-in. Confirm the email to continue, or disable Confirm email in Supabase Authentication settings for immediate sign-in.");
        return;
      }

      const { data, error } = await client.auth.signInWithPassword({ email, password });
      if (error) throw error;
      if (data.user) {
        await saveUserProfile(data.user, data.user.user_metadata?.full_name || data.user.user_metadata?.name || data.user.email);
      }
      passwordInput.value = "";
      window.location.assign(getHomeUrl());
    } catch (error) {
      console.error("Authentication failed.", error);
      showError(error.message || "Authentication failed. Please try again.");
    } finally {
      submitButton.disabled = false;
    }
  });

  createAccountButton.addEventListener("click", () => {
    setCreateMode(!isCreateMode);
  });

  socialProviderButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const providerName = button.dataset.provider;
      const provider = { Google: "google", Facebook: "facebook", X: "x" }[providerName];
      if (!provider) {
        showError("Instagram login needs a separate Instagram OAuth integration. No social password is collected here.");
        return;
      }

      button.disabled = true;
      try {
        const { error } = await client.auth.signInWithOAuth({
          provider,
          options: { redirectTo: new URL(getHomeUrl(), window.location.href).href }
        });
        if (error) throw error;
      } catch (error) {
        console.error(`${providerName} sign-in failed.`, error);
        showError(error.message || `${providerName} sign-in is not configured. Please use email login instead.`);
        button.disabled = false;
      }
    });
  });

  setCreateMode(startInCreateMode);
})();
