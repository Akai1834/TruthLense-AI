(() => {
  const config = window.TRUTHLENSE_CONFIG || {};
  const form = document.getElementById("loginForm");
  const emailInput = document.getElementById("emailInput");
  const passwordInput = document.getElementById("passwordInput");
  const formMessage = document.getElementById("formMessage");
  const submitButton = document.getElementById("submitButton");
  const createAccountButton = document.getElementById("createAccountButton");
  const params = new URLSearchParams(window.location.search);
  const requestedUpload = params.get("upload");
  const uploadKind = ["image", "video"].includes(requestedUpload) ? requestedUpload : null;
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

  if (!client) {
    showError("Authentication is not configured yet. Add your Supabase project URL and publishable key to config.js.");
    form.querySelector("button[type=submit]").disabled = true;
    createAccountButton.disabled = true;
    document.querySelector('[data-provider="Google"]').disabled = true;
    return;
  }

  client.auth
    .getSession()
    .then(({ data, error }) => {
      if (error) throw error;
      if (data.session) window.location.replace(getHomeUrl());
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
      const { error } = await client.auth.signInWithPassword({
        email: emailInput.value.trim(),
        password: passwordInput.value
      });
      if (error) throw error;
      passwordInput.value = "";
      window.location.assign(getHomeUrl());
    } catch (error) {
      console.error("Email sign-in failed.", error);
      showError(error.message || "Sign-in failed. Check your credentials and try again.");
    } finally {
      submitButton.disabled = false;
    }
  });

  createAccountButton.addEventListener("click", async () => {
    if (!form.reportValidity()) return;
    createAccountButton.disabled = true;
    formMessage.textContent = "";

    try {
      const { data, error } = await client.auth.signUp({
        email: emailInput.value.trim(),
        password: passwordInput.value,
        options: { emailRedirectTo: new URL(getHomeUrl(), window.location.href).href }
      });
      if (error) throw error;

      if (data.session) {
        passwordInput.value = "";
        window.location.assign(getHomeUrl());
        return;
      }
      showMessage("Account created. Check your email to confirm it, then log in.");
    } catch (error) {
      console.error("Account creation failed.", error);
      showError(error.message || "Could not create the account. Please try again.");
    } finally {
      createAccountButton.disabled = false;
    }
  });

  document.querySelector('[data-provider="Google"]').addEventListener("click", async () => {
    try {
      const { error } = await client.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: new URL(getHomeUrl(), window.location.href).href }
      });
      if (error) throw error;
    } catch (error) {
      console.error("Google sign-in failed.", error);
      showError(error.message || "Google sign-in is not configured for this project.");
    }
  });
})();
