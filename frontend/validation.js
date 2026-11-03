(function () {
  function clearErrors(form) {
    form.querySelectorAll(".field-error").forEach(e => e.remove());
    form.querySelectorAll(".invalid").forEach(e => e.classList.remove("invalid"));
    const b = document.getElementById("errorBanner");
    if (b) { b.textContent = ""; b.classList.remove("show"); }
  }

  function showError(input, msg) {
    input.classList.add("invalid");
    const d = document.createElement("div");
    d.className = "field-error";
    d.textContent = msg;
    input.insertAdjacentElement("afterend", d);
  }
  
  // Returns an object of clean values, or null if anything is invalid (errors are shown on screen).
  function validateForm(form, rules) {
    clearErrors(form);
    const values = {};
    let firstBad = null;

    for (const [id, r] of Object.entries(rules)) {
      const input = document.getElementById(id);
      const raw = input.value.trim();
      const isText = r.type === "select" || r.type === "text";
      let err = null;

      if (!isText && input.validity && input.validity.badInput) {
        err = `${r.label} must be a number.`;
      } else if (raw === "") {
        if (r.required !== false) err = `${r.label} is required.`;
        else values[id] = null;
      } else if (isText) {
        values[id] = raw;
      } else {
        const n = Number(raw);
        if (!Number.isFinite(n)) err = `${r.label} must be a number.`;
        else if (r.type === "int" && !Number.isInteger(n)) err = `${r.label} must be a whole number.`;
        else if (r.min !== undefined && n < r.min) err = `${r.label} must be at least ${r.min}.`;
        else if (r.max !== undefined && n > r.max) err = `${r.label} must be at most ${r.max}.`;
        else values[id] = n;
      }

      if (err) { showError(input, err); firstBad = firstBad || input; }
    }

    if (firstBad) {
      const b = document.getElementById("errorBanner");
      if (b) { b.textContent = "Please fix the highlighted fields before submitting."; b.classList.add("show"); }
      firstBad.focus();
      return null;
    }
    return values;
  }

  // Uses the Render URL from config.js; falls back to the local address while it is still the placeholder.
  function apiOrigin(localFallback) {
    const base = (window.API_BASE_URL || "").trim().replace(/\/+$/, "");
    if (!base || base.includes("YOUR-SERVICE-NAME")) return localFallback;
    return base;
  }

  // Turns a failed request into a message a person can understand.
  function describeError(err) {
    if (err && (err.name === "TimeoutError" || err.name === "AbortError")) {
      return "The server took too long to respond. If it is hosted on Render's free tier it may be waking up. Wait a minute and try again.";
    }
    if (err instanceof TypeError) {
      return "Could not reach the API. Check your internet connection, that the API URL in config.js is correct, and that the server is running.";
    }
    return "Request failed: " + (err && err.message ? err.message : err);
  }

  window.FormValidation = { validateForm, clearErrors, showError, apiOrigin, describeError };
})();
