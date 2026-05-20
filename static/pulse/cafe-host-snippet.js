/**
 * Drop into your Café Web app to accept Jarvis Pulse autologin handoff.
 * Listens for postMessage { type: 'jarvis-cafe-login', username, password }.
 */
(function () {
  "use strict";

  const DEFAULT_SELECTORS = {
    username: "input[name=username], input[name=email], #username, #email",
    password: "input[type=password], #password",
    submit: "button[type=submit], input[type=submit]",
  };

  function fillLogin(payload, selectors) {
    const userEl = document.querySelector(selectors.username);
    const passEl = document.querySelector(selectors.password);
    const submitEl = document.querySelector(selectors.submit);
    if (userEl && payload.username) userEl.value = payload.username;
    if (passEl && payload.password) passEl.value = payload.password;
    if (submitEl) submitEl.click();
  }

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.type !== "jarvis-cafe-login") return;
    fillLogin(data, DEFAULT_SELECTORS);
  });

  const user = sessionStorage.getItem("jarvis-cafe-user");
  const pass = sessionStorage.getItem("jarvis-cafe-pass");
  if (user && pass) {
    fillLogin({ username: user, password: pass }, DEFAULT_SELECTORS);
    sessionStorage.removeItem("jarvis-cafe-user");
    sessionStorage.removeItem("jarvis-cafe-pass");
  }
})();
