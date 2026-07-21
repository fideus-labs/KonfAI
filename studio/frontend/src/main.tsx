import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

// A session that expires mid-use makes every API call 401. Surface that once as a `ks-unauth` event so
// the app can drop back to the lock screen instead of silently failing. No-op when auth is off.
const nativeFetch = window.fetch.bind(window);
window.fetch = async (input, init) => {
  const resp = await nativeFetch(input, init);
  if (resp.status === 401) window.dispatchEvent(new Event("ks-unauth"));
  return resp;
};

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
