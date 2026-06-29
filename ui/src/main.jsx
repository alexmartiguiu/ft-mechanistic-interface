import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/app.css";

// NB: no React.StrictMode. The pipeline narration is driven by setTimeout chains;
// StrictMode's intentional double-invoke of effects (dev only) cancels those timers
// on the simulated unmount and the fireOnce guards block re-scheduling, so the agent
// stream stalls after the first line. Dev now behaves like the production build.
createRoot(document.getElementById("root")).render(<App />);
