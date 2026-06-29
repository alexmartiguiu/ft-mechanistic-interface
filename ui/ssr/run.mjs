import esbuild from "esbuild";
import { pathToFileURL } from "url";
import path from "path";

const outfile = path.resolve("ssr/out.mjs");
await esbuild.build({
  entryPoints: ["ssr/smoke-entry.jsx"],
  bundle: true,
  format: "esm",
  platform: "node",
  jsx: "automatic",
  outfile,
  external: ["react", "react-dom", "react-dom/server", "react/jsx-runtime"],
  logLevel: "warning",
});
const mod = await import(pathToFileURL(outfile).href);
const sizes = mod.run();
console.log("SSR render OK — html lengths:");
for (const [k, v] of Object.entries(sizes)) console.log(`  ${k}: ${v}`);
