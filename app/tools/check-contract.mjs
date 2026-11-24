// Validate an export directory with the SAME validator the app uses in the browser.
//   node app/tools/check-contract.mjs [app/data]
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { validate } from "../validate.js";

const here = dirname(fileURLToPath(import.meta.url));
const dir = process.argv[2] || join(here, "..", "data");
const read = (p) => JSON.parse(readFileSync(join(dir, p), "utf8"));
const schema = (n) => read(`schemas/${n}.schema.json`);
let files = 0, failed = 0;
const check = (path, name) => {
  files += 1;
  const errors = validate(read(path), schema(name));
  if (errors.length) { failed += 1; console.error(`FAIL ${path}\n  ${errors.join("\n  ")}`); }
};
const m = read("manifest.json");
check("manifest.json", "manifest");
check(m.files.teams, "teams");
check(m.files.games, "games");
check(m.files.scorecard, "scorecard");
for (const name of ["tendency_validation", "hot_hand"]) if (m.files[name]) check(m.files[name], name);   // contract 1.2, optional
for (const f of readdirSync(join(dir, m.files.team_reports_dir))) check(`${m.files.team_reports_dir}/${f}`, "team_report");
for (const f of readdirSync(join(dir, m.files.replays_dir))) check(`${m.files.replays_dir}/${f}`, "replay");

// The validator must also reject broken documents, or passing means nothing.
const bad = { ...m, data_kind: "probably real" };
delete bad.coverage;
const caught = validate(bad, schema("manifest")).length;
if (caught < 2) { failed += 1; console.error("FAIL validator accepted a deliberately broken manifest"); }
console.log(`${files - failed}/${files} files valid; broken-manifest check caught ${caught} errors`);
process.exit(failed ? 1 : 0);
