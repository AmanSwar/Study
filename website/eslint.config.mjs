import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Static design system + vendored KaTeX/highlight.js for HTML study modules (plain ES5/ES2020, not TS).
    "public/**",
    // One-off migration script; its inputs were deleted after it ran.
    "scripts/**",
  ]),
]);

export default eslintConfig;
