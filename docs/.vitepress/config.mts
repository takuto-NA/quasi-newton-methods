import { defineConfig } from "vitepress";

export default defineConfig({
  title: "quasi-newton-methods",
  description: "Evidence-first implementations and benchmarks for quasi-Newton methods",
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "References", link: "/references/implementations" },
      { text: "Evidence", link: "/evidence/methodology" }
    ],
    sidebar: [
      {
        text: "References",
        items: [
          { text: "Implementations", link: "/references/implementations" },
          { text: "SciPy", link: "/references/scipy" },
          { text: "Articles", link: "/references/articles" },
          { text: "Papers", link: "/references/papers" }
        ]
      },
      {
        text: "Evidence",
        items: [
          { text: "Methodology", link: "/evidence/methodology" },
          { text: "Baseline Results", link: "/evidence/baseline_results" }
        ]
      }
    ]
  }
});

