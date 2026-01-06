import { defineConfig } from "vitepress";
import mathjax3 from "markdown-it-mathjax3";

export default defineConfig({
  title: "quasi-newton-methods",
  description: "Evidence-first implementations and benchmarks for quasi-Newton methods",
  base: "/quasi-newton-methods/",
  markdown: {
    config: (md) => {
      md.use(mathjax3);
    },
  },
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
      { text: "Theory", link: "/theory/concepts" },
      { text: "References", link: "/references/implementations" },
      { text: "Evidence", link: "/evidence/methodology" }
    ],
    sidebar: [
      {
        text: "Theory",
        items: [
          { text: "Concepts", link: "/theory/concepts" },
        ]
      },
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

