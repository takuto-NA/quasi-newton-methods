import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";
import mathjax3 from "markdown-it-mathjax3";

export default withMermaid(defineConfig({
  title: "quasi-newton-methods",
  description: "Evidence-first implementation and verification of quasi-Newton methods (BFGS / L-BFGS / L-BFGS-B)",
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
      { text: "References", link: "/references/papers" },
      { text: "Evidence", link: "/evidence/methodology" }
    ],
    sidebar: [
      {
        text: "Theory",
        items: [
          { text: "Overview", link: "/theory/concepts" },
          { text: "BFGS", link: "/theory/bfgs" },
          { text: "L-BFGS", link: "/theory/lbfgs" },
          { text: "L-BFGS-B", link: "/theory/lbfgsb" }
        ]
      },
      {
        text: "References",
        items: [
          { text: "Papers & Books", link: "/references/papers" },
          { text: "Implementation Comparison (BFGS / L-BFGS)", link: "/references/implementation_comparison" },
          { text: "External Implementations", link: "/references/implementations" },
          { text: "SciPy", link: "/references/scipy" },
          { text: "Articles", link: "/references/articles" }
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
}));
