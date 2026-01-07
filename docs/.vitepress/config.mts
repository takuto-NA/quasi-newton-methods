import { defineConfig } from "vitepress";
import mathjax3 from "markdown-it-mathjax3";

export default defineConfig({
  title: "quasi-newton-methods",
  description: "準ニュートン法（BFGS / L-BFGS / L-BFGS-B）の Evidence-first 実装と検証",
  base: "/quasi-newton-methods/",
  markdown: {
    config: (md) => {
      md.use(mathjax3);
    },
  },
  themeConfig: {
    nav: [
      { text: "ホーム", link: "/" },
      { text: "理論", link: "/theory/concepts" },
      { text: "出典", link: "/references/papers" },
      { text: "検証", link: "/evidence/methodology" }
    ],
    sidebar: [
      {
        text: "理論",
        items: [
          { text: "理論まとめ", link: "/theory/concepts" },
        ]
      },
      {
        text: "出典",
        items: [
          { text: "論文・教科書", link: "/references/papers" },
          { text: "実装比較（BFGS / L-BFGS）", link: "/references/implementation_comparison" },
          { text: "実装（外部）", link: "/references/implementations" },
          { text: "SciPy", link: "/references/scipy" },
          { text: "記事", link: "/references/articles" }
        ]
      },
      {
        text: "検証（Evidence）",
        items: [
          { text: "方法（Methodology）", link: "/evidence/methodology" },
          { text: "ベースライン結果", link: "/evidence/baseline_results" }
        ]
      }
    ]
  }
});

