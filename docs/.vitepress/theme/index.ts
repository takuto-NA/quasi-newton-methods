import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import OptimizerVisualizer from "./components/OptimizerVisualizer.vue";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("OptimizerVisualizer", OptimizerVisualizer);
  },
} satisfies Theme;
