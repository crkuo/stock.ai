import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  css: {
    preprocessorOptions: {
      less: {
        javascriptEnabled: true,
        modifyVars: {
          // 這裡設定 Ant Design 主題變數 👇
          "primary-color": "#00bcc2",
          "link-color": "#00bcc2",
          "border-radius-base": "6px",
          "font-size-base": "14px",
        },
      },
    },
  },
})
