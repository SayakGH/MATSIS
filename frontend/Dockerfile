FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Build the static production bundle
ARG VITE_BACKEND_URL=http://backend:8000
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}
RUN npm run build

# --- Production stage with Nginx ---
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Nginx config: serve static files and reverse-proxy /api + /ws to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
