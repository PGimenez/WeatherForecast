# Build stage
FROM node:18-alpine as build-stage

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy project files
COPY . .

# Build the project with environment variables
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from build stage
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Copy custom nginx config to serve on port 8000
RUN echo 'server { \
    listen 8000; \
    location / { \
        root /usr/share/nginx/html; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

# Expose port 8000
EXPOSE 8000

CMD ["nginx", "-g", "daemon off;"] 