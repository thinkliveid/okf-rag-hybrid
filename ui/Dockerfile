FROM nginx:1.27-alpine

ENV API_UPSTREAM=http://api:8000
ENV NGINX_ENVSUBST_FILTER=API_UPSTREAM

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY index.html style.css app.js /usr/share/nginx/html/
